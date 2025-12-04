import glob
import os
import os.path as osp
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from libzhifan.geometry import visualize_mesh
from PIL import Image
from pytorch3d.transforms import rotation_conversions as rotcvt
from pytorch3d.transforms import so3_rotation_angle

from code_arctic.data_reader_onthefly import SeqReaderOnTheFly
from code_arctic.fragment import fragmentation_fps
from code_arctic.mesh_interaction import (mesh_distance_approx,
                                          single_mesh_distance_approx)
from code_arctic.video_subtitle import frame_subtitle, video_subtitle

LEFT = 'left'
RIGHT = 'right'


def relative_poses(T1, T2):
    """
    return T12, where T2 = T12 @ T1 
    """
    return torch.einsum('aij,bjk->abik', T2, torch.inverse(T1))


def relative_metrics(T1, T2):
    """ 
    Args:
        T1: (N, 4, 4)
        T2: (M, 4, 4)
    Returns:
        rel_angles: (N, M) in deg
        rel_distance: (N, M) in meter
    """
    rel_poses = relative_poses(T1, T2)  # (N, M, 4, 4)
    rel_rot = rel_poses[..., :3, :3]
    rel_transl = rel_poses[..., :3, -1]
    _n, _m, _, _ = rel_rot.shape
    # rel_axisang = rotcvt.matrix_to_axis_angle(rel_rot.reshape(-1, 3, 3)).reshape(_n, _m, 3)
    # rel_ang = rel_axisang.norm(dim=-1) / np.pi * 180 # degree
    rel_ang = so3_rotation_angle(rel_rot.reshape(_n*_m, 3, 3)).reshape(_n, _m) / np.pi * 180
    rel_dis = rel_transl.norm(dim=-1) # meter
    return rel_ang, rel_dis


def relative_angles(R1, R2):
    """
    Args:
        R1: (N, 3, 3)
        R2: (M, 3, 3)
    Returns:
        angles: (N, M) in degrees. Returns angle from 2->1 (hence N first then M)
    """
    N = len(R1)
    M = len(R2)
    rel_rots = torch.matmul(
        R1.reshape(N, 1, 3, 3).permute(0, 1, 3, 2),
        R2.reshape(1, M, 3, 3))
    return so3_rotation_angle(rel_rots.reshape(N*M, 3, 3)).reshape(N, M) / np.pi * 180


# def concat_segments(segments, frame_threshold=2, ST=0, ED=1):
#     segments = deepcopy(segments)
#     if not segments:
#         return []
#     # Sort the segments based on their start times
#     segments.sort(key=lambda x: x[ST])
#     merged_segments = [segments[0]]  # Initialize with the first segment
#     for segment in segments[1:]:
#         # Check if the current segment can be merged with the last merged segment
#         if segment[ST] - merged_segments[-1][ED] <= frame_threshold:
#             # Merge the segments by extending the end time of the last merged segment
#             merged_segments[-1][ED] = max(segment[ED], merged_segments[-1][ED])
#         else:
#             # Add the current segment as a new merged segment
#             merged_segments.append(segment)
#     return merged_segments


# class GraspFinderContext(SeqReader):
class GraspFinderContext(SeqReaderOnTheFly):

    def __init__(self, *args, **kwargs):
        super(GraspFinderContext, self).__init__(*args, **kwargs)
        self.images = list(map(
            lambda x:
                np.asarray(Image.open(x)),
            sorted(glob.glob(f'./arctic_data/cropped_images/{self.sid}/{self.seq_name}/0/*.jpg'))
        ))

    def __repr__(self):
        return f"GraspFinderContext({self.sid}/{self.seq_name})"

    def set_current_hand(self, side):
        self.current_hand = side
        T_o2l, T_o2r = self.pose_obj2hand()
        if side == 'left':
            T_o2h = T_o2l
        else:
            T_o2h = T_o2r
        self.angs, self.dists = relative_metrics(T_o2h, T_o2h)

        N = self.num_frames
        tips = self.hand_tips(side, space='ego')
        tip_diff = tips.reshape(N, 1, 21, 3) - tips.reshape(1, N, 21, 3)  # N, N, 21, 3
        tip_dists = (tip_diff**2).sum(dim=(-2, -1)).sqrt()  # N, N
        self.tip_dists = tip_dists
        
    def vert_fragments(self, frag_ratio=0.1):
        """
        Returns:
            frag_mapping: (V,) list mapping each vertex to fragment id
        """
        vo = self.obj_verts(frame_idx=0, space='ego').cpu().numpy()
        _, frag_mapping = fragmentation_fps(vo, int(len(vo) * frag_ratio))
        return frag_mapping
    
    """ V2: use contact map """
    def calc_obj_contact_indices(self, 
                                 side, 
                                 along_normal=True, 
                                 frag_ratio=1.0,
                                 contact_thr=0.0,
                                 ret_dist=False):
        """
        Args:
            along_normal: if True, the vh-to-vo distance is computed considering the normal direction
                otherwise is computed use absolute distance (longer than along_normal)
            frag_ratio: if less than 1.0, process vertices into fragments (face can be considered as fine-grained fragments)

        Returns: list, length == num_frame, each is a set of hand vertex indices
        """
        device = 'cuda'
        faces_h = torch.as_tensor(self.fl) if side == LEFT else torch.as_tensor(self.fr)
        faces_o = torch.as_tensor(self.fo)
        faces_h = faces_h.to(device)
        faces_o = faces_o.to(device)

        indices_list = []
        vh = self.hand_verts(None, side, space='ego').to(device)
        vo = self.obj_verts(None, space='ego').to(device)
        _, d2 = mesh_distance_approx(vh, vo, faces_h, faces_o, along_normal=along_normal)
        for f in range(self.num_frames):
            nz = torch.nonzero(d2[f] < contact_thr).view(-1)
            indices_list.append(set(nz.tolist()))
        if frag_ratio < 1.0:
            # map_to_frag_ind(ind_list, v_frag_ids)
            frag_mapping = self.vert_fragments(frag_ratio)
            indices_list = [set(map(lambda x: frag_mapping[x], v)) for v in indices_list]
        if ret_dist:
            return indices_list, d2
        return indices_list
    
    def compute_indices_ious(self, inds_list):
        """
        Args:
            inds_list: list of N sets
        
        Returns:
            ious: (N, N) iou 
        """
        N = self.num_frames
        ious = np.empty([N, N], dtype=np.float32)
        for i in range(N):
            for j in range(i, N):
                ind_i = inds_list[i]
                ind_j = inds_list[j]
                inter = ind_i.intersection(ind_j)
                union = ind_i.union(ind_j)
                ious[i, j] = len(inter) / (len(union) + 1e-5)
                ious[j, i] = ious[i, j]
        return ious

    def find_stable_segments_v2_obj(self, 
                                    iou_thr: float,
                                    frag_ratio: float,
                                    along_normal=True, 
                                    contact_thr=0.0,
                                    # num_concat_thr=2,
                                    debug=False):
        """ Find stable grasp segments by Stable Contact Area 
        s.t. any two frames have at least XX % percent IoU overlap.
        Area is define as group (fragments) of vertices.
        Contact is defined as vertices (or any vertices in the fragments) having signed-distance less than contact_thr.
        Segments with less than 10 frames won't be kept.

        Implementation: we greedily find such fragments from longest one.
        Implementation-2: we use contact_thr=0.0 i.e only obj vert goes into the hand will be considered.
        Implementation-3: ideally, frag_ratio should take into account the physical area computed by this frag_ratio; we use a fixed one.
        Implementation-4: don't concat! otherwise they will violate the rules. Respect those noise!

        Args:
            iou_thr: negative number. make sure any two frames have at least XX percent IoU
                in terms inf vertices / fragments
            frag_ratio: float in [0.0, 1.0]. If 1.0, no fragmation will be performed.
            along_normal: I haven't investigate this, but since contact_thr is set to 0, only the sign matters

            # num_concat_thr: if two segments are YY frames apart, merge them
            #     e.g. [501, 537] and [539, 550] will be concatenated
        """
        MIN_LEN = 10
        l_ind_list = self.calc_obj_contact_indices(
            LEFT, along_normal=along_normal, frag_ratio=frag_ratio, contact_thr=contact_thr)
        l_ious = self.compute_indices_ious(l_ind_list)
        l_segs = max_length_under_constraint([-l_ious], [iou_thr], MIN_LEN)
        r_ind_list = self.calc_obj_contact_indices(
            RIGHT, along_normal=along_normal, frag_ratio=frag_ratio, contact_thr=contact_thr)
        r_ious = self.compute_indices_ious(r_ind_list)
        r_segs = max_length_under_constraint([-r_ious], [iou_thr], MIN_LEN)

        self.set_current_hand(LEFT)
        l_segs = [[LEFT, s, e, self.angs[s:e, s:e].max().item()] for s, e in sorted(l_segs)]
        # l_segs = concat_segments(l_segs, frame_threshold=num_concat_thr, ST=1, ED=2)

        self.set_current_hand(RIGHT)
        r_segs = [[RIGHT, s, e, self.angs[s:e, s:e].max().item()] for s, e in sorted(r_segs)]
        # r_segs = concat_segments(r_segs, frame_threshold=num_concat_thr, ST=1, ED=2)

        if debug:
            return l_segs + r_segs, l_ious, r_ious
        return l_segs + r_segs
    
    """ Visualize utils """
    def max_angle_achiver(self, i, j, side):  # typo: achiever
        self.set_current_hand(side)
        angs = self.angs[i:j, i:j]
        ind = torch.argmax(self.angs[i:j, i:j], keepdim=False)
        m = ind // angs.shape[0]
        n = ind % angs.shape[0]
        return i + m.item(), i + n.item(), angs[m, n].item()

    def plot_dist_matrix(self, i, j):
        plt.imshow(self.dists[i, j])
        plt.colorbar()
        plt.show()

    """ print obj rel angle and dis w.r.t to first world frame """
    def make_compare_scene(self, i, j, side):
        hm_i = self.hand_verts(i, side, space=side,as_mesh=True)
        om_i = self.obj_verts(i, space=side, as_mesh=True)
        hm_j = self.hand_verts(j, side, space=side,as_mesh=True)
        om_j = self.obj_verts(j, space=side, as_mesh=True, tex_color='yellow')
        hm_j.apply_translation_([0, 0.25, 0])
        om_j.apply_translation_([0, 0.25, 0])
        return visualize_mesh([hm_i, hm_j, om_i, om_j], show_axis=True, viewpoint='pytorch3d')

    def make_compare_frame(self, i, j) -> np.ndarray:
        images = self.images
        img1 = images[i]
        img2 = images[j]
        img = np.hstack([img1, img2])
        d_ang = self.angs[i, j].item()
        d_transl = self.dists[i, j].item() * 100
        sub = f"frame ({i}->{j}) {d_ang:.03f} deg, {d_transl:.03f} cm"
        return frame_subtitle(img, sub)
    
    def make_video(self, st_frame):
        subtitles = []
        for i in range(len(self.angs)):
            sub = f"frame {i}, {self.angs[st_frame, i].item():.02f} deg, {self.dists[st_frame, i].item() * 100:.02f} cm"
            subtitles.append(sub)
        clip = video_subtitle(self.images[st_frame:], subtitles[st_frame:], fps=20)
        clip.write_videofile('/tmp/tmp.mp4')
    
    def generate_segments_videos(self, segments, out_dir, with_rend=False):
        """
        Args:
            out_dir: e.g. 'outputs/arctic_stable'
        """
        os.makedirs(out_dir, exist_ok=True)
        for hand, s, e, *rest in tqdm.tqdm(segments):
            frames = []
            subtitles = []
            for f in range(s, e):
                frame = self.render_image_mesh(f, with_rend=with_rend, with_triview=True,
                                               side=hand) * 255
                frames.append(frame)
                subtitles.append(f'{f}')

            fname = f'{self.sid}_{self.seq_name}_{hand}_{s}_{e}.mp4'
            clip = video_subtitle(frames, subtitles, fps=15)
            clip.write_videofile(osp.join(out_dir, fname), logger=None, verbose=False)
    

    # def find_stable_segments_v1(self):
    #     """
    #     Returns: list of ['left'/'right', start_frame, end_frame]
    #     """
    #     angle_thr = 30  # 30 degrees
    #     transl_thr = 0.03  # 0.03 meter == 3 cm
    #     MIN_SEGMENT_LEN = 10
    #     CONTACT_THR = 0.005  # 0.5cm

    #     faces_l = torch.as_tensor(self.fl)
    #     faces_r = torch.as_tensor(self.fr)
    #     faces_o = torch.as_tensor(self.fo)
    #     results = []

    #     self.set_current_hand(LEFT)
    #     segments = max_length_under_constraint(
    #         [self.angs, self.dists], [angle_thr, transl_thr], min_len=MIN_SEGMENT_LEN)
    #     for s, e in segments:
    #         vh = self.hand_verts(s, LEFT, space='ego')
    #         vo = self.obj_verts(s, space='ego', as_mesh=False)
    #         d1, d2 = single_mesh_distance_approx(vh, vo, faces_l, faces_o)
    #         if torch.min(d1) < CONTACT_THR:
    #             results.append([LEFT, s, e])

    #     self.set_current_hand(RIGHT)
    #     segments = max_length_under_constraint(
    #         [self.angs, self.dists], [angle_thr, transl_thr], min_len=MIN_SEGMENT_LEN)
    #     for s, e in segments:
    #         vh = self.hand_verts(s, RIGHT, space='ego')
    #         vo = self.obj_verts(s, space='ego', as_mesh=False)
    #         d1, d2 = single_mesh_distance_approx(vh, vo, faces_r, faces_o)
    #         if torch.min(d1) < CONTACT_THR:
    #             results.append([RIGHT, s, e])
    #     return results

    def calc_hand_contact_indices(self, side, contact_thr=0.00):
        """
        Returns: list, length == num_frame, each is a set of hand vertex indices
        """
        device = 'cuda'
        faces_h = torch.as_tensor(self.fl) if side == LEFT else torch.as_tensor(self.fr)
        faces_o = torch.as_tensor(self.fo)
        faces_h = faces_h.to(device)
        faces_o = faces_o.to(device)

        indices_list = []

        vh = self.hand_verts(None, side, space='ego').to(device)
        vo = self.obj_verts(None, space='ego').to(device)
        d1, _ = mesh_distance_approx(vh, vo, faces_h, faces_o)
        for f in range(self.num_frames):
            nz = torch.nonzero(d1[f] < contact_thr).view(-1)
            indices_list.append(set(nz.tolist()))
        return indices_list

    # def find_stable_segments_v2(self, iou_thr=-0.3):
    #     MIN_LEN = 10
    #     IOU_THR = iou_thr
    #     ind_list = self.calc_contact_indices(LEFT)
    #     ious = self.compute_indices_ious(ind_list)
    #     l_segs = max_length_under_constraint([-ious], [IOU_THR], MIN_LEN)
    #     ind_list = self.calc_contact_indices(RIGHT)
    #     ious = self.compute_indices_ious(ind_list)
    #     r_segs = max_length_under_constraint([-ious], [IOU_THR], MIN_LEN)

    #     l_segs = [[LEFT, s, e] for s, e in sorted(l_segs)]
    #     r_segs = [[RIGHT, s, e] for s, e in sorted(r_segs)]
    #     return l_segs + r_segs


def max_length_under_constraint(err_list, thr_list, min_len):
    """
    Find segments I1, I2,...
    s.t. for any err in err_list, and the corresponding thr in thr_list,
    any p,q in I, err[p, q] < thr and |I| is max.

    Args:
        err_list: list of multiple (N, N) error correlation matrix
        thr_list: multiple threshold to be satisfied
    
    Returns:
        segments: list of [start, end)
    """
    def get_satisfied_segments(err_list, thr_list, min_len):
        """ See max_length_under_constraint for arguments
        """
        num_criteria = len(thr_list)
        N = len(err_list[0])
        assert len(err_list) == len(thr_list)
        for k in range(num_criteria):
            assert err_list[k].shape == err_list[0].shape, \
                f"Got {err_list[k].shape} != {err_list[0].shape}"

        segments = []
        for i in range(N):
            e = 1e8
            for k in range(num_criteria):
                err = np.asarray(err_list[k])
                thr = thr_list[k]
                _, jj = np.nonzero(err[[i], i:] > thr)
                if len(jj)  == 0:  # last segment
                    e = min(e, N)
                else:
                    e = min(e, i + jj[0])
            segment = [int(i), int(e)]  # longest possible
            segments.append(segment)
        
        segments = [v for v in segments if v[1] - v[0] >= min_len]
        return segments

    segments = get_satisfied_segments(err_list, thr_list, min_len)

    # Now that we have N segments, greedily find longest non-overlapping segments
    # Here's a brute-force implementation,
    # Can use Heap + Segment Tree to achieve O(NlgN) complexity
    results = []
    while len(segments) > 0:
        max_seg = [-1, -1, -1]
        for i, j in segments:
            if j - i > max_seg[0]:
                max_seg = [j-i, i, j]
        _, s, e = max_seg
        segments.pop( segments.index([s, e]))
        results.append([s, e])
        segments = [[i, j] for i, j in segments if j <= s or i >= e]
    return results


def max_length_under_constraint_test():
    err = np.float32([
        [0, 0, 0, 2, 3],
        [0, 0, 0, 2, 4],
        [0, 0, 0, 3, 2],
        [3, 3, 2, 0, 0],
        [3, 3, 3, 0, 0]
    ])
    exp = [
        [0, 3], [3, 6]
    ]
    out = max_length_under_constraint([err], [1.0], min_len=0)
    print("Expect:", exp) 
    print("Got: ", out)


def find_all_stable_grasp(out_dir, iou_thr, frag_ratio, generate_video=True):
    """
    """
    all_sid_seqs = []  # (sid, seq_name)
    for sid in os.listdir('arctic_outputs/processed_verts/seqs'):
        sid_dir = os.path.join('arctic_outputs/processed_verts/seqs', sid)
        for seq_name in os.listdir(sid_dir):
            all_sid_seqs.append((sid, seq_name))
    
    rows = []
    for sid, seq_name in tqdm.tqdm(sorted(all_sid_seqs)):
        seq_data_path = os.path.join('arctic_outputs/processed_verts/seqs', sid, seq_name)
        # seq_data = np.load(seq_data_path, allow_pickle=True).item()
        obj_name = seq_name.split('_')[0]
        seq_name_base = seq_name.replace('.npy', '')
        # ctx = GraspFinderContext(seq_data, sid, seq_name_base, obj_name=obj_name)
        ctx = GraspFinderContext(
            sid, seq_name_base, obj_name=obj_name, obj_version='original', preload_obj_faces=True)
        segments = ctx.find_stable_segments_v2_obj(
            iou_thr=iou_thr, frag_ratio=frag_ratio)
        if generate_video:
            ctx.generate_segments_videos(segments, out_dir=out_dir)
        sid_seq_name = f'{sid}/{seq_name_base}'
        for side, s, e, angle_diff in segments:
            rows.append([sid_seq_name, side, s, e, angle_diff])

        df = pd.DataFrame(
            rows, columns=['sid_seq_name', 'side', 'start', 'end', 'angle_diff'])
        df.to_csv('arctic_outputs/stable_grasps_v3_frag.csv', index=False)
        

if __name__ == '__main__':
    # max_length_under_constraint_test()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--csv_only', default=False, action='store_true')
    args = parser.parse_args()

    generate_video = False if args.csv_only else True

    find_all_stable_grasp(
        out_dir='outputs/find_stable_frag001_IoU50',
        iou_thr=-0.50,
        frag_ratio=0.01,
        generate_video=generate_video
    )