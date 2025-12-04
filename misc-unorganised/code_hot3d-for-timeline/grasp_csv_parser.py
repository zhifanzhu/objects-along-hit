import os
import numpy as np
import os.path as osp
import pandas as pd
from code_arctic.stable_grasps_finder import max_length_under_constraint
from libzhifan import io
# from code_hot3d.grasp_csv_parser import get_phase_segs_v2


GRASP_INFO_DIR = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information'
# existing_vis_dir = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/visualisation'
LEFT = 'left'
RIGHT = 'right'


def get_grasp_df(seq_name: str):
    return pd.read_csv(osp.join(GRASP_INFO_DIR, seq_name) + '.csv')


def get_phase_segs(seq_name: str):
    """
    Returns:
        all_incontact: dict
            key: category
            value: list of tuples [start_frame, end_frame, side]
        all_statics: dict. This is the complement set of all_incontact
            key: category
            value: list of tuples [start_frame, end_frame]
    """

    def add_num_contact(df):
        """ add number of in-contact vertices on the object """
        df['left_num_contact'] = df.apply(
            lambda row: len(row['left_overlap'][1:-1].strip().split()), axis=1)
        df['right_num_contact'] = df.apply(
            lambda row: len(row['right_overlap'][1:-1].strip().split()), axis=1)
        return df

    df = pd.read_csv(osp.join(GRASP_INFO_DIR, seq_name) + '.csv')
    df = add_num_contact(df)
    cats = set(df['right_object']).union(set(df['left_object']))
    cats = [cat for cat in cats if type(cat) != float]  # remove nan
    all_incontact = {cat: [] for cat in cats}
    for cat in cats:
        for side in [LEFT, RIGHT]:
            st = None
            for i, row in df.iterrows():
                if row[f'{side}_object'] == cat:
                    if row[f'{side}_num_contact'] > 0 and st is None:
                        st = i
                    if row[f'{side}_num_contact'] == 0 and st is not None:
                        all_incontact[cat].append([st, i-1, side])
                        st = None
        all_incontact[cat] = sorted(all_incontact[cat], key=lambda x: x[0])

    num_frames = len(df)
    all_statics = {cat: [] for cat in cats}
    for cat in cats:
        incontact_segs = all_incontact[cat]
        static_segs = []
        # Get the complement set of incontact_segs
        if len(incontact_segs) == 0:
            static_segs.append([0, num_frames-1])
        else:
            if incontact_segs[0][0] != 0:
                static_segs.append([0, incontact_segs[0][0]-1])
            for i in range(len(incontact_segs)-1):
                static_segs.append([incontact_segs[i][1]+1, incontact_segs[i+1][0]-1])
            if incontact_segs[-1][1] != num_frames-1:
                static_segs.append([incontact_segs[-1][1]+1, num_frames-1])
        all_statics[cat] = static_segs
    
    return all_incontact, all_statics


def compute_inview_static_segs(
        static_segs: list, 
        inview_segs: list,
        min_len):
    """ Intersection of static_segs and inview_segs on a given object category

    Args:
        static_segs: list of tuples [start_frame, end_frame]
        inview_segs: list of tuples [start_frame, end_frame]
        min_len: int. For 30FPS, 10 frames is 1/3 second
    """
    def seg_intersect(seg1, seg2):
        """ Return the intersection of two segments """
        s = max(seg1[0], seg2[0])
        e = min(seg1[1], seg2[1])
        if s > e:
            return None
        return [s, e]

    satisfied_segs = []
    for static_seg in static_segs:
        for inview_seg in inview_segs:
            inter = seg_intersect(static_seg, inview_seg)
            if inter is not None:
                satisfied_segs.append(inter)
        
    # Filter out short segments
    satisfied_segs = [seg for seg in satisfied_segs if seg[1]-seg[0]+1 >= min_len]

    return satisfied_segs


class StatefulPhaseParser:

    def __init__(self, seq_df):
        self.df = pd.read_csv(seq_df)
        self.seq_name = seq_df.split('/')[-1].split('.')[0]
        self.num_frames = len(self.df)
        cats = set(self.df['right_object']).union(set(self.df['left_object']))
        cats = [cat for cat in cats if type(cat) != float]  # remove nan
        self.cats = cats

    def parse(self, clean_overlapping=True):
        self._all_statics = self._calc_all_statics()
        self._all_sg_segs = self._calc_all_sg()
        self._all_dynamics = self._calc_all_dynamics(self._all_statics, self._all_sg_segs)
        timelines = self._combine_segments(
            [self._all_statics, self._all_sg_segs, self._all_dynamics])
        
        if clean_overlapping:
            timelines = self._cleanup_overlapping(timelines)
        
        # timelines = self._break_at_static(timelines)
        return timelines
    
    def parse_to_json(self):
        timelines = self.parse(clean_overlapping=True)
        return self.to_json_timelines(timelines)

    def _calc_all_statics(self):
        """
        Returns:
            all_statics: dict
                key: category
                value: list of tuples [start_frame, end_frame, 'scene_static']
        """
        df = self.df
        all_sg_segs = {}
        for cat in self.cats:
            all_sg_segs[cat] = []
            st = None
            for i, row in df.iterrows():
                has_contact = False
                for side in [LEFT, RIGHT]:
                    if row[f'{side}_object'] == cat:
                        raw_ov = row[f'{side}_overlap'][1:-1].strip().split()
                        if len(raw_ov) > 0:
                            has_contact = True
                            break
                if has_contact:
                    if st is not None:
                        all_sg_segs[cat].append([st, i-1, 'scene_static'])
                        st = None
                else:
                    if st is None:
                        st = i
            if st is not None:
                all_sg_segs[cat].append([st, self.num_frames-1, 'scene_static'])
        return all_sg_segs

    def _calc_all_sg(self):
        """ Calculate all stable grasp segments """
        df = self.df
        all_sg_segs = {}
        for cat in self.cats:
            all_sg_segs[cat] = []
            for side in [LEFT, RIGHT]:
                st = None
                for i, row in df.iterrows():
                    if row[f'{side}_object'] == cat and row[f'{side}_grasp'] == True:
                        in_sg = True
                    else:
                        in_sg = False
                    if in_sg:
                        if st is None:
                            st = i
                    else:  # outside
                        if st is not None:
                            all_sg_segs[cat].append([st, i-1, side])
                            st = None
        return all_sg_segs

    def _calc_all_dynamics(self, all_statics, all_sg_segs):
        """
        dynamic segments are complementary to static and sg segments.
        Note: we force continuity of dynamic segments.
        """
        all_dynamics = {}
        for cat in self.cats:
            all_dynamics[cat] = []
            existing_segs = all_statics[cat] + all_sg_segs[cat]
            existing_segs = sorted(existing_segs, key=lambda x: x[0])

            running_end = 0
            if existing_segs[0][0] != 0:
                all_dynamics[cat].append([0, existing_segs[0][0], 'scene_dynamic'])
                running_end = existing_segs[0][1]
            else:
                running_end = existing_segs[0][1]  # I assume the second segment doesn't start at 0

            for i in range(1, len(existing_segs)):
                if running_end < existing_segs[i][0]:
                    all_dynamics[cat].append([running_end, existing_segs[i][0], 'scene_dynamic'])
                    running_end = existing_segs[i][1]
                else:
                    running_end = max(existing_segs[i][1], running_end)
            if existing_segs[-1][1] != self.num_frames-1:
                all_dynamics[cat].append([existing_segs[-1][1], self.num_frames-1, 'scene_dynamic'])
        return all_dynamics

    def _combine_segments(self, seg_dict_list):
        """ Combine all segments and sort them """
        all_timelines = {cat: [] for cat in self.cats}
        for cat in self.cats:
            all_segs = []
            for seg_dict in seg_dict_list:
                all_segs += seg_dict[cat]
            all_segs = sorted(all_segs, key=lambda x: x[0])
            all_timelines[cat] = all_segs
        return all_timelines

    def _break_at_static(self, timelines):
        """
        break one single timeline of A->B->C->C...A->B...
        into [A->...->A, A->...->A, ...]
        where A is scene_static, C is inhand
        """
        pass

    def check_overlap(self, timelines):
        """ Check if the timeline is valid """
        for cat in self.cats:
            all_segs = timelines[cat]
            for i in range(len(all_segs)-1):
                if all_segs[i][1] > all_segs[i+1][0]:
                    print(f"Overlap found: {cat=} {all_segs[i]} and {all_segs[i+1]}")
    
    def _cleanup_overlapping(self, timelines):
        """
        Shrink non-trivial overlapping of inhand.
        Remove if it's completely inside.
        """
        new_timelines = {}
        for cat in self.cats:
            all_segs = timelines[cat]
            new_segs = [all_segs[0]]
            for i in range(len(all_segs)-1):
                if all_segs[i+1][0] < all_segs[i][1]: # case: overlap
                    if all_segs[i][2] not in {'left', 'right'} \
                        and all_segs[i+1][2] not in {'left', 'right'}:
                        raise ValueError("Not allowed situation!")
                    print(f"Shrink overlap: {cat=} {all_segs[i]} and {all_segs[i+1]}")
                    all_segs[i+1][0] = all_segs[i][1]
                    if all_segs[i+1][0] <= all_segs[i+1][1]:
                        new_segs.append(all_segs[i+1])
                else:
                    new_segs.append(all_segs[i+1])
            new_timelines[cat] = new_segs
        return new_timelines
    
    def to_json_timelines(self, timelines) -> dict:
        """ ready to be used by the POTIM """
        json_timelines = []
        for cat in self.cats:
            tl = {}
            total_start = timelines[cat][0][0]
            total_end = timelines[cat][-1][1] + 1
            tl['total_start'] = total_start
            tl['total_end'] = total_end
            tl['timeline_name'] = f"{self.seq_name}_{cat}_{total_start:05d}_{total_end:05d}"
            tl['seq_name'] = self.seq_name
            tl['segments'] = []
            tl['cat'] = cat
            for raw_seg in timelines[cat]:
                seg = {
                    'st': raw_seg[0],
                    'ed': raw_seg[1],
                    'side': raw_seg[2] if raw_seg[2] in {'left', 'right'} else None,
                    'ref': 'inhand' if raw_seg[2] in {'left', 'right'} else raw_seg[2]
                }
                tl['segments'].append(seg)
            json_timelines.append(tl)
        return json_timelines


def get_phase_segs_v2(seq_name: str):
    """
    Returns:
        dict:
            key: category
            value: list of tuples [start_frame, end_frame, side, ref]
                ref is one of ['scene_static', 'scene_dynamic', 'inhand']
    """
    raise NotImplementedError
    def compute_indices_ious(inds_list):
        """
        Args:
            inds_list: list of N sets
        
        Returns:
            ious: (N, N) iou 
        """
        N = len(inds_list)
        ious = np.empty([N, N], dtype=np.float32)
        for i in range(N):
            # Speedup: if ind_i is empty, then ious[i, j] = 0
            if len(inds_list[i]) == 0:
                ious[i, :] = 0
                continue
            for j in range(i, N):
                ind_i = inds_list[i]
                ind_j = inds_list[j]
                inter = ind_i.intersection(ind_j)
                union = ind_i.union(ind_j)
                ious[i, j] = len(inter) / (len(union) + 1e-5)
                ious[j, i] = ious[i, j]
        return ious

    # Step-1: get all incontact obj-verts 
    all_incontact_verts = {
        cat: {
            LEFT: [set() for _ in range(num_frames)], 
            RIGHT: [set() for _ in range(num_frames)]
            } for cat in cats}
    for cat in cats:
        for side in [LEFT, RIGHT]:
            for i, row in df.iterrows():
                if row[f'{side}_object'] == cat:
                    raw_ov = row[f'{side}_overlap'][1:-1].strip().split()
                    all_incontact_verts[cat][side][i] = \
                        set(map(int, raw_ov))
    print("Done with incontact verts")
    
    # Step-2: find the global argmax stable grasp
    all_sg_segs = {}
    iou_thr = -0.5
    MIN_LEN = 10
    for cat in cats:
        for side in [LEFT, RIGHT]:
            print(f"Computing {cat} {side}")
            ious = compute_indices_ious(all_incontact_verts[cat][side])
            sg_segs = max_length_under_constraint([-ious], [iou_thr], MIN_LEN)
            all_sg_segs[(cat, side)] = sg_segs

    print("Done with SG segs")
    return all_sg_segs

    all_timelines = {cat: [] for cat in cats}


if __name__ == '__main__':
    seq_name = 'P0018_a082e8a6'
    seq_name = 'P0002_016222d1'
    df = pd.read_csv(f'/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information/{seq_name}.csv')
    parser = StatefulPhaseParser(seq_name)
    json_tl = parser.parse_to_json()
    io.write_json(json_tl, f'./code_hot3d/image_sets/{seq_name}_longest.json', indent=2)