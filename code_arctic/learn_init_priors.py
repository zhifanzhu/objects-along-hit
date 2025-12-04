import os
import torch
import pandas as pd
import tqdm
from code_arctic.data_reader_onthefly import SeqReaderOnTheFly

from pytorch3d.transforms import rotation_conversions as cvt
from sklearn.cluster import DBSCAN, KMeans


def rot_kmeans_axisang(rot_mats, n_clusters):
    """ Kmeans clustering of rotation matrices via
    axis-angle representation with euclidean distance """
    axisangs = cvt.matrix_to_axis_angle(rot_mats)
    mod = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(axisangs)
    axisangs_cen = mod.cluster_centers_
    rot_fit = cvt.axis_angle_to_matrix(torch.from_numpy(axisangs_cen))
    return rot_fit

LEFT = 'left'
RIGHT = 'right'

def main():
    df = pd.read_csv('DATA_STORAGE/arctic_outputs/stable_grasps_v3_frag.csv')

    T_o2l_avgs = dict()
    T_o2r_avgs = dict()

    for sid_seq_name, subdf in tqdm.tqdm(df.groupby('sid_seq_name')):
        sid, seq_name = sid_seq_name.split('/')
        obj_name = seq_name.split('_')[0]
        if obj_name not in T_o2l_avgs:
            T_o2l_avgs[obj_name] = []
        if obj_name not in T_o2r_avgs:
            T_o2r_avgs[obj_name] = []
        reader = SeqReaderOnTheFly(sid, seq_name, obj_name, 'original')
        vo, fo, T_o2l_neut, T_o2r_neut = reader.neutralized_obj_params()
        for i, row in subdf.iterrows():
            start = row.start
            end = row.end
            if row.side == 'left':
                T_o2l_avg = torch.eye(4)
                T_o2l_avg[:3, :3] = T_o2l_neut[int(start+end)//2, :3, :3]
                T_o2l_avg[:3, -1] = T_o2l_neut[start:end, :3, -1].mean(0)
                T_o2l_avgs[obj_name].append(T_o2l_avg)
            else:
                T_o2r_avg = torch.eye(4)
                T_o2r_avg[:3, :3] = T_o2r_neut[int(start+end)//2, :3, :3]
                T_o2r_avg[:3, -1] = T_o2r_neut[start:end, :3, -1].mean(0)                
                T_o2r_avgs[obj_name].append(T_o2r_avg)

    # T_o2l_avgs = torch.stack(T_o2l_avgs, dim=0)
    # T_o2r_avgs = torch.stack(T_o2r_avgs, dim=0)

    for obj_name in T_o2l_avgs.keys():
        T_o2l_avgs[obj_name] = torch.stack(T_o2l_avgs[obj_name], dim=0)
        T_o2r_avgs[obj_name] = torch.stack(T_o2r_avgs[obj_name], dim=0)


    n_clusters = 10
    neutral_priors = dict()
    for obj_name in T_o2l_avgs.keys():
        R_o2l = rot_kmeans_axisang(T_o2l_avgs[obj_name][:, :3, :3], n_clusters=n_clusters)
        R_o2r = rot_kmeans_axisang(T_o2r_avgs[obj_name][:, :3, :3], n_clusters=n_clusters)
        t_o2l = T_o2l_avgs[obj_name][:, :3, -1].mean(0)
        t_o2r = T_o2r_avgs[obj_name][:, :3, -1].mean(0)
        neutral_priors[obj_name] = dict(
            R_o2l=R_o2l, R_o2r=R_o2r,
            t_o2l=t_o2l, t_o2r=t_o2r)

    os.makedirs('weights/pose_priors/arctic', exist_ok=True)
    cache = dict()
    for k, v in neutral_priors.items():
        cache[f'arctic_{k}'] = v
    # torch.save(cache, f'weights/pose_priors/arctic/neutral_priors_{n_clusters}.pth')

if __name__ == '__main__':
    main()