
import os
from tqdm import tqdm
from libzhifan import io

from code_hot3d.grasp_csv_parser import StatefulPhaseParser

folder_path = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2/'
save_path = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2_timelines/'
csvs = [os.path.join(folder_path, item) for item in os.listdir(folder_path) if 'csv' in item]
for csv in tqdm(csvs):
    print(f'Processing {csv}')
    csv_name = os.path.basename(csv).split('.')[0]
    csv_save_path = f'{save_path}/{csv_name}_longest.json'
    if os.path.exists(csv_save_path):
        print(f"Timeline already exists at {csv_save_path}. Skipping...")
        continue
    # if csv_name in ['P0021_7febe2a6', 'P0014_e40eec5d']:
    #     print(f"Skipping {csv_name} due to parallel processing ...")
    #     continue
    parser = StatefulPhaseParser(csv)
    timelines = parser.parse()
    parser.check_overlap(timelines)
    json_tl = parser.parse_to_json()
    io.write_json(json_tl, csv_save_path, indent=2)
    print(f'CSV written to {csv_save_path} ...')
