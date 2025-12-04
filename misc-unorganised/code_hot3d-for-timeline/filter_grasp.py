
import os
import pandas as pd
from tqdm import tqdm

from hot3d_constants import DATA_CONSTANTS

sequences = os.listdir(DATA_CONSTANTS['grasp_information_save_dir'])
sequences = [seq for seq in sequences if seq.startswith('P')]


def mark_consecutive_true(df, column_name, threshold=10):
    df[f'{column_name}_smooth'] = False
    bool_series = df[column_name]
    start_index = -1
    consecutive_count = 0
    for i in range(len(bool_series)):
        if bool_series[i] == True:
            if consecutive_count == 0:
                start_index = i
            consecutive_count += 1
        else:
            if consecutive_count >= threshold:
                df.loc[start_index:i-1, f'{column_name}_smooth'] = True
            consecutive_count = 0
            start_index = -1
    if consecutive_count >= threshold:
        df.loc[start_index:i, f'{column_name}_smooth'] = True
    return df

for sequence in tqdm(sequences):
    data = pd.read_csv(f"{DATA_CONSTANTS['grasp_information_save_dir']}/{sequence}")
    print(f'{sequence}: {data.shape[0]}')
    mark_consecutive_true(data, 'left_grasp')
    mark_consecutive_true(data, 'right_grasp')
    save_path = f'/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_smooth/{sequence}'
    if os.path.exists(save_path):
        print(f'{sequence} already exists')
        continue
    data.to_csv(save_path, index=False)
