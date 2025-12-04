
import os
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv_dir',
    default='/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_smooth/',
    type=str,
    help='Directory containing the csv files',
)
args = parser.parse_args()

info_files = os.listdir(args.csv_dir)
info_files = [os.path.join(args.csv_dir, f) for f in info_files if f.startswith('P')]

def detect_grasps(df, hand):
    grasp_col = f"{hand}_grasp_smooth"
    object_col = f"{hand}_object"
    
    events = []
    grasping = False
    start_time = None
    current_object = None
    
    for i in range(len(df)):
        if df.iloc[i][grasp_col] == 1 and not grasping:
            # Start of a grasp
            grasping = True
            start_time = df.iloc[i]["timestamp_ns"]
            current_object = df.iloc[i][object_col]
        elif df.iloc[i][grasp_col] == 0 and grasping:
            # End of a grasp
            grasping = False
            end_time = df.iloc[i - 1]["timestamp_ns"]
            duration = end_time - start_time
            # duration in sec
            duration = duration / 1e9
            # if duration < 1: breakpoint()
            events.append((current_object, start_time, end_time, duration))
    
    return events

# Initialize a list to collect all grasp events
all_grasp_events = []

# Process each file
for file in tqdm(info_files):
    # print(file)
    data = pd.read_csv(file)
    right_grasps = detect_grasps(data, 'right')
    left_grasps = detect_grasps(data, 'left')
    grasp_events = right_grasps + left_grasps
    all_grasp_events.extend(grasp_events)

# Convert the collected events into a DataFrame
grasp_df = pd.DataFrame(all_grasp_events, columns=["object", "start_time", "end_time", "duration"])

# Generate the summary
grasp_summary = grasp_df.groupby("object").agg(
    grasp_count=pd.NamedAgg(column="object", aggfunc="count"),
    total_duration=pd.NamedAgg(column="duration", aggfunc="sum"),
    avg_duration=pd.NamedAgg(column="duration", aggfunc="mean"),
    std_duration=pd.NamedAgg(column="duration", aggfunc="std"),
    max_duration=pd.NamedAgg(column="duration", aggfunc="max"),
    durations=pd.NamedAgg(column="duration", aggfunc=list)
).reset_index()

# Display the summary
print(grasp_summary)
