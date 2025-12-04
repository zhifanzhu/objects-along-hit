import os
import re
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

from hot3d_utils import HOT3DUTILS
from grasp_utils import GRASPUTILS
from hot3d_constants import DATA_CONSTANTS

parser = argparse.ArgumentParser()
parser.add_argument(
    '--closeness_threshold',
    default=0.05,
    type=float,
    help='Only check for overlapping vertices if chamfer distance between hand and closes object is less than closeness_threshold',
)
parser.add_argument(
    '--iou_threshold',
    default=0.5,
    type=float,
    help='Threshold for IOU to check if grasp is stable or not',
)
# Flag for debugging a single sequence
parser.add_argument(
    '--debug_sequence',
    action='store_true',
    help='Flag to debug a single sequence',
)
parser.add_argument(
    '--skip',
    nargs='*',
    required=False,
    help='List of videos to ignore in case processes are running in parallel',
)
parser.add_argument(
    '--machine',
    default='cibo',
    type=str,
    help='Name of the machine where the code is running. Used for diving the load',
)
parser.add_argument(
    '--bottom',
    action='store_true',
    help='If true, run from the back of the sequences',
)
parser.add_argument(
    '--use_approx',
    action='store_true',
    help='If true, use approximate method to get contact points',
)
parser.add_argument(
    '--approx_tight',
    action='store_true',
    help='If true, use trimesh signed distance over approximate method to get contact points',
)
args = parser.parse_args()
print(args)

sequences = os.listdir(DATA_CONSTANTS['dataset_path'])
sequences = [seq for seq in sequences if seq.startswith('P')]
print(f'{len(sequences)} sequences found')
# assignements = pickle.load(open('/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset_quest/assignment.pkl', 'rb'))
# sequences = assignements[args.machine]
if args.bottom:
    sequences = sequences[::-1]

for sequence in sequences:
    if args.skip:
        if sequence in args.skip:
            print(f'{sequence} is being processed parallely...')
            continue
    if DATA_CONSTANTS['preferred_sequence'] != sequence and args.debug_sequence:
        continue
    else:
        print('Processing sequence: ', sequence)
    test_participant = False
    for test_participant_name in DATA_CONSTANTS['test_participants']:
        if test_participant_name in sequence:
            print(f'{sequence} from test set. Skipping...')
            test_participant = True
            break
    if test_participant:
        continue
    info_dataframe_path = f"{DATA_CONSTANTS['grasp_information_save_dir']}/{sequence}.csv"
    seq_info = HOT3DUTILS(sequence=sequence)
    grasp_utils = GRASPUTILS()
    seq_timestamps = seq_info.timestamps
    if os.path.exists(info_dataframe_path):
        # Check if all the timestamps are presnet for the given sequence
        info_dict = pickle.load(open(f'{info_dataframe_path.replace(".csv", ".pkl")}', 'rb'))
        info_dataframe = pd.read_csv(info_dataframe_path)
        if len(seq_timestamps) == info_dataframe.shape[0]:
            print(f'Sequence {sequence} processed completely...')
            continue
        else:
            # We need to start from where we left off.
            old_count = info_dataframe.shape[0]
            timestamp_processed = info_dataframe.iloc[-1]['timestamp_ns']
            grasp_utils.right_overlap_vertices_tracker.push(
                [int(num) for num in re.findall(r'\d+', info_dataframe.iloc[-2]['right_overlap'])]
            )
            grasp_utils.left_overlap_vertices_tracker.push(
                [int(num) for num in re.findall(r'\d+', info_dataframe.iloc[-2]['left_overlap'])]
            )
            grasp_utils.right_overlap_vertices_tracker.push(
                [int(num) for num in re.findall(r'\d+', info_dataframe.iloc[-1]['right_overlap'])]
            )
            grasp_utils.left_overlap_vertices_tracker.push(
                [int(num) for num in re.findall(r'\d+', info_dataframe.iloc[-1]['left_overlap'])]
            )
            grasp_utils.iou_tracker_right.push(info_dataframe.iloc[-2]['right_iou'])
            grasp_utils.iou_tracker_left.push(info_dataframe.iloc[-2]['left_iou'])
            grasp_utils.iou_tracker_right.push(info_dataframe.iloc[-1]['right_iou'])
            grasp_utils.iou_tracker_left.push(info_dataframe.iloc[-1]['left_iou'])
    else:
        info_dict = {}
        info_dataframe = pd.DataFrame(
            columns=[
                'count',
                'timestamp_ns',
                'right_overlap',
                'left_overlap',
                'right_iou',
                'left_iou',
                'right_grasp',
                'left_grasp',
                'right_object',
                'left_object',
            ],
            dtype=object,
        )
        old_count = 0
        timestamp_processed = None
    for count, timestamp_ns in tqdm(
        enumerate(seq_timestamps),
        desc=f'Processing {sequence}',
        total=len(seq_timestamps),
    ):
        if timestamp_processed is not None:
            if timestamp_processed >= timestamp_ns:
                if timestamp_processed == timestamp_ns:
                    print(f'Skipped till {count}: {timestamp_ns}')
                continue
        closest_objects = seq_info.get_closest_objects_v2(timestamp_ns)
        overlapping_object_vertices_right, object_right_name = seq_info.get_contact_points(
            timestamp_ns=timestamp_ns,
            hand_side='right',
            closeness_threshold=args.closeness_threshold,
            closest_objects=closest_objects,
            use_approx=args.use_approx,
            approx_tight=args.approx_tight,
        )
        grasp_utils.right_overlap_vertices_tracker.push(overlapping_object_vertices_right)
        overlapping_object_vertices_left, object_left_name = seq_info.get_contact_points(
            timestamp_ns=timestamp_ns,
            hand_side='left',
            closeness_threshold=args.closeness_threshold,
            closest_objects=closest_objects,
            use_approx=args.use_approx,
            approx_tight=args.approx_tight,
        )
        grasp_utils.left_overlap_vertices_tracker.push(overlapping_object_vertices_left)
        if count >= 1:
            right_iou = grasp_utils.get_iou(
                grasp_utils.right_overlap_vertices_tracker.elements[0],
                grasp_utils.right_overlap_vertices_tracker.elements[1],
            )
            left_iou = grasp_utils.get_iou(
                grasp_utils.left_overlap_vertices_tracker.elements[0],
                grasp_utils.left_overlap_vertices_tracker.elements[1],
            )
            grasp_utils.iou_tracker_right.push(right_iou)
            grasp_utils.iou_tracker_left.push(left_iou)
            if count >= 2:
                left_grasp, right_grasp = grasp_utils.check_iou()
        else:
            right_iou = None
            left_iou = None
            right_grasp = None
            left_grasp = None
        info_dict[count] = {
            'timestamp_ns': timestamp_ns,
            'right_overlap': overlapping_object_vertices_right,
            'left_overlap': overlapping_object_vertices_left,
            'right_iou': right_iou,
            'left_iou': left_iou,
            'right_grasp': right_grasp,
            'left_grasp': left_grasp,
            'right_object': object_right_name,
            'left_object': object_left_name,
        }
        info_dataframe.loc[count] = [
            count,
            timestamp_ns,
            overlapping_object_vertices_right,
            overlapping_object_vertices_left,
            right_iou,
            left_iou,
            right_grasp,
            left_grasp,
            object_right_name,
            object_left_name,
        ]
        # Save the dataframe every 50 counts
        if count % 500 == 0:
            info_dataframe.to_csv(info_dataframe_path, index=False)
            pickle.dump(info_dict, open(f'{info_dataframe_path.replace(".csv", ".pkl")}', 'wb'))
    info_dataframe.to_csv(info_dataframe_path, index=False)
    pickle.dump(info_dict, open(f'{info_dataframe_path.replace(".csv", ".pkl")}', 'wb'))
    print(f'Processed sequence: {sequence} and saved the information at {info_dataframe_path}')
