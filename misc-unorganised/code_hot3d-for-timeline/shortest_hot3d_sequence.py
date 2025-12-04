import os
import json

from hot3d_utils import HOT3DUTILS
from hot3d_constants import DATA_CONSTANTS

shortest_duration = float('inf')
shortest_sequence = None
sequences = os.listdir(DATA_CONSTANTS['dataset_path'])
sequences = [seq for seq in sequences if seq.startswith('P')]
print(f'{len(sequences)} sequences found')
tracker = json.load(open('shortest_sequence.json', 'r'))

for sequence in sequences:
    if sequence in tracker:
        duration = tracker[sequence]
        print(f'Skipping {sequence} as it has already been processed')
    else:
        seq_info = HOT3DUTILS(sequence=sequence)
        duration = len(seq_info.timestamps)
        tracker[sequence] = duration
        json.dump(tracker, open('shortest_sequence.json', 'w'))
    if duration < shortest_duration:
        shortest_duration = duration
        shortest_sequence = sequence
print(f'Shortest sequence: {shortest_sequence} with {shortest_duration} frames')
