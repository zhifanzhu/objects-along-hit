from argparse import ArgumentParser
import os
import os.path as osp
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm
import pandas as pd

from code_arctic.data_reader import SeqReader


""" For SeqReader, all frame_idx are 0-based.
For disk-saving, all masks and boxes 0-based;
    imagees are ioi-based;

boxes: (sid, seq_name, frame_idx, left/right/object, x1, y1, x2, y2)
"""

def main(args):
    raw_seqs_dir = Path('arctic_data/raw_seqs')
    out_root = Path('arctic_outputs/masks_low/')
    sids = sorted(os.listdir(raw_seqs_dir))[args.st:args.ed]

    for sid in tqdm.tqdm(sids):
        sid_dir = raw_seqs_dir / sid
        seq_names = list(set([x.split('.')[0] for x in os.listdir(sid_dir)]))
        for seq_name in tqdm.tqdm(seq_names):
            print(sid, seq_name)
            obj_name = seq_name.split('_')[0]
            seq_data = np.load(f'arctic_outputs/processed/seqs/{sid}/{seq_name}.npy', allow_pickle=True).item()
            seq_reader = SeqReader(seq_data, sid, seq_name, obj_name=obj_name)

            os.makedirs(out_root / sid / seq_name, exist_ok=True)

            all_boxes = []
            for f in tqdm.trange(seq_reader.num_frames):
                lbox, rbox, obox, mask = seq_reader.get_boxes_and_mask(f)
                # lbox, rbox, obox, mask = seq_reader.get_boxes_and_mask(f)
                # mp = Image.fromarray(mask)
                # mp.putpalette([0,0,0,0,255,0,0,0,255,255,0,0])
                # mp.save(out_root / sid / seq_name / f'{f:05d}.png')
                all_boxes.append([sid, seq_name, f, lbox, rbox, obox])

            all_boxes = pd.DataFrame(all_boxes,
                                     columns=['sid', 'seq_name', 'frame_idx',
                                              'lbox', 'rbox', 'obox'])
            all_boxes.to_csv(out_root / sid / f'{seq_name}_boxes.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--ed', type=int, default=999)
    args = parser.parse_args()
    main(args)
