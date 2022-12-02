import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd_data_dir')
    parser.add_argument('out_file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    scan_ids = [x.split('.')[0] for x in os.listdir(args.pcd_data_dir)]
    scan_ids.sort()

    scene_locs = {}
    for scan_id in tqdm(scan_ids):
        points, colors, _, inst_labels = torch.load(
            os.path.join(args.pcd_data_dir, '%s.pth'%scan_id)
        )
        scene_center = np.mean(points, 0)
        scene_size = points.max(0) - points.min(0)
        scene_locs[scan_id] = np.concatenate([scene_center, scene_size], 0).tolist()
        
    json.dump(scene_locs, open(os.path.join(args.out_file), 'w'))

if __name__ == '__main__':
    main()

