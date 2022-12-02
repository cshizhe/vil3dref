import os
import argparse
import json
import numpy as np
import pprint
import time
import multiprocessing as mp
from functools import partial

from plyfile import PlyData

import torch

def process_per_scan(scan_id, scan_dir, out_dir, apply_global_alignment=True, is_test=False):
    pcd_out_dir = os.path.join(out_dir, 'pcd_with_global_alignment' if apply_global_alignment else 'pcd_no_global_alignment')
    os.makedirs(pcd_out_dir, exist_ok=True)
    obj_out_dir = os.path.join(out_dir, 'instance_id_to_name')
    os.makedirs(obj_out_dir, exist_ok=True)

    # Load point clouds with colors
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply'%(scan_id)), 'rb') as f:
        plydata = PlyData.read(f) # elements: vertex, face
    points = np.array([list(x) for x in plydata.elements[0]]) # [[x, y, z, r, g, b, alpha]]
    coords = np.ascontiguousarray(points[:, :3])
    colors = np.ascontiguousarray(points[:, 3:6])
    # # TODO: normalize the coords and colors
    # coords = coords - coords.mean(0)
    # colors = colors / 127.5 - 1

    if apply_global_alignment:
        align_matrix = np.eye(4)
        with open(os.path.join(scan_dir, scan_id, '%s.txt'%(scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break
        # Transform the points
        pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)
        pts[:, 0:3] = coords
        coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords)) == 0)

    # Load point labels if any
    if is_test:
        sem_labels = None
        instance_labels = None
    else:
        # colored by nyu40 labels (ply property 'label' denotes the nyu40 label id)
        with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f:
            plydata = PlyData.read(f)
        sem_labels = np.array(plydata.elements[0]['label']).astype(np.long)
        assert len(coords) == len(colors) == len(sem_labels)        

        # Map each point to segment id
        with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id)), 'r') as f:
            d = json.load(f)
        seg = d['segIndices']
        segid_to_pointid = {}
        for i, segid in enumerate(seg):
            segid_to_pointid.setdefault(segid, [])
            segid_to_pointid[segid].append(i)

        # Map object to segments
        instance_class_labels = []
        instance_segids = []
        with open(os.path.join(scan_dir, scan_id, '%s.aggregation.json'%(scan_id)), 'r') as f:
            d = json.load(f)
        for i, x in enumerate(d['segGroups']):
            assert x['id'] == x['objectId'] == i
            instance_class_labels.append(x['label'])
            instance_segids.append(x['segments'])
        
        instance_labels = np.ones(sem_labels.shape[0], dtype=np.long) * -100
        for i, segids in enumerate(instance_segids):
            pointids = []
            for segid in segids:
                pointids += segid_to_pointid[segid]
            if np.sum(instance_labels[pointids] != -100) > 0:
                # scene0217_00 contains some overlapped instances
                print(scan_id, i, np.sum(instance_labels[pointids] != -100), len(pointids))
            else:
                instance_labels[pointids] = i
                assert len(np.unique(sem_labels[pointids])) == 1, 'points of each instance should have the same label'

        json.dump(
            instance_class_labels, 
            open(os.path.join(obj_out_dir, '%s.json'%scan_id), 'w'), 
            indent=2
        )

    torch.save(
        (coords, colors, sem_labels, instance_labels), 
        os.path.join(pcd_out_dir, '%s.pth'%(scan_id))
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans')

    # Optional arguments.
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--apply_global_alignment', default=False, action='store_true',
                        help='rotate/translate entire scan globally to aligned it with other scans')
    args = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(args))
    print(args_string)

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # for split in ['scans', 'scans_test']:
    for split in ['scans']:
        scannet_dir = os.path.join(args.scannet_dir, split)
        fn = partial(
            process_per_scan, 
            scan_dir=scannet_dir, 
            out_dir=args.output_dir, 
            apply_global_alignment=args.apply_global_alignment,
            is_test='test' in split
        )

        scan_ids = os.listdir(scannet_dir)
        scan_ids.sort()
        print(split, '%d scans' % (len(scan_ids)))

        start_time = time.time()
        if args.num_workers == -1:
            num_workers = min(mp.cpu_count(), len(scan_ids))

        pool = mp.Pool(num_workers)
        pool.map(fn, scan_ids)
        pool.close()
        pool.join()

        print("Process data took {:.4} minutes.".format((time.time() - start_time) / 60.0))


if __name__ == '__main__':
    main()
