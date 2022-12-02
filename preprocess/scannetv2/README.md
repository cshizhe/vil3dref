# ScanNet

To get access to the ScanNet scans please refer to [ScanNet Official Repo](https://github.com/ScanNet/ScanNet#scannet-data) for getting the download instructions.
you will need to download the following files for each scan.
```
*.aggregation.json
*.txt, 
*_vh_clean_2.0.010000.segs.json
*_vh_clean_2.ply
*_vh_clean_2.labels.ply
```

# Preprocess data for ReferIt3DNet
```
python prepare_scannet_data.py --scannet_dir ~/scratch/datasets/scannet/data --output_dir ~/scratch/datasets/referit3d/scan_data --apply_global_alignment
```

It will generate {scan_id}_with_global_alignment.pth and {scan_id}_instance_id_to_name.json
```
points, colors, semantic_labels, instance_labels = torch.load({scan_id}_with_global_alignment.pth)
points: xyz, (n, 3)
colors: rgb (n, 3) [0, 255]
semantic_labels: (n, ), nyu_40 classes
instance_labels: (n, ), instance_ids for each object, -100 for unlabeled
```