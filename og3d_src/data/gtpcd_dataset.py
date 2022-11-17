import os
import jsonlines
import json
import numpy as np
import random
import time

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .common import pad_tensors, gen_seq_masks
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES

class GTPcdDataset(GTLabelDataset):
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, 
        num_points=1024, max_txt_len=50, max_obj_len=80,
        random_rotate=False, in_memory=False
    ):
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate, 
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
        )
        self.num_points = num_points
        self.in_memory = in_memory

        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)

    def get_scan_pcd_data(self, scan_id):
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']
        
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # time consuming
            obj_pcds.append(pcds[mask])
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds
        return obj_pcds

    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, theta=None):
        tgt_obj_type = obj_labels[tgt_obj_idx]
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            remained_obj_idxs = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            rot_matrix = None

        obj_fts, obj_locs = [], []
        for obj_pcd in obj_pcds:
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # sample points
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        obj_fts = np.stack(obj_fts, 0)
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)
            
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx

    def __getitem__(self, idx):
        item = self.data[idx]
        scan_id = item['scan_id']
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        obj_pcds = self.get_scan_pcd_data(scan_id)
        obj_labels = self.scans[scan_id]['inst_labels']
        obj_gmm_colors = self.scans[scan_id]['inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]
        
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0

        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels,  \
            aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
                obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
                theta=theta
            )
            
        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])

        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_gmm_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes, 
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
        }
        return outs

def gtpcd_collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
    return outs
