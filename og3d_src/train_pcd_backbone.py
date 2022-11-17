import os
import sys
import json
import numpy as np
import time
from collections import defaultdict, Counter
from tqdm import tqdm
from easydict import EasyDict
import pprint
import jsonlines
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from utils.logger import LOGGER, TB_LOGGER, AverageMeter, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from model.obj_encoder import PcdObjEncoder
from model.referit3d_net import get_mlp_head

class PcdClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obj_encoder = PcdObjEncoder(config.obj_encoder)
        self.obj3d_clf_pre_head = get_mlp_head(
            config.hidden_size, config.hidden_size, 
            config.num_obj_classes, dropout=config.dropout
        )
    
    def forward(self, obj_pcds):
        obj_embeds = self.obj_encoder.pcd_net(obj_pcds)
        obj_embeds = self.obj_encoder.dropout(obj_embeds)
        logits = self.obj3d_clf_pre_head(obj_embeds)
        return logits


class PcdDataset(Dataset):
    def __init__(
        self, scan_id_file, scan_dir, category_file, num_points=1024,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        og3d_subset_file=None, with_rgb=True,
    ):
        scan_ids = [x.strip() for x in open(scan_id_file, 'r')]

        if og3d_subset_file is not None:
            og3d_scanids = set()
            with jsonlines.open(og3d_subset_file, 'r') as f:
                for item in f:
                    og3d_scanids.add(item['scan_id'])
            scan_ids = [scan_id for scan_id in scan_ids if scan_id in og3d_scanids]

        self.scan_ids = scan_ids
        self.scan_dir = scan_dir
        self.keep_background = keep_background
        self.random_rotate = random_rotate
        self.num_points = num_points
        self.with_rgb = with_rgb

        self.int2cat = json.load(open(category_file, 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}

        self.data = []
        for scan_id in self.scan_ids:
            pcds, colors, _, instance_labels = torch.load(
                os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
            )
            obj_labels = json.load(open(
                os.path.join(self.scan_dir, 'instance_id_to_name', '%s.json'%scan_id)
            ))
            for i, obj_label in enumerate(obj_labels):
                if (not self.keep_background) and obj_label in ['wall', 'floor', 'ceiling']:
                    continue
                mask = instance_labels == i 
                assert np.sum(mask) > 0, 'scan: %s, obj %d' %(scan_id, i)
                obj_pcd = pcds[mask]
                obj_color = colors[mask]
                # normalize
                obj_pcd = obj_pcd - obj_pcd.mean(0)
                max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
                if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                    max_dist = 1
                obj_pcd = obj_pcd / max_dist
                obj_color = obj_color / 127.5 - 1
                if self.with_rgb:
                    self.data.append((np.concatenate([obj_pcd, obj_color], 1), self.cat2int[obj_label]))
                else:
                    self.data.append((obj_pcd, self.cat2int[obj_label]))

    def __len__(self):
        return len(self.data)

    def _get_augmented_pcd(self, full_obj_pcds, theta=None):
        # sample points
        pcd_idxs = np.random.choice(len(full_obj_pcds), size=self.num_points, replace=(len(full_obj_pcds) < self.num_points))
        obj_pcds = full_obj_pcds[pcd_idxs]
        
        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            obj_pcds[:, :3] = np.matmul(obj_pcds[:, :3], rot_matrix.transpose())
        return obj_pcds

    def __getitem__(self, idx):
        full_obj_pcds, obj_label = self.data[idx]

        obj_pcds = []
        if self.random_rotate:
            theta = np.random.choice([0, np.pi/2, np.pi, np.pi*3/2])
        else:
            theta = 0
        obj_pcds = self._get_augmented_pcd(full_obj_pcds, theta=theta)
            
        outs = {
            'obj_pcds': torch.from_numpy(obj_pcds),
            'obj_labels': obj_label,
        }
        return outs

def pcd_collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    outs['obj_pcds'] = torch.stack(outs['obj_pcds'], 0)
    outs['obj_labels'] = torch.LongTensor(outs['obj_labels'])
    return outs



def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )
 
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        if not opts.test:
            save_training_meta(opts)
            TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
            model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
            add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Prepare model
    model_config = EasyDict(opts.model)
    model = PcdClassifier(model_config)
    model = wrap_model(model, device, opts.local_rank)

    num_weights, num_trainable_weights = 0, 0
    for p in model.parameters():
        psize = np.prod(p.size())
        num_weights += psize
        if p.requires_grad:
            num_trainable_weights += psize 
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)

    if opts.resume_files:
        checkpoint = {}
        for resume_file in opts.resume_files:
            checkpoint.update(torch.load(resume_file, map_location=lambda storage, loc: storage))
        print('resume #params:', len(checkpoint))
        model.load_state_dict(checkpoint, strict=False)

    # load data training set
    data_cfg = EasyDict(opts.dataset)
    trn_dataset = PcdDataset(
        data_cfg.trn_scan_split, data_cfg.scan_dir, data_cfg.category_file,
        num_points=data_cfg.num_points, 
        random_rotate=data_cfg.random_rotate if not opts.test else False,
        keep_background=data_cfg.keep_background,
        og3d_subset_file=data_cfg.og3d_subset_file,
        with_rgb=data_cfg.with_rgb,
    )
    val_dataset = PcdDataset(
        data_cfg.val_scan_split, data_cfg.scan_dir, data_cfg.category_file,
        random_rotate=False, 
        num_points=data_cfg.num_points,
        keep_background=data_cfg.keep_background,
        og3d_subset_file=data_cfg.og3d_subset_file,
        with_rgb=data_cfg.with_rgb,
    )
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))

    # Build data loaders
    trn_dataloader = DataLoader(
        trn_dataset, batch_size=opts.batch_size, shuffle=True, 
        num_workers=opts.num_workers, collate_fn=pcd_collate_fn,
        pin_memory=True, drop_last=False, 
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False, 
        num_workers=opts.num_workers, collate_fn=pcd_collate_fn,
        pin_memory=True, drop_last=False, 
    )
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch

    if opts.test:
        val_log = validate(model, trn_dataloader, device)
        val_log = validate(model, val_dataloader, device)
        return

    # Prepare optimizer
    optimizer, _ = build_optimizer(model, opts)

    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)

    # to compute training statistics
    avg_metrics = defaultdict(AverageMeter)

    global_step = 0

    model.train()
    optimizer.zero_grad()
    optimizer.step()
    
    val_best_scores =  {'epoch': -1, 'acc': -float('inf')}
    for epoch in tqdm(range(opts.num_epoch)):
        start_time = time.time()
        for batch in tqdm(trn_dataloader):
            for bk, bv in batch.items():
                batch[bk] = bv.to(device)
            batch_size = len(batch['obj_pcds'])
            logits = model(batch['obj_pcds'])
            loss = F.cross_entropy(logits, batch['obj_labels'])
            loss.backward()

            # optimizer update and logging
            global_step += 1
            # learning rate scheduling: TODO adhoc for txt_encoder
            lr_this_step = get_lr_sched(global_step, opts)
            for kp, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_this_step 
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            avg_metrics['loss'].update(loss.data.item(), n=batch_size)
            TB_LOGGER.log_scalar_dict({'loss': loss.data.item()})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            
        LOGGER.info(
            'Epoch %d, lr: %.6f, %s', epoch+1,  
            optimizer.param_groups[0]['lr'],
            ', '.join(['%s: %.4f'%(lk, lv.avg) for lk, lv in avg_metrics.items()])
        )
        if (epoch+1) % opts.val_every_epoch == 0:
            LOGGER.info(f'------Epoch {epoch+1}: start validation------')
            val_log = validate(model, val_dataloader, device)
            TB_LOGGER.log_scalar_dict(
                {f'valid/{k}': v.avg for k, v in val_log.items()}
            )
            # model_saver.save(model, epoch+1, optimizer=optimizer)
            if val_log['acc'].avg > val_best_scores['acc']:
                output_model_file = model_saver.save(model, epoch+1)
                val_best_scores['acc'] = val_log['acc'].avg
                val_best_scores['uw_acc'] = val_log['uw_acc'].avg
                val_best_scores['epoch'] = epoch + 1
                # remove non-best ckpts
                for ckpt_file in os.listdir(model_saver.output_dir):
                    ckpt_file = os.path.join(model_saver.output_dir, ckpt_file)
                    if ckpt_file != output_model_file:
                        os.remove(ckpt_file)
    
    LOGGER.info('Finished training!')
    LOGGER.info(
        'best epoch: %d, best acc %.4f, uw_acc: %.4f', 
        val_best_scores['epoch'], val_best_scores['acc'], val_best_scores['uw_acc']
    )

@torch.no_grad()
def validate(model, val_dataloader, device):
    model.eval()

    avg_metrics = defaultdict(AverageMeter)
    uw_acc_metrics = defaultdict(AverageMeter)
    for batch in val_dataloader:
        batch_size = len(batch['obj_pcds'])
        for bk, bv in batch.items():
            batch[bk] = bv.to(device)
        logits = model(batch['obj_pcds'])
        loss = F.cross_entropy(logits, batch['obj_labels']).data.item()
        preds = torch.argmax(logits, 1)
        acc = torch.mean((preds == batch['obj_labels']).float()).item()
        avg_metrics['loss'].update(loss, n=batch_size)
        avg_metrics['acc'].update(acc, n=batch_size)

        for pred, label in zip(preds.cpu().numpy(), batch['obj_labels'].cpu().numpy()):
            uw_acc_metrics[label].update(pred == label, n=1)

    avg_metrics['uw_acc'].update(np.mean([x.avg for x in uw_acc_metrics.values()]), n=1)
    LOGGER.info(', '.join(['%s: %.4f'%(lk, lv.avg) for lk, lv in avg_metrics.items()]))

    model.train()
    return avg_metrics



def build_args():
    parser = load_parser()
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts

if __name__ == '__main__':
    args = build_args()
    pprint.pprint(args)
    main(args)