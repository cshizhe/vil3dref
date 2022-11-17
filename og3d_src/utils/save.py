"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
import torch


def save_training_meta(args):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'logs', 'config.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_epoch', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, epoch, optimizer=None, save_latest_optim=False):
        output_model_file = os.path.join(self.output_dir,
                                 f"{self.prefix}_{epoch}.{self.suffix}")
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()
            else:
                state_dict[k] = v
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'epoch': epoch, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            if save_latest_optim:
                torch.save(dump, f'{self.output_dir}/train_state_lastest.pt')
            else:
                torch.save(dump, f'{self.output_dir}/train_state_{epoch}.pt')
        return output_model_file

    def remove_previous_models(self, cur_epoch):
        for saved_model_name in os.listdir(self.output_dir):
            if saved_model_name.startswith(self.prefix):
                saved_model_epoch = int(os.path.splitext(saved_model_name)[0].split('_')[-1])
                if saved_model_epoch != cur_epoch:
                    os.remove(os.path.join(self.output_dir, saved_model_name))  

