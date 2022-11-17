import copy

import torch
import torch.nn as nn
import einops

from .backbone.point_net_pp import PointNetPP

class GTObjEncoder(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.hidden_size = hidden_size

        if self.config.onehot_ft:
            self.ft_linear = [nn.Embedding(self.config.num_obj_classes, self.config.hidden_size)]
        else:
            self.ft_linear = [nn.Linear(self.config.dim_ft, self.config.hidden_size)]
        self.ft_linear.append(nn.LayerNorm(self.config.hidden_size))
        self.ft_linear = nn.Sequential(*self.ft_linear)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, obj_fts):
        '''
        Args:
            obj_fts: LongTensor (batch, num_objs), or, FloatTensor (batch, num_objs, dim_ft)
            obj_locs: FloatTensor (batch, num_objs, dim_loc)
        '''
        obj_embeds = self.ft_linear(obj_fts)
        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds

class PcdObjEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pcd_net = PointNetPP(
            sa_n_points=config.sa_n_points,
            sa_n_samples=config.sa_n_samples,
            sa_radii=config.sa_radii,
            sa_mlps=config.sa_mlps,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()
        # obj_embeds = self.pcd_net(
        #     einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        # )
        # obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)

        # obj_embeds = []
        # for i in range(num_objs):
        #     obj_embeds.append(self.pcd_net(obj_pcds[:, i]))
        # obj_embeds = torch.stack(obj_embeds, 1)

        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds


class ObjColorEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

    def forward(self, obj_colors):
        # obj_colors: (batch, nobjs, 3, 4)
        gmm_weights = obj_colors[..., :1]
        gmm_means = obj_colors[..., 1:]

        embeds = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        return embeds
        