# import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from gem_cnn.transform.scale_mask import mask_idx, invert_index
from gem_cnn.utils.rep_act import rep_act
from torch_geometric.nn import MessagePassing


class DownConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.Conv1d(out_channels, out_channels, kernel_size, 2, bias=False, **kwargs),
            nn.BatchNorm1d(out_channels)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size, 2, bias=False, **kwargs)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.sequential(x)
        out += identity
        return F.leaky_relu(out, 0.2)


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = config["encoder"]

        self.dim = config["dim"]
        self.dim_out = self.dim[-1] * config["l"] // (2 ** len(self.dim))
        self.l_in = config["l"]
        self.n_channels = config["n_channels"]

        # model layers
        self.encoder = self.make_encoder()

    def make_encoder(self):
        layers = [DownConv1D(self.n_channels, self.dim[0], 3, padding=1)]
        for i in range(len(self.dim) - 1):
            layers.append(DownConv1D(self.dim[i], self.dim[i+1], 3, padding=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape

        x = rearrange(x, 'b c n l -> (b n) c l')
        x = self.encoder(x)
        x = rearrange(x, '(b n) c l -> b 1 n (c l)', n=shape[2])
        return x


"""Adapted from the gem_cnn library, this module contains a corrected Parallel Transport pool class."""

class ParallelTransportPool(MessagePassing):
    def __init__(self, coarse_lvl, *, unpool):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.coarse_lvl = coarse_lvl
        self.unpool = unpool

    def forward(self, x, data):
        pool_edge_mask = mask_idx(2 * self.coarse_lvl, data.edge_mask)
        node_idx_fine = torch.nonzero(data.node_mask >= self.coarse_lvl - 1).view(-1)
        node_idx_coarse = torch.nonzero(data.node_mask >= self.coarse_lvl).view(-1)
        node_idx_all_to_fine = invert_index(node_idx_fine, data.num_nodes)
        node_idx_all_to_coarse = invert_index(node_idx_coarse, data.num_nodes)

        coarse, fine = data.edge_index[:, pool_edge_mask]
        coarse_idx_coarse = node_idx_all_to_coarse[coarse]
        fine_idx_fine = node_idx_all_to_fine[fine]

        num_fine, num_coarse = node_idx_fine.shape[0], node_idx_coarse.shape[0]

        if self.unpool:
            connection = -data.connection[pool_edge_mask]  # Parallel transport inverse
            edge_index = torch.stack([fine_idx_fine, coarse_idx_coarse])  # Coarse to fine
            size = (num_fine, num_coarse)
        else:  # Pool
            connection = data.connection[pool_edge_mask]  # Parallel transport
            edge_index = torch.stack([coarse_idx_coarse, fine_idx_fine])  # Fine to coarse
            size = (num_coarse, num_fine)

        out = self.propagate(edge_index=edge_index, x=x, connection=connection, size=size)
        return out

    def message(self, x_j, connection):
        x_j_transported = rep_act(x_j, connection)
        return x_j_transported
