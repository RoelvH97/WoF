# import necessary modules
import einops
import math
import torch

from torch import nn
from torch.nn.functional import grid_sample


class PolarTransform(nn.Module):
    def __init__(self, config, device="cuda"):
        super(PolarTransform, self).__init__()
        self.device = torch.device(device)

        self.p_theta = config["p_theta"] / config["n_theta"] * 2 * math.pi
        self.n_theta = config["n_theta"] + 2 * config["p_theta"]  # theta samples after transform
        self.n_rad = config["n_rad"]      # radius samples after transform
        self.n_z = 600                    # length of the z-axis, in pixels
        self.l_rad = config["l_rad"]      # length of the radius axis before transform, in pixels
        self.shape = config["shape"]      # shape of the input planar image

        # normalize
        H, W = self.shape
        self.l_rad = self.l_rad / H * (H / W) * 2

        # create polar grid
        self.grid = self.init_planar_grid()

    def init_planar_grid(self):
        # cartesian meshgrid
        axis_rad = torch.linspace(self.l_rad, 0, self.n_rad, dtype=torch.float32)
        axis_theta = torch.linspace(2.5 * math.pi + self.p_theta,
                                    0.5 * math.pi - self.p_theta,
                                    self.n_theta, dtype=torch.float32)
        axis_z = torch.arange(self.n_z, dtype=torch.float32)
        z, rad, theta = torch.meshgrid(axis_z, axis_rad, axis_theta, indexing="ij")
        grid = torch.stack([z, rad, theta], dim=-1).float()

        # warped grid
        warped_grid = torch.zeros_like(grid)
        warped_grid[..., 0] = grid[..., 0]
        warped_grid[..., 2] = torch.sin(grid[..., 2]) * grid[..., 1]
        warped_grid[..., 1] = torch.cos(grid[..., 2]) * grid[..., 1]
        return warped_grid.to(self.device)

    def forward(self, x, offset_range=None, random_scale=False):
        while len(x.shape) < 5:
            x = x.unsqueeze(0)

        # subsample the grid given length of the z-axis
        grid = self.grid[:x.shape[-1]]
        grid = einops.repeat(grid, "Z R θ c -> N Z R θ c", N=x.shape[0])

        # normalize z-axis
        center = torch.Tensor([x.shape[-1] / 2, 0, 0]).to(self.device)
        std = torch.Tensor([x.shape[-1] / 2, 1, 1]).to(self.device)
        grid = (grid - center) / std

        if random_scale:
            # generate a random scale per sample
            scale = torch.rand(x.shape[0], 3, dtype=torch.float32)
            scale[:, 0] = 1
            scale[:, 1] = scale[:, 1] * 2
            scale[:, 2] = scale[:, 2] * 2
            scale = scale[:, None, None, None, :]

            # apply scale
            grid = grid * scale.to(self.device)

        if offset_range:  # in pixels
            # generate an offset per sample
            center_offset = torch.randint(offset_range[0], offset_range[1], (x.shape[0], 3), dtype=torch.float32)

            # normalize
            center_offset[:, 0] = 0
            center_offset[:, 1] = center_offset[:, 1] / (self.shape[1] / 2)
            center_offset[:, 2] = center_offset[:, 2] / (self.shape[0] / 2)
            center_offset = center_offset[:, None, None, None, :]

            # apply offset
            grid = grid + center_offset.to(self.device)
            return grid_sample(x, grid, padding_mode='zeros', align_corners=True, mode='bilinear')
        else:
            return grid_sample(x, grid, padding_mode='zeros', align_corners=True, mode='bilinear')
