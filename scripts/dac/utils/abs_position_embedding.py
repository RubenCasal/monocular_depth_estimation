import math
from typing import Optional

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # TODO: prepare the positional encoding in forward is straightforward in implementation,
    # but it is not the most efficient way. Can be pre-prepared in constructor
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class PositionEmbeddingSineERP(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(
        self, x: torch.Tensor, lat_range: Optional[torch.Tensor] = None, long_range: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # create meshgrid directly from a batch of lat_range and long_range
        x_embed = torch.zeros(x.size(0), x.size(2), x.size(3), device=x.device)
        y_embed = torch.zeros(x.size(0), x.size(2), x.size(3), device=x.device)
        for b in range(lat_range.size(0)):
            lat_range[b] = lat_range[b]
            long_range[b] = long_range[b]
            x_embed[b] = torch.tile(torch.linspace(long_range[b, 0], long_range[b, 1], x.size(3), device=x.device).unsqueeze(0), (x.size(2), 1))
            # TODO: order of lat_range need to be reversed? Current data prepared as positive downward aligned with y-axis
            y_embed[b] = torch.tile(torch.linspace(lat_range[b, 0], lat_range[b, 1], x.size(2), device=x.device).unsqueeze(0), (x.size(3), 1)).T
            
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=2):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)