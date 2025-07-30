"""
Modified in DAC:

V1:
    The modification free the image features from positional encoding. 
    The PE is included for computating attention, but not in feature aggregation

ERP:
    Positional encoding is based on spherical coordinates. No normalization is applied to the positional encoding.

"""

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from dac.utils import (AttentionLayerIsoPE, PositionEmbeddingSineERP,
                         _get_activation_cls, get_norm)


class ISDHead(nn.Module):
    def __init__(
        self,
        depth: int,
        pixel_dim: int = 256,
        query_dim: int = 256,
        pe_dim: int = 64,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
        pe_mlp = False,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pe_mlp = pe_mlp
        self.pixel_pe = PositionEmbeddingSineERP(pe_dim // 2)
        for i in range(self.depth):
            setattr(
                self,
                f"cross_attn_{i+1}",
                AttentionLayerIsoPE(
                    sink_dim=pixel_dim,
                    hidden_dim=pixel_dim,
                    source_dim=query_dim,
                    output_dim=pixel_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=False,
                ),
            )
            setattr(
                self,
                f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )
            if pe_mlp:
                setattr(
                    self,
                    f"mlp_pe_{i+1}",
                    nn.Sequential(
                        get_norm(norm, pe_dim),
                        nn.Linear(pe_dim, expansion * pe_dim),
                        _get_activation_cls(activation),
                        nn.Linear(expansion * pe_dim, pe_dim),
                    ),
                )
        setattr(
            self,
            "proj_output",
            nn.Sequential(
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, pixel_dim),
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, output_dim),
            ),
        )

    def forward(self, feature_map: torch.Tensor, idrs: torch.Tensor, lat_range: torch.Tensor, long_range: torch.Tensor):
        b, c, h, w = feature_map.shape
        cur_pe = self.pixel_pe(feature_map, lat_range, long_range)
        cur_pe = rearrange(cur_pe, "b c h w -> b (h w) c")
        feature_map = rearrange(feature_map, "b c h w -> b (h w) c")

        # update feature_map without fusing positional encoding
        for i in range(self.depth):
            if self.pe_mlp:
                cur_pe = cur_pe + getattr(self, f"mlp_pe_{i+1}")(cur_pe)
            # concat updated feature_map and positional encoding for attention
            feature_map_ext = torch.cat([feature_map, cur_pe], dim=-1)
            update = getattr(self, f"cross_attn_{i+1}")(feature_map_ext.clone(), idrs)
            feature_map = feature_map + update[:, :, :c]
            feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())
        out = getattr(self, "proj_output")(feature_map)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out


class ISD(nn.Module):
    def __init__(
        self,
        num_resolutions,
        depth,
        pixel_dim=128,
        query_dim=128,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        for i in range(num_resolutions):
            setattr(
                self,
                f"head_{i+1}",
                ISDHead(
                    depth=depth,
                    pixel_dim=pixel_dim,
                    query_dim=query_dim,
                    num_heads=num_heads,
                    output_dim=output_dim,
                    expansion=expansion,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(
        self, xs: Tuple[torch.Tensor, ...], idrs: Tuple[torch.Tensor, ...], lat_range: torch.Tensor, long_range: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        outs, attns = [], []
        for i in range(self.num_resolutions):
            out = getattr(self, f"head_{i+1}")(xs[i], idrs[i], lat_range, long_range)
            outs.append(out)
        return tuple(outs)

    @classmethod
    def build(cls, config):
        obj = cls(
            num_resolutions=config["model"]["isd"]["num_resolutions"],
            depth=config["model"]["isd"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            query_dim=config["model"]["afp"]["latent_dim"],
            output_dim=config["model"]["output_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj


class AFP(nn.Module):
    def __init__(
        self,
        num_resolutions: int,
        depth: int = 3,
        pixel_dim: int = 256,
        pe_dim: int = 64,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
        pe_mlp = False,
        # top_k_attn = True,
        top_k_ratio = None,
        top_ks = None,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps
        self.pe_mlp = pe_mlp
        # self.top_k_attn = top_k_attn
        self.top_k_ratio = top_k_ratio
        self.top_ks = top_ks
        # self.top_ks = [500, 2000, 8000]
        
        bottlenck_dim = expansion * latent_dim
        for i in range(self.num_resolutions):
            setattr(
                self,
                f"pixel_pe_{i+1}",
                PositionEmbeddingSineERP(pe_dim // 2),
            )
            setattr(
                self,
                f"mu_{i+1}",
                nn.Parameter(torch.randn(1, self.num_slots, latent_dim + pe_dim)),
            )
            if self.pe_mlp:
                setattr(
                    self,
                    f"mlp_pe_{i+1}",
                    nn.Sequential(
                        get_norm(norm, pe_dim),
                        nn.Linear(pe_dim, expansion * pe_dim),
                        _get_activation_cls(activation),
                        nn.Linear(expansion * pe_dim, pe_dim),
                    ),
                )

        # Set up attention iterations
        # TODO: j is not used but always setting d1 is strange
        for j in range(self.iters):
            for i in range(self.num_resolutions):
                setattr(
                    self,
                    f"cross_attn_{i+1}_d{1}",
                    AttentionLayerIsoPE(
                        sink_dim=latent_dim,
                        hidden_dim=latent_dim,
                        source_dim=pixel_dim,
                        output_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        pre_norm=True,
                        sink_competition=False,
                    ),
                )
                setattr(
                    self,
                    f"mlp_cross_{i+1}_d{1}",
                    nn.Sequential(
                        get_norm(norm, latent_dim),
                        nn.Linear(latent_dim, bottlenck_dim),
                        _get_activation_cls(activation),
                        nn.Linear(bottlenck_dim, latent_dim),
                    ),
                )

    def forward(
        self, feature_maps: Tuple[torch.Tensor, ...], lat_range: torch.Tensor, long_range: torch.Tensor, attn_masks: Tuple[torch.Tensor, ...] = None
    ) -> Tuple[torch.Tensor, ...]:
        b, *_ = feature_maps[0].shape
        idrs = []
        feature_maps_flat = []
        for i in range(self.num_resolutions):
            # feature maps embedding pre-process
            feature_map, (h, w) = feature_maps[i], feature_maps[i].shape[-2:]
            cur_pe = getattr(self, f"pixel_pe_{i+1}")(feature_map, lat_range, long_range)
            feature_map = rearrange(feature_map, "b d h w -> b (h w) d")
            cur_pe = rearrange(cur_pe, "b d h w -> b (h w) d")
            if self.pe_mlp:
                cur_pe = getattr(self, f"mlp_pe_{i+1}")(cur_pe)
            feature_maps_flat.append(torch.concat([feature_map, cur_pe], dim=-1))
            # IDRs generation
            idrs.append(getattr(self, f"mu_{i+1}").expand(b, -1, -1))

        # layers
        for i in range(self.num_resolutions):
            if attn_masks is not None:
                attn_mask = rearrange(attn_masks[i], "b d h w -> b (h w) d")
            else:
                attn_mask = None
            
            for _ in range(self.iters):
                if self.top_k_ratio is not None and self.top_k_ratio < 1:
                    top_k = int(feature_maps_flat[i].shape[1] * self.top_k_ratio)
                elif self.top_ks is not None:
                    top_k = self.top_ks[i]
                else:
                    top_k = None
                
                # Cross attention ops
                idrs[i] = idrs[i] + getattr(self, f"cross_attn_{i+1}_d{1}")(
                    idrs[i].clone(), feature_maps_flat[i], attn_mask, 
                    top_t = top_k
                )
                # mlp only applied to the feature dimension
                idrs[i] = torch.cat([
                    idrs[i][..., :self.latent_dim] + getattr(self, f"mlp_cross_{i+1}_d{1}")(idrs[i][..., :self.latent_dim].clone()), 
                    idrs[i][..., self.latent_dim:]
                    ], dim=-1)

        return tuple(idrs)

    @classmethod
    def build(cls, config):
        output_num_resolutions = (
            len(config["model"]["pixel_encoder"]["embed_dims"])
            - config["model"]["afp"]["context_low_resolutions_skip"]
        )
        if "top_k_ratio" in config["model"]["afp"]:
            top_k_ratio = config["model"]["afp"]["top_k_ratio"]
        else:
            top_k_ratio = None
            
        # if "top_ks" in config["model"]["afp"]:
        #     top_ks = config["model"]["afp"]["top_ks"]
        # else:
        #     top_ks = None
        
        obj = cls(
            num_resolutions=output_num_resolutions,
            depth=config["model"]["afp"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_latents=config["model"]["afp"]["num_latents"],
            latent_dim=config["model"]["afp"]["latent_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
            top_k_ratio=top_k_ratio,
            # top_ks=top_ks,
        )
        return obj
