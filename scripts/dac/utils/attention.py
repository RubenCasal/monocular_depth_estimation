"""
Original Author: Luigi Piccinelli
Licensed under the ECL-2.0 license (https://opensource.org/license/ecl-2-0/)

Modified in DAC:
The modification free the image features from positional encoding.

"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
        out_attn: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (
            hidden_dim % num_heads
        ) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = (
            norm_layer(source_dim, eps=eps) if source_dim is not None else None
        )

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(
            hidden_dim, sink_dim if output_dim is None else output_dim
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition
        self.out_attn = out_attn

    def forward(
        self, sink: torch.Tensor, source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            sink = self.norm(sink)
            if source is not None:
                source = self.norm_context(source)

        q = self.to_q(sink)
        source = source if source is not None else sink
        k, v = self.to_kv(source).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if self.sink_competition:
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)

        if self.out_attn:
            return out, attn
        return out

"Added in DAC, given inputs are img features concate with position encodings, processes them separately."
class AttentionLayerIsoPE(nn.Module):
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
        out_attn: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (hidden_dim % num_heads) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = norm_layer(source_dim, eps=eps) if source_dim is not None else None

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(hidden_dim, sink_dim if output_dim is None else output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition
        self.out_attn = out_attn

        self.sink_dim = sink_dim
        self.source_dim = source_dim if source_dim is not None else sink_dim

    def forward(self, sink: torch.Tensor, source: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, top_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            sink = sink.clone()
            sink[:, :, :self.sink_dim] = self.norm(sink[:, :, :self.sink_dim].clone())
            if source is not None:
                source = source.clone()
                source[:, :, :self.source_dim] = self.norm_context(source[:, :, :self.source_dim].clone())

        q = self.to_q(sink[:, :, :self.sink_dim].clone())
        source = source if source is not None else sink
        k, v = self.to_kv(source[:, :, :self.source_dim].clone()).chunk(2, dim=-1)

        q = torch.cat([q, sink[:, :, self.sink_dim:].clone()], dim=-1)
        k = torch.cat([k, source[:, :, self.source_dim:].clone()], dim=-1)
        v = torch.cat([v, source[:, :, self.source_dim:].clone()], dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale
        
        # attn_mask: [B, N] is the mask for the attention on source, 1 for valid, 0 for invalid
        if attn_mask is not None:
            # similarity_matrix = similarity_matrix + (attn_mask.transpose(-1, -2) == 0).float() * -1e9
            similarity_matrix = similarity_matrix.masked_fill(attn_mask.transpose(-1, -2).repeat(1, similarity_matrix.shape[1], 1) == 0, -1e9)

        # for each query, only keep value wt. top_t attention values
        if top_t is not None:        
            top_t_values, top_t_indices = torch.topk(similarity_matrix, top_t, dim=2)
            top_t_mask = torch.zeros_like(similarity_matrix)
            top_t_mask.scatter_(2, top_t_indices, 1)
            similarity_matrix = similarity_matrix.masked_fill(top_t_mask == 0, -1e9)

        if self.sink_competition:
            # TODO: sink_competition will deactivate attn mask
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)

        out_feature = self.to_out(out[:, :, :self.sink_dim].clone())

        out = torch.cat([out_feature, out[:, :, self.sink_dim:].clone()], dim=-1)

        if self.out_attn:
            return out, attn
        return out
