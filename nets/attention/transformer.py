from typing import Tuple

import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention

#Ref: https://github.com/zju3dv/LoFTR/blob/master/src/loftr/loftr_module/transformer.py
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
        d_model: int,
        nhead: int,
        attention: str = 'linear'
    ):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model: int, nhead: int, layer_name: str, attention: str):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_name = layer_name
        encoder_layer = LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0: torch.Tensor, feat1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.layer_name == 'self':
            feat0 = self.layers[0](feat0, feat0)
            feat1 = self.layers[0](feat1, feat1)
        elif self.layer_name == 'cross':
            feat0 = self.layers[0](feat0, feat1)
            feat1 = self.layers[0](feat1, feat0)
        else:
            raise KeyError

        return feat0, feat1
