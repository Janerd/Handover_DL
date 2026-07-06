"""
algorithms/transformer.py — Transformer 切换策略

模型结构：[B, T, F] → 线性投影 → 位置编码 → Transformer Encoder → CLS token → FC → [B, C]

特点：因果掩码（只看历史）、CLS token 聚合序列信息、小型设计（适合实时推理）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import TransformerConfig, HandoverConfig, TRANSFORMER_CFG, HO_CFG
from .base import DLHandoverPolicy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len + 1, cfg.dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_encoder_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        self._mask = None
        self._mask_len = 0

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        if self._mask is None or self._mask_len != T:
            total = T + 1  # +1 for CLS token
            mask = torch.triu(torch.ones(total, total, dtype=torch.bool, device=device), diagonal=1)
            mask[0, :] = False  # CLS token 可以看到所有位置
            self._mask = mask
            self._mask_len = T
        return self._mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=self._causal_mask(T, x.device), is_causal=False)
        return self.classifier(x[:, 0])


class TransformerHandoverPolicy(DLHandoverPolicy):
    def __init__(self, cfg: TransformerConfig = TRANSFORMER_CFG, ho_cfg: HandoverConfig = HO_CFG):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = TransformerModel(self.cfg).to(self.device)
        return self.model