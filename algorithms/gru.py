"""
algorithms/gru.py — GRU 切换策略

模型结构：[B, T, F] → GRU → 最后时隙隐状态 → FC → [B, C]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GRUConfig, HandoverConfig, GRU_CFG, HO_CFG
from .base import DLHandoverPolicy


class GRUModel(nn.Module):
    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(out_size, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class GRUHandoverPolicy(DLHandoverPolicy):
    def __init__(self, cfg: GRUConfig = GRU_CFG, ho_cfg: HandoverConfig = HO_CFG):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = GRUModel(self.cfg).to(self.device)
        return self.model