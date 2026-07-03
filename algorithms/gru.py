"""
algorithms/gru.py
=================
GRU 切换策略

模型结构：
  输入 [B, T, F] → GRU(hidden=128, layers=2) → 最后时隙隐状态 → FC → 输出 [B, C]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GRUConfig, HandoverConfig, GRU_CFG, HO_CFG
from .base import DLHandoverPolicy


class GRUModel(nn.Module):
    """GRU 时序分类模型"""

    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        gru_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(gru_out_size, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: [B, T, F]

        返回：
            logits: [B, num_classes]
        """
        # GRU 输出：output [B, T, H]，hidden [num_layers, B, H]
        output, _ = self.gru(x)
        # 取最后时隙的输出
        last_hidden = output[:, -1, :]  # [B, H]
        logits = self.classifier(last_hidden)  # [B, C]
        return logits


class GRUHandoverPolicy(DLHandoverPolicy):
    """GRU 切换策略"""

    def __init__(
        self,
        cfg: GRUConfig = GRU_CFG,
        ho_cfg: HandoverConfig = HO_CFG,
    ):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = GRUModel(self.cfg).to(self.device)
        return self.model