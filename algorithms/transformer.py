"""
algorithms/transformer.py
==========================
小型 Transformer 切换策略

模型结构：
  输入 [B, T, F] → 线性投影 → 位置编码 → Transformer Encoder → CLS token → FC → 输出 [B, C]

设计特点：
  - 小型（d_model=64，2层，4头）：适合 UE 侧实时推理
  - 因果掩码（只看历史，不看未来）：保证实时性
  - CLS token 聚合序列信息（类似 BERT）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import TransformerConfig, HandoverConfig, TRANSFORMER_CFG, HO_CFG
from .base import DLHandoverPolicy


# =========================================================
# 位置编码
# =========================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =========================================================
# Transformer 模型
# =========================================================

class TransformerModel(nn.Module):
    """小型 Transformer 时序分类模型"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # 输入投影：F → d_model
        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.max_seq_len + 1,  # +1 for CLS token
            dropout=cfg.dropout,
        )

        # CLS token（可学习）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN（更稳定）
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

        # 因果掩码（预计算，避免每次重新生成）
        self._causal_mask: torch.Tensor = None
        self._mask_len: int = 0

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码（上三角为 True = 被遮蔽）"""
        if self._causal_mask is None or self._mask_len != seq_len:
            # 包含 CLS token，所以 seq_len+1
            total_len = seq_len + 1
            mask = torch.triu(
                torch.ones(total_len, total_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            # CLS token 可以看到所有位置（第一行全为 False）
            mask[0, :] = False
            self._causal_mask = mask
            self._mask_len = seq_len
        return self._causal_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: [B, T, F]

        返回：
            logits: [B, num_classes]
        """
        B, T, _ = x.shape

        # 输入投影：[B, T, F] → [B, T, d_model]
        x = self.input_proj(x)

        # 拼接 CLS token：[B, T+1, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 位置编码
        x = self.pos_enc(x)

        # 因果掩码
        causal_mask = self._get_causal_mask(T, x.device)

        # Transformer Encoder
        x = self.transformer(x, mask=causal_mask, is_causal=False)

        # 取 CLS token 的输出：[B, d_model]
        cls_out = x[:, 0, :]

        # 分类
        logits = self.classifier(cls_out)
        return logits


class TransformerHandoverPolicy(DLHandoverPolicy):
    """Transformer 切换策略"""

    def __init__(
        self,
        cfg: TransformerConfig = TRANSFORMER_CFG,
        ho_cfg: HandoverConfig = HO_CFG,
    ):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = TransformerModel(self.cfg).to(self.device)
        return self.model