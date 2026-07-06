"""
algorithms/tcn.py — TCN 切换策略

模型结构：[B, T, F] → 因果膨胀卷积堆叠 → 最后时刻特征 → FC → [B, C]

特点：因果卷积（只看历史）、膨胀卷积（指数增大感受野）、残差连接。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import TCNConfig, HandoverConfig, TCN_CFG, HO_CFG
from .base import DLHandoverPolicy


class CausalConv1d(nn.Module):
    """因果卷积（只看历史，不看未来）"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积 + 残差连接"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNModel(nn.Module):
    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_size, cfg.num_channels[0])
        layers = []
        in_ch = cfg.num_channels[0]
        for i, out_ch in enumerate(cfg.num_channels):
            layers.append(TCNBlock(in_ch, out_ch, cfg.kernel_size, 2**i, cfg.dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_channels[-1], cfg.num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x).transpose(1, 2)  # [B, C, T]
        x = self.tcn(x)[:, :, -1]               # [B, C_last]
        return self.classifier(x)


class TCNHandoverPolicy(DLHandoverPolicy):
    def __init__(self, cfg: TCNConfig = TCN_CFG, ho_cfg: HandoverConfig = HO_CFG):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = TCNModel(self.cfg).to(self.device)
        return self.model