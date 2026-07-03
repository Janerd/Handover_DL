"""
algorithms/tcn.py
=================
TCN（Temporal Convolutional Network）切换策略

模型结构：
  输入 [B, T, F] → 转置为 [B, F, T] → 因果膨胀卷积堆叠 → 最后时刻特征 → FC → 输出 [B, C]

TCN 特点：
  - 因果卷积（只看历史，不看未来）
  - 膨胀卷积（指数增大感受野）
  - 残差连接（稳定训练）
  - 并行计算（比 RNN 快）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TCNConfig, HandoverConfig, TCN_CFG, HO_CFG
from .base import DLHandoverPolicy


# =========================================================
# TCN 基础模块
# =========================================================

class CausalConv1d(nn.Module):
    """因果卷积（只看历史）"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        # 去掉右侧填充（保持因果性）
        return out[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积 + 残差连接"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # 残差连接（如果通道数不同，用 1×1 卷积对齐）
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TCNModel(nn.Module):
    """TCN 时序分类模型"""

    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.cfg = cfg

        # 输入投影：将特征维度映射到第一个通道数
        self.input_proj = nn.Linear(cfg.input_size, cfg.num_channels[0])

        # TCN 块堆叠（膨胀因子指数增长：1, 2, 4, 8, ...）
        layers = []
        in_ch = cfg.num_channels[0]
        for i, out_ch in enumerate(cfg.num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=cfg.kernel_size,
                dilation=dilation,
                dropout=cfg.dropout,
            ))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_channels[-1], cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: [B, T, F]

        返回：
            logits: [B, num_classes]
        """
        # 输入投影：[B, T, F] → [B, T, C0]
        x = self.input_proj(x)

        # 转置为 [B, C0, T]（Conv1d 需要 channel-first）
        x = x.transpose(1, 2)

        # TCN 块：[B, C0, T] → [B, C_last, T]
        x = self.tcn(x)

        # 取最后时刻的特征：[B, C_last]
        x = x[:, :, -1]

        # 分类：[B, num_classes]
        logits = self.classifier(x)
        return logits


class TCNHandoverPolicy(DLHandoverPolicy):
    """TCN 切换策略"""

    def __init__(
        self,
        cfg: TCNConfig = TCN_CFG,
        ho_cfg: HandoverConfig = HO_CFG,
    ):
        super().__init__(ho_cfg=ho_cfg, device=cfg.device)
        self.cfg = cfg
        self._window_size = cfg.seq_len
        self._num_features = cfg.input_size

    def build_model(self) -> nn.Module:
        self.model = TCNModel(self.cfg).to(self.device)
        return self.model