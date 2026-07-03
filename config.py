"""
config.py
=========
切换策略实验配置

数据来源：C:/PC_Simu/Sionna/outputs/trajectory_data.npz
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# =========================================================
# 路径配置
# =========================================================

# Sionna 仿真输出目录（家用电脑路径）
SIONNA_OUTPUT_DIR = Path("C:/PC_Simu/Sionna/outputs")

# 本项目输出目录
OUTPUT_DIR = Path(__file__).parent / "outputs"

# 数据文件路径
TRAJECTORY_DATA_PATH = SIONNA_OUTPUT_DIR / "trajectory_data.npz"
DATASET_PATH = SIONNA_OUTPUT_DIR / "dataset.npz"


# =========================================================
# 数据配置
# =========================================================

@dataclass
class DataConfig:
    # 时隙时长 [s]
    slot_duration: float = 0.04  # 40ms

    # 特征窗口大小（历史时隙数）
    window_size: int = 10

    # 预测时域（未来时隙数）
    pred_horizon: int = 5

    # 基站数量
    num_cells: int = 7

    # 特征维度（10 类 × 7 基站）
    num_features: int = 70

    # 载波频率 [Hz]
    fc: float = 3.5e9

    # 特征索引（每类特征的起始索引，步长为 num_cells）
    # [0:7]   RSRP_l3
    # [7:14]  RSRQ
    # [14:21] SINR
    # [21:28] Doppler_est
    # [28:35] BeamID_norm
    # [35:42] RSRP_diff
    # [42:49] BeamID_diff
    # [49:56] DelaySpread_norm
    # [56:63] K_factor_norm
    # [63:70] min_tau_norm


# =========================================================
# A3 算法配置
# =========================================================

@dataclass
class A3Config:
    # 默认偏置 [dB]（每个邻区对可独立设置）
    default_offset_db: float = 3.0

    # 触发时间 [ms]
    ttt_ms: float = 80.0

    # 迟滞量 [dB]
    hysteresis_db: float = 2.0

    # 优化时的搜索范围 [dB]
    offset_search_range: tuple = (-3.0, 9.0)

    # 优化时的搜索步长 [dB]
    offset_search_step: float = 0.5

    # RLF 判定阈值：SINR 低于此值持续超过 rfl_duration_ms 判定为 RLF
    rlf_sinr_threshold_db: float = -6.0
    rlf_duration_ms: float = 200.0  # 200ms

    # 优化目标权重
    sinr_weight: float = 1.0
    ho_count_penalty: float = 0.1   # 每次切换的惩罚（dB 等效）
    rlf_penalty: float = 5.0        # 每次 RLF 的惩罚（dB 等效）


# =========================================================
# 深度学习模型配置
# =========================================================

@dataclass
class ModelConfig:
    # 输入维度
    input_size: int = 70       # num_features
    seq_len: int = 10          # window_size
    num_classes: int = 7       # num_cells

    # 训练参数
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10         # Early stopping patience

    # 类别权重（处理标签不均衡）
    use_class_weights: bool = True

    # 设备
    device: str = "auto"       # "auto", "cuda", "cpu"


@dataclass
class GRUConfig(ModelConfig):
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False


@dataclass
class TCNConfig(ModelConfig):
    num_channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    kernel_size: int = 3
    dropout: float = 0.2


@dataclass
class TransformerConfig(ModelConfig):
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_seq_len: int = 10


# =========================================================
# 切换执行配置
# =========================================================

@dataclass
class HandoverConfig:
    # DL 滑动窗口：连续 N 次预测同一邻区才执行切换
    sliding_window_n: int = 3   # 3 × 40ms = 120ms，类似 TTT=80ms

    # RLF 判定（与 A3Config 保持一致）
    rlf_sinr_threshold_db: float = -6.0
    rlf_duration_ms: float = 200.0

    # 切换冷却时间（切换后多少 slots 内不再切换，防止乒乓）
    ho_cooldown_slots: int = 5  # 5 × 40ms = 200ms


# =========================================================
# 评估配置
# =========================================================

@dataclass
class EvalConfig:
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: [
        "mean_sinr_db",          # 平均 SINR [dB]
        "p5_sinr_db",            # 5th percentile SINR [dB]
        "ho_count",              # 切换次数
        "rlf_count",             # RLF 次数
        "ping_pong_rate",        # 乒乓率（1s 内来回切换）
        "inference_time_us",     # 推理时延 [μs]（DL 模型）
    ])

    # 乒乓判定时间窗口 [slots]
    ping_pong_window_slots: int = 25  # 25 × 40ms = 1s


# =========================================================
# 全局默认配置
# =========================================================

DATA_CFG = DataConfig()
A3_CFG = A3Config()
GRU_CFG = GRUConfig()
TCN_CFG = TCNConfig()
TRANSFORMER_CFG = TransformerConfig()
HO_CFG = HandoverConfig()
EVAL_CFG = EvalConfig()