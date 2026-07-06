"""
config.py — 全局配置
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# 项目根目录（相对路径，无需修改）
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# 数据文件路径
TRAJECTORY_DATA_PATH = DATA_DIR / "trajectory_data.npz"
DATASET_PATH = DATA_DIR / "dataset.npz"
MODEL_DIR = OUTPUT_DIR / "models"


@dataclass
class DataConfig:
    """数据集参数"""
    slot_duration: float = 0.04   # 时隙时长 [s]
    window_size: int = 10          # 特征窗口（历史时隙数）
    pred_horizon: int = 5          # 预测时域（时隙数）
    num_cells: int = 7             # 基站数量
    num_features: int = 70         # 特征维度（10 类 × 7 基站）
    fc: float = 3.5e9             # 载波频率 [Hz]


@dataclass
class A3Config:
    """A3 切换算法参数"""
    default_offset_db: float = 3.0    # 默认偏置 [dB]
    ttt_ms: float = 80.0              # 触发时间 [ms]
    hysteresis_db: float = 2.0        # 迟滞量 [dB]
    offset_search_range: tuple = (-3.0, 9.0)   # 优化搜索范围 [dB]
    offset_search_step: float = 0.5            # 搜索步长 [dB]
    rlf_sinr_threshold_db: float = -6.0        # RLF 判定 SINR 阈值 [dB]
    rlf_duration_ms: float = 200.0             # RLF 判定持续时间 [ms]
    sinr_weight: float = 1.0
    ho_count_penalty: float = 0.1
    rlf_penalty: float = 5.0


@dataclass
class ModelConfig:
    """深度学习模型基础参数"""
    input_size: int = 70
    seq_len: int = 10
    num_classes: int = 7
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    use_class_weights: bool = True
    device: str = "auto"


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


@dataclass
class HandoverConfig:
    """切换执行参数"""
    sliding_window_n: int = 3      # 滑动窗口长度（连续 N 次预测才切换）
    rlf_sinr_threshold_db: float = -6.0
    rlf_duration_ms: float = 200.0
    ho_cooldown_slots: int = 5     # 切换冷却时间 [slots]


@dataclass
class EvalConfig:
    """评估参数"""
    ping_pong_window_slots: int = 25   # 乒乓判定时间窗口（25 slots = 1s）


# 全局默认实例
DATA_CFG = DataConfig()
A3_CFG = A3Config()
GRU_CFG = GRUConfig()
TCN_CFG = TCNConfig()
TRANSFORMER_CFG = TransformerConfig()
HO_CFG = HandoverConfig()
EVAL_CFG = EvalConfig()