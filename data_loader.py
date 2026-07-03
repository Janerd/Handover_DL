"""
data_loader.py
==============
加载 Sionna 仿真生成的轨迹数据，供切换策略使用。

数据格式（trajectory_data.npz）：
  - pos:        [N] object array，每个元素 [T, 2]，UE 坐标 [m]
  - RSRP_l3:    [N] object array，每个元素 [T, C]，L3 滤波 RSRP [dBm]
  - SINR:       [N] object array，每个元素 [T, C]，SINR [dB]
  - RSRQ:       [N] object array，每个元素 [T, C]
  - doppler:    [N] object array，每个元素 [T, C]
  - beam_id:    [N] object array，每个元素 [T, C]
  - delay_spread: [N] object array，每个元素 [T, C]
  - k_factor:   [N] object array，每个元素 [T, C]
  - min_tau:    [N] object array，每个元素 [T, C]
  - serving_l3: [N] object array，每个元素 [T]，L3 服务小区 ID
  - speed_kmh:  [N]，每条轨迹的速度 [km/h]
  - traj_types: [N]，轨迹类型字符串
  - splits:     [N]，"train"/"val"/"test"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import TRAJECTORY_DATA_PATH, DATASET_PATH, DataConfig, DATA_CFG


# =========================================================
# 轨迹数据结构
# =========================================================

@dataclass
class TrajectoryData:
    """单条轨迹的完整数据"""
    traj_id: int
    speed_kmh: float
    traj_type: str
    split: str                  # "train" / "val" / "test"
    num_slots: int

    # 信道数据 [T, C]
    rsrp_l3: np.ndarray         # L3 滤波 RSRP [dBm]
    sinr: np.ndarray            # SINR [dB]（用于质量评估）
    rsrq: np.ndarray            # RSRQ [dB]
    doppler: np.ndarray         # Doppler 频移 [Hz]
    beam_id: np.ndarray         # 波束 ID
    delay_spread: np.ndarray    # 时延扩展 [s]
    k_factor: np.ndarray        # Ricean K 因子
    min_tau: np.ndarray         # 最短路径时延 [s]

    # 位置数据 [T, 2]
    pos: np.ndarray

    # 参考服务小区（L3 滤波后的最强 RSRP 小区）[T]
    serving_l3: np.ndarray

    @property
    def num_cells(self) -> int:
        return self.rsrp_l3.shape[1]

    def get_feature_matrix(self, cfg: DataConfig = DATA_CFG) -> np.ndarray:
        """
        构建特征矩阵 [T, num_features]，与 dataset.npz 中的 X_raw 格式一致。

        特征顺序（10 类 × C 基站）：
          RSRP_l3, RSRQ, SINR, Doppler, BeamID_norm,
          RSRP_diff, BeamID_diff, DelaySpread_norm, K_factor_norm, min_tau_norm
        """
        C = self.num_cells
        T = self.num_slots

        # 归一化参数（与 channel.py 保持一致）
        RSRP_MIN, RSRP_MAX = -140.0, -40.0
        RSRQ_MIN, RSRQ_MAX = -30.0, 0.0
        SINR_MIN, SINR_MAX = -20.0, 40.0
        DOPPLER_MAX = 500.0
        DS_MAX = 1e-6
        K_MAX = 30.0
        TAU_MAX = 2e-6

        rsrp_norm = np.clip((self.rsrp_l3 - RSRP_MIN) / (RSRP_MAX - RSRP_MIN), 0, 1)
        rsrq_norm = np.clip((self.rsrq - RSRQ_MIN) / (RSRQ_MAX - RSRQ_MIN), 0, 1)
        sinr_norm = np.clip((self.sinr - SINR_MIN) / (SINR_MAX - SINR_MIN), 0, 1)
        doppler_norm = np.clip(self.doppler / DOPPLER_MAX, -1, 1)
        beam_norm = self.beam_id.astype(np.float32) / max(C - 1, 1)

        rsrp_diff = np.diff(self.rsrp_l3, axis=0, prepend=self.rsrp_l3[:1])
        rsrp_diff_norm = np.clip(rsrp_diff / 5.0, -1, 1)

        beam_diff = np.diff(self.beam_id.astype(np.float32), axis=0,
                            prepend=self.beam_id[:1].astype(np.float32))
        beam_diff_norm = np.clip(beam_diff / max(C - 1, 1), -1, 1)

        ds_norm = np.clip(self.delay_spread / DS_MAX, 0, 1)
        k_norm = np.clip(self.k_factor / K_MAX, 0, 1)
        tau_norm = np.clip(self.min_tau / TAU_MAX, 0, 1)

        feat = np.concatenate([
            rsrp_norm, rsrq_norm, sinr_norm, doppler_norm, beam_norm,
            rsrp_diff_norm, beam_diff_norm, ds_norm, k_norm, tau_norm,
        ], axis=1)  # [T, 10*C]

        return feat.astype(np.float32)


# =========================================================
# 数据加载函数
# =========================================================

def load_trajectories(
    data_path: Optional[Path] = None,
    split: Optional[str] = None,
    speed_filter: Optional[float] = None,
) -> List[TrajectoryData]:
    """
    加载轨迹数据

    参数：
        data_path:    数据文件路径（默认使用 config 中的路径）
        split:        数据集划分（"train"/"val"/"test"/None=全部）
        speed_filter: 速度筛选 [km/h]（None=全部）

    返回：
        轨迹数据列表
    """
    if data_path is None:
        data_path = TRAJECTORY_DATA_PATH

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"轨迹数据文件不存在：{data_path}\n"
            "请先运行 Sionna 仿真生成数据集"
        )

    print(f"加载轨迹数据：{data_path}")
    raw = np.load(data_path, allow_pickle=True)

    num_traj = len(raw["traj_ids"])
    print(f"  总轨迹数：{num_traj}")

    trajectories = []
    for i in range(num_traj):
        traj_split = str(raw["splits"][i])
        traj_speed = float(raw["speed_kmh"][i])

        # 筛选
        if split is not None and traj_split != split:
            continue
        if speed_filter is not None and abs(traj_speed - speed_filter) > 1.0:
            continue

        # 提取数据
        sinr_raw = raw["SINR"][i].astype(np.float32)
        rsrp_raw = raw["RSRP_l3"][i].astype(np.float32)

        traj = TrajectoryData(
            traj_id=int(raw["traj_ids"][i]),
            speed_kmh=traj_speed,
            traj_type=str(raw["traj_types"][i]),
            split=traj_split,
            num_slots=int(raw["num_slots"][i]),
            rsrp_l3=rsrp_raw,
            sinr=sinr_raw,
            rsrq=raw["RSRQ"][i].astype(np.float32),
            doppler=raw["doppler"][i].astype(np.float32),
            beam_id=raw["beam_id"][i].astype(np.int32),
            delay_spread=raw["delay_spread"][i].astype(np.float32),
            k_factor=raw["k_factor"][i].astype(np.float32),
            min_tau=raw["min_tau"][i].astype(np.float32),
            pos=raw["pos"][i].astype(np.float32),
            serving_l3=raw["serving_l3"][i].astype(np.int32),
        )
        trajectories.append(traj)

    print(f"  筛选后轨迹数：{len(trajectories)}"
          + (f"（split={split}）" if split else "")
          + (f"（speed={speed_filter} km/h）" if speed_filter else ""))

    return trajectories


def load_dataset_splits(
    dataset_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载样本级数据集（用于 DL 模型训练）

    返回：
        X_train, Y_train, X_val, Y_val, X_test, Y_test
        X 形状：[N, window_size, num_features]
        Y 形状：[N]（目标小区 ID）
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")

    print(f"加载样本级数据集：{dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    X = data["X_raw"]          # [N, W, F]
    Y = data["Y_cell"]         # [N]
    train_mask = data["split_train"]
    val_mask = data["split_val"]
    test_mask = data["split_test"]

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    X_val = X[val_mask]
    Y_val = Y[val_mask]
    X_test = X[test_mask]
    Y_test = Y[test_mask]

    print(f"  训练集：{len(X_train)} 样本")
    print(f"  验证集：{len(X_val)} 样本")
    print(f"  测试集：{len(X_test)} 样本")
    print(f"  特征形状：{X.shape[1:]}（窗口 × 特征）")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# =========================================================
# 特征归一化（与 get_feature_matrix() 保持一致）
# =========================================================

# 归一化参数（与 channel.py 保持一致）
NORM_PARAMS = {
    "rsrp":    {"min": -140.0, "max": -40.0,  "mode": "minmax"},
    "rsrq":    {"min": -30.0,  "max":  0.0,   "mode": "minmax"},
    "sinr":    {"min": -20.0,  "max":  40.0,  "mode": "minmax"},
    "doppler": {"scale": 500.0,               "mode": "scale"},   # [-1, 1]
    "beam":    {"scale": 6.0,                 "mode": "scale01"}, # [0, 1]
    "rsrp_diff": {"scale": 5.0,               "mode": "scale"},   # [-1, 1]
    "beam_diff": {"scale": 6.0,               "mode": "scale"},   # [-1, 1]
    "ds":      {"scale": 1e-6,                "mode": "scale01"}, # [0, 1]
    "k":       {"scale": 30.0,                "mode": "scale01"}, # [0, 1]
    "tau":     {"scale": 2e-6,                "mode": "scale01"}, # [0, 1]
}


def normalize_X(X: np.ndarray, num_cells: int = 7) -> np.ndarray:
    """
    对 dataset.npz 中的 X_raw 做归一化，使其与 get_feature_matrix() 输出一致。

    参数：
        X:         [N, W, F] 原始特征（物理量，未归一化）
        num_cells: 基站数量

    返回：
        X_norm: [N, W, F] 归一化后的特征，范围 [0,1] 或 [-1,1]
    """
    C = num_cells
    X_norm = X.copy().astype(np.float32)

    # 特征组顺序（与 channel.py 的 build_feature_matrix 一致）
    # [0:C]    RSRP_l3
    # [C:2C]   RSRQ
    # [2C:3C]  SINR
    # [3C:4C]  Doppler
    # [4C:5C]  BeamID
    # [5C:6C]  RSRP_diff
    # [6C:7C]  BeamID_diff
    # [7C:8C]  DelaySpread
    # [8C:9C]  K_factor
    # [9C:10C] min_tau

    def minmax(arr, vmin, vmax):
        return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    def scale_sym(arr, scale):
        return np.clip(arr / scale, -1.0, 1.0)

    def scale_pos(arr, scale):
        return np.clip(arr / scale, 0.0, 1.0)

    X_norm[..., 0*C:1*C] = minmax(X[..., 0*C:1*C], -140.0, -40.0)   # RSRP_l3
    X_norm[..., 1*C:2*C] = minmax(X[..., 1*C:2*C], -30.0,   0.0)    # RSRQ
    X_norm[..., 2*C:3*C] = minmax(X[..., 2*C:3*C], -20.0,  40.0)    # SINR
    X_norm[..., 3*C:4*C] = scale_sym(X[..., 3*C:4*C], 500.0)        # Doppler
    X_norm[..., 4*C:5*C] = scale_pos(X[..., 4*C:5*C], 6.0)          # BeamID
    X_norm[..., 5*C:6*C] = scale_sym(X[..., 5*C:6*C], 5.0)          # RSRP_diff
    X_norm[..., 6*C:7*C] = scale_sym(X[..., 6*C:7*C], 6.0)          # BeamID_diff
    X_norm[..., 7*C:8*C] = scale_pos(X[..., 7*C:8*C], 1e-6)         # DelaySpread
    X_norm[..., 8*C:9*C] = scale_pos(X[..., 8*C:9*C], 30.0)         # K_factor
    X_norm[..., 9*C:10*C] = scale_pos(X[..., 9*C:10*C], 2e-6)       # min_tau

    return X_norm


def compute_class_weights(Y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    计算类别权重（处理标签不均衡）

    使用 sklearn 的 balanced 策略：weight_c = N / (num_classes × count_c)

    返回：
        weights: [num_classes] float32
    """
    counts = np.bincount(Y_train, minlength=num_classes).astype(np.float32)
    N = len(Y_train)
    weights = N / (num_classes * np.maximum(counts, 1))
    weights = weights / weights.mean()  # 归一化，均值为 1
    print(f"类别权重：{weights.round(2)}")
    return weights.astype(np.float32)


def get_neighbor_pairs(network_config_path: Optional[Path] = None) -> List[Tuple[int, int]]:
    """
    从 network_config.json 读取邻区对

    返回：
        [(serving_cell, neighbor_cell), ...] 有向邻区对列表
    """
    import json

    if network_config_path is None:
        # 尝试从 Sionna 项目目录读取
        candidates = [
            Path("C:/PC_Simu/Sionna/network_config.json"),
            Path(__file__).parent.parent / "Sionna" / "network_config.json",
        ]
        for p in candidates:
            if p.exists():
                network_config_path = p
                break

    if network_config_path is None or not Path(network_config_path).exists():
        # 默认：全连接邻区对
        print("警告：未找到 network_config.json，使用全连接邻区对")
        num_cells = DATA_CFG.num_cells
        return [(i, j) for i in range(num_cells) for j in range(num_cells) if i != j]

    with open(network_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    relations = cfg.get("neighbor_config", {}).get("relations", {})
    pairs = []
    for serving_str, neighbors in relations.items():
        serving = int(serving_str)
        for nb in neighbors:
            pairs.append((serving, int(nb)))

    print(f"邻区对：{len(pairs)} 个（来自 network_config.json）")
    return pairs