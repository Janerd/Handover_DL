"""
data_loader.py — 数据加载
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import TRAJECTORY_DATA_PATH, DATASET_PATH, DATA_CFG, DataConfig


@dataclass
class TrajectoryData:
    """单条轨迹的完整数据"""
    traj_id: int
    speed_kmh: float
    traj_type: str
    split: str
    num_slots: int
    rsrp_l3: np.ndarray      # [T, C] L3 滤波 RSRP [dBm]
    sinr: np.ndarray          # [T, C] SINR [dB]（仅用于评估）
    rsrq: np.ndarray          # [T, C]
    doppler: np.ndarray       # [T, C]
    beam_id: np.ndarray       # [T, C]
    delay_spread: np.ndarray  # [T, C]
    k_factor: np.ndarray      # [T, C]
    min_tau: np.ndarray       # [T, C]
    pos: np.ndarray           # [T, 2]
    serving_l3: np.ndarray    # [T] 参考服务小区

    @property
    def num_cells(self) -> int:
        return self.rsrp_l3.shape[1]

    def get_feature_matrix(self) -> np.ndarray:
        """
        构建归一化特征矩阵 [T, 70]。

        特征顺序（10 类 × 7 基站）：
          RSRP_l3, RSRQ, SINR, Doppler, BeamID,
          RSRP_diff, BeamID_diff, DelaySpread, K_factor, min_tau
        """
        C = self.num_cells

        def minmax(x, lo, hi):
            return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

        def sym(x, scale):
            return np.clip(x / scale, -1.0, 1.0)

        def pos_scale(x, scale):
            return np.clip(x / scale, 0.0, 1.0)

        rsrp_diff = np.diff(self.rsrp_l3, axis=0, prepend=self.rsrp_l3[:1])
        beam_diff = np.diff(self.beam_id.astype(np.float32), axis=0,
                            prepend=self.beam_id[:1].astype(np.float32))

        feat = np.concatenate([
            minmax(self.rsrp_l3, -140.0, -40.0),
            minmax(self.rsrq, -30.0, 0.0),
            minmax(self.sinr, -20.0, 40.0),
            sym(self.doppler, 500.0),
            pos_scale(self.beam_id.astype(np.float32), max(C - 1, 1)),
            sym(rsrp_diff, 5.0),
            sym(beam_diff, max(C - 1, 1)),
            pos_scale(self.delay_spread, 1e-6),
            pos_scale(self.k_factor, 30.0),
            pos_scale(self.min_tau, 2e-6),
        ], axis=1)

        return feat.astype(np.float32)


def load_trajectories(
    split: Optional[str] = None,
    data_path: Optional[Path] = None,
) -> List[TrajectoryData]:
    """
    加载轨迹数据。

    参数：
        split:     "train" / "val" / "test" / None（全部）
        data_path: 数据文件路径（默认使用 config 中的路径）
    """
    path = data_path or TRAJECTORY_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"轨迹数据文件不存在：{path}")

    raw = np.load(path, allow_pickle=True)
    trajectories = []

    for i in range(len(raw["traj_ids"])):
        traj_split = str(raw["splits"][i])
        if split is not None and traj_split != split:
            continue

        traj = TrajectoryData(
            traj_id=int(raw["traj_ids"][i]),
            speed_kmh=float(raw["speed_kmh"][i]),
            traj_type=str(raw["traj_types"][i]),
            split=traj_split,
            num_slots=int(raw["num_slots"][i]),
            rsrp_l3=raw["RSRP_l3"][i].astype(np.float32),
            sinr=raw["SINR"][i].astype(np.float32),
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

    return trajectories


def load_dataset(
    dataset_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载样本级数据集（用于模型训练）。

    返回：X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    path = dataset_path or DATASET_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"数据集文件不存在：{path}")

    data = np.load(path, allow_pickle=True)
    X = data["X_raw"]
    Y = data["Y_cell"]

    return (
        X[data["split_train"]], Y[data["split_train"]],
        X[data["split_val"]],   Y[data["split_val"]],
        X[data["split_test"]],  Y[data["split_test"]],
    )


def normalize_X(X: np.ndarray, num_cells: int = 7) -> np.ndarray:
    """对原始特征矩阵做归一化（与 get_feature_matrix() 一致）。"""
    C = num_cells
    X = X.copy().astype(np.float32)

    def minmax(arr, lo, hi):
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    def sym(arr, scale):
        return np.clip(arr / scale, -1.0, 1.0)

    def pos_scale(arr, scale):
        return np.clip(arr / scale, 0.0, 1.0)

    X[..., 0*C:1*C] = minmax(X[..., 0*C:1*C], -140.0, -40.0)
    X[..., 1*C:2*C] = minmax(X[..., 1*C:2*C], -30.0, 0.0)
    X[..., 2*C:3*C] = minmax(X[..., 2*C:3*C], -20.0, 40.0)
    X[..., 3*C:4*C] = sym(X[..., 3*C:4*C], 500.0)
    X[..., 4*C:5*C] = pos_scale(X[..., 4*C:5*C], 6.0)
    X[..., 5*C:6*C] = sym(X[..., 5*C:6*C], 5.0)
    X[..., 6*C:7*C] = sym(X[..., 6*C:7*C], 6.0)
    X[..., 7*C:8*C] = pos_scale(X[..., 7*C:8*C], 1e-6)
    X[..., 8*C:9*C] = pos_scale(X[..., 8*C:9*C], 30.0)
    X[..., 9*C:10*C] = pos_scale(X[..., 9*C:10*C], 2e-6)

    return X


def compute_class_weights(Y: np.ndarray, num_classes: int) -> np.ndarray:
    """计算类别权重（处理标签不均衡）。"""
    counts = np.bincount(Y, minlength=num_classes).astype(np.float32)
    weights = len(Y) / (num_classes * np.maximum(counts, 1))
    return (weights / weights.mean()).astype(np.float32)


def get_neighbor_pairs() -> List[Tuple[int, int]]:
    """从 network_config.json 读取邻区对（如果存在）。"""
    import json
    cfg_path = Path(__file__).parent / "data" / "network_config.json"
    if not cfg_path.exists():
        # 默认：全连接
        C = DATA_CFG.num_cells
        return [(i, j) for i in range(C) for j in range(C) if i != j]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    relations = cfg.get("neighbor_config", {}).get("relations", {})
    pairs = []
    for s, neighbors in relations.items():
        for nb in neighbors:
            pairs.append((int(s), int(nb)))
    return pairs