"""
algorithms/base.py — 深度学习切换策略基类

公共逻辑：
  - 滑动窗口切换执行（连续 N 次预测同一邻区才切换）
  - 推理时延测量
  - 模型保存/加载
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import HandoverConfig, HO_CFG
from data_loader import TrajectoryData
from evaluator import evaluate_trajectory, aggregate_results, AlgorithmResult


class DLHandoverPolicy(ABC):
    """深度学习切换策略基类"""

    def __init__(self, ho_cfg: HandoverConfig = HO_CFG, device: str = "auto"):
        self.ho_cfg = ho_cfg
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self._window_size: int = 10
        self._num_features: int = 70

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """批量预测目标小区，输入 [N, W, F]，返回 [N] int。"""
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self.device))
            return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)

    def run(self, traj: TrajectoryData) -> np.ndarray:
        """
        在单条轨迹上执行切换策略（滑动窗口）。

        每时隙预测目标小区，连续 N 次预测同一邻区才执行切换。
        """
        assert self.model is not None
        T = traj.num_slots
        W = self._window_size
        N = self.ho_cfg.sliding_window_n
        cooldown_max = self.ho_cfg.ho_cooldown_slots

        feat = traj.get_feature_matrix()  # [T, F]

        # 批量预测所有时隙（t < W 时用零填充）
        X_all = np.zeros((T, W, self._num_features), dtype=np.float32)
        for t in range(T):
            s = max(0, t - W + 1)
            X_all[t, W - (t - s + 1):] = feat[s:t + 1]
        predictions = self.predict_batch(X_all)

        serving = np.zeros(T, dtype=np.int32)
        serving[0] = int(np.argmax(traj.rsrp_l3[0]))
        counters = {}
        cooldown = 0

        for t in range(1, T):
            cur = serving[t - 1]
            pred = int(predictions[t])

            if cooldown > 0:
                serving[t] = cur
                cooldown -= 1
                continue

            if pred != cur:
                counters[pred] = counters.get(pred, 0) + 1
                for c in list(counters):
                    if c != pred:
                        counters[c] = 0
            else:
                counters = {}

            best = max(counters, key=counters.get, default=-1)
            if best >= 0 and counters[best] >= N and best != cur:
                serving[t] = best
                counters = {}
                cooldown = cooldown_max
            else:
                serving[t] = cur

        return serving

    def measure_inference_time(self, num_runs: int = 100) -> float:
        """测量单次推理时延 [μs]。"""
        assert self.model is not None
        self.model.eval()
        X = np.random.randn(1, self._window_size, self._num_features).astype(np.float32)
        X_t = torch.from_numpy(X).float().to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                self.model(X_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                self.model(X_t)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)

        return float(np.mean(times))

    def evaluate_on_trajectories(
        self, trajectories: List[TrajectoryData], name: str = "DL"
    ) -> AlgorithmResult:
        infer_time = self.measure_inference_time()
        results = [
            evaluate_trajectory(traj, self.run(traj), inference_time_us=infer_time)
            for traj in tqdm(trajectories, desc=f"评估 {name}")
        ]
        return aggregate_results(name, results)

    def save(self, path: str) -> None:
        assert self.model is not None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def count_parameters(self) -> int:
        assert self.model is not None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)