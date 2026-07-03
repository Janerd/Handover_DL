"""
algorithms/base.py
==================
DL 切换策略基类

所有深度学习切换策略的公共逻辑：
  1. 滑动窗口预测（连续 N 次预测同一邻区才执行切换）
  2. 推理时延测量
  3. 模型保存/加载
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
    """
    深度学习切换策略基类

    子类需要实现：
        build_model() → nn.Module
        predict_batch(X) → np.ndarray  （批量预测，返回目标小区 ID）
    """

    def __init__(
        self,
        ho_cfg: HandoverConfig = HO_CFG,
        device: str = "auto",
    ):
        self.ho_cfg = ho_cfg

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self._window_size: int = 10
        self._num_features: int = 70

    @abstractmethod
    def build_model(self) -> nn.Module:
        """构建模型，返回 nn.Module"""
        pass

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测目标小区

        参数：
            X: [N, window_size, num_features] float32

        返回：
            pred: [N] int，预测的目标小区 ID
        """
        assert self.model is not None, "请先调用 build_model() 或 load()"
        self.model.eval()

        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)  # [N, num_classes]
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        return pred.astype(np.int32)

    def run(self, traj: TrajectoryData) -> np.ndarray:
        """
        在单条轨迹上执行 DL 切换策略（滑动窗口）

        滑动窗口逻辑：
            每个时隙 t，用 [t-W+1, ..., t] 的特征预测目标小区。
            如果预测的目标小区 ≠ 当前服务小区，计数器 +1。
            计数器达到 N 时执行切换，计数器重置。

        参数：
            traj: 轨迹数据

        返回：
            serving_cells: [T] 每时隙的服务小区 ID
        """
        assert self.model is not None, "请先训练或加载模型"

        T = traj.num_slots
        W = self._window_size
        N = self.ho_cfg.sliding_window_n
        cooldown_max = self.ho_cfg.ho_cooldown_slots

        # 构建特征矩阵 [T, F]
        feat = traj.get_feature_matrix()

        serving_cells = np.zeros(T, dtype=np.int32)
        # 初始服务小区：第一个时隙 RSRP 最强的小区
        serving_cells[0] = int(np.argmax(traj.rsrp_l3[0]))

        # 滑动窗口计数器：对每个候选小区，记录连续预测次数
        slide_counters = {}  # {cell_id: count}
        cooldown = 0

        # 批量预测所有时隙（提高效率）
        # 对 t < W 的时隙，用零填充
        X_all = np.zeros((T, W, self._num_features), dtype=np.float32)
        for t in range(T):
            start = max(0, t - W + 1)
            end = t + 1
            length = end - start
            X_all[t, W - length:] = feat[start:end]

        predictions = self.predict_batch(X_all)  # [T]

        for t in range(1, T):
            current_cell = serving_cells[t - 1]
            pred_cell = int(predictions[t])

            if cooldown > 0:
                serving_cells[t] = current_cell
                cooldown -= 1
                continue

            if pred_cell != current_cell:
                # 预测的目标小区不是当前服务小区，计数器 +1
                slide_counters[pred_cell] = slide_counters.get(pred_cell, 0) + 1
                # 其他小区的计数器重置
                for cell in list(slide_counters.keys()):
                    if cell != pred_cell:
                        slide_counters[cell] = 0
            else:
                # 预测当前小区最好，重置所有计数器
                slide_counters = {}

            # 检查是否有小区达到切换阈值
            best_cell = -1
            for cell, count in slide_counters.items():
                if count >= N and cell != current_cell:
                    best_cell = cell
                    break

            if best_cell >= 0:
                serving_cells[t] = best_cell
                slide_counters = {}
                cooldown = cooldown_max
            else:
                serving_cells[t] = current_cell

        return serving_cells

    def measure_inference_time(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        """
        测量单次推理时延（μs）

        参数：
            num_warmup: 预热次数（不计入统计）
            num_runs:   正式测量次数

        返回：
            mean_time_us: 平均推理时延 [μs]
        """
        assert self.model is not None
        self.model.eval()

        # 单个样本的输入
        X = np.random.randn(1, self._window_size, self._num_features).astype(np.float32)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(X_tensor)

        # 同步 GPU（如果使用 CUDA）
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 正式测量
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(X_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)  # 转换为 μs

        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        print(f"  推理时延：{mean_time:.2f} ± {std_time:.2f} μs（{num_runs} 次）")
        return mean_time

    def evaluate_on_trajectories(
        self,
        trajectories: List[TrajectoryData],
        algorithm_name: str = "DL",
    ) -> AlgorithmResult:
        """
        在轨迹集上评估 DL 策略

        参数：
            trajectories:   轨迹列表
            algorithm_name: 算法名称

        返回：
            AlgorithmResult
        """
        # 测量推理时延
        infer_time = self.measure_inference_time()

        traj_results = []
        for traj in tqdm(trajectories, desc=f"评估 {algorithm_name}"):
            serving = self.run(traj)
            result = evaluate_trajectory(traj, serving, inference_time_us=infer_time)
            traj_results.append(result)

        return aggregate_results(algorithm_name, traj_results)

    def save(self, path: str) -> None:
        """保存模型权重"""
        assert self.model is not None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到：{path}")

    def load(self, path: str) -> None:
        """加载模型权重"""
        if self.model is None:
            self.model = self.build_model().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 {path} 加载")

    def count_parameters(self) -> int:
        """统计模型参数量"""
        assert self.model is not None
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  总参数量：{total:,}，可训练：{trainable:,}")
        return trainable