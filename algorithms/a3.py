"""
algorithms/a3.py
================
A3 切换算法（3GPP TS 36.331 / 38.331）

A3 事件触发条件：
    RSRP_neighbor - RSRP_serving > offset(serving, neighbor) + hysteresis

持续满足 TTT（Time-To-Trigger）后执行切换。

优化：
    在训练集上，对每个邻区对独立优化 offset，
    目标：最大化平均 SINR，同时约束 RLF 次数。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from config import A3Config, HandoverConfig, A3_CFG, HO_CFG
from data_loader import TrajectoryData, get_neighbor_pairs
from evaluator import evaluate_trajectory, aggregate_results, AlgorithmResult


# =========================================================
# A3 切换策略
# =========================================================

class A3HandoverPolicy:
    """
    A3 切换策略

    每个邻区对 (serving, neighbor) 有独立的 offset_db。
    TTT 和 hysteresis 全局共享。

    使用方法：
        policy = A3HandoverPolicy()
        policy.optimize(train_trajectories)  # 在训练集上优化 offset
        serving_cells = policy.run(traj)     # 在测试轨迹上执行
    """

    def __init__(
        self,
        cfg: A3Config = A3_CFG,
        ho_cfg: HandoverConfig = HO_CFG,
        neighbor_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        self.cfg = cfg
        self.ho_cfg = ho_cfg

        # 邻区对
        if neighbor_pairs is None:
            neighbor_pairs = get_neighbor_pairs()
        self.neighbor_pairs = neighbor_pairs

        # 每个邻区对的 offset [dB]，初始化为默认值
        self.offsets: Dict[Tuple[int, int], float] = {
            pair: cfg.default_offset_db for pair in neighbor_pairs
        }

        # TTT 对应的时隙数
        self._ttt_slots = max(1, int(cfg.ttt_ms / 40.0))  # 40ms per slot

    def run(self, traj: TrajectoryData) -> np.ndarray:
        """
        在单条轨迹上执行 A3 切换策略

        参数：
            traj: 轨迹数据

        返回：
            serving_cells: [T] 每时隙的服务小区 ID
        """
        T = traj.num_slots
        C = traj.num_cells
        rsrp = traj.rsrp_l3  # [T, C]

        serving_cells = np.zeros(T, dtype=np.int32)
        # 初始服务小区：第一个时隙 RSRP 最强的小区
        serving_cells[0] = int(np.argmax(rsrp[0]))

        # TTT 计数器：对每个邻区，记录连续满足 A3 条件的时隙数
        ttt_counters = np.zeros(C, dtype=np.int32)
        # 冷却计数器（切换后不立即再切换）
        cooldown = 0

        for t in range(1, T):
            current_cell = serving_cells[t - 1]
            current_rsrp = rsrp[t, current_cell]

            # 检查每个邻区是否满足 A3 条件
            for nb in range(C):
                if nb == current_cell:
                    ttt_counters[nb] = 0
                    continue

                pair = (current_cell, nb)
                offset = self.offsets.get(pair, self.cfg.default_offset_db)
                hysteresis = self.cfg.hysteresis_db

                # A3 条件：邻区 RSRP > 服务小区 RSRP + offset + hysteresis
                nb_rsrp = rsrp[t, nb]
                if nb_rsrp > current_rsrp + offset + hysteresis:
                    ttt_counters[nb] += 1
                else:
                    ttt_counters[nb] = 0

            # 检查是否有邻区满足 TTT
            best_nb = -1
            best_rsrp = -np.inf

            if cooldown <= 0:
                for nb in range(C):
                    if nb == current_cell:
                        continue
                    if ttt_counters[nb] >= self._ttt_slots:
                        if rsrp[t, nb] > best_rsrp:
                            best_rsrp = rsrp[t, nb]
                            best_nb = nb

            if best_nb >= 0:
                # 执行切换
                serving_cells[t] = best_nb
                ttt_counters[:] = 0
                cooldown = self.ho_cfg.ho_cooldown_slots
            else:
                serving_cells[t] = current_cell
                if cooldown > 0:
                    cooldown -= 1

        return serving_cells

    def optimize(
        self,
        train_trajectories: List[TrajectoryData],
        verbose: bool = True,
    ) -> None:
        """
        在训练集上优化每个邻区对的 offset

        优化策略：
            对每个邻区对，在搜索范围内网格搜索，
            目标：最大化 (平均 SINR - ho_count_penalty × HO次数 - rlf_penalty × RLF次数)

        参数：
            train_trajectories: 训练集轨迹
            verbose:            是否打印优化过程
        """
        if verbose:
            print(f"\n优化 A3 offset（{len(self.neighbor_pairs)} 个邻区对）...")
            print(f"  训练轨迹数：{len(train_trajectories)}")
            print(f"  搜索范围：{self.cfg.offset_search_range} dB，步长：{self.cfg.offset_search_step} dB")

        search_values = np.arange(
            self.cfg.offset_search_range[0],
            self.cfg.offset_search_range[1] + self.cfg.offset_search_step / 2,
            self.cfg.offset_search_step,
        )

        # 对每个邻区对独立优化
        for pair in tqdm(self.neighbor_pairs, desc="优化邻区对", disable=not verbose):
            best_offset = self.cfg.default_offset_db
            best_score = -np.inf

            for offset_val in search_values:
                # 临时设置这个邻区对的 offset
                self.offsets[pair] = float(offset_val)

                # 在训练集上评估
                score = self._evaluate_score(train_trajectories)

                if score > best_score:
                    best_score = score
                    best_offset = float(offset_val)

            self.offsets[pair] = best_offset

        if verbose:
            print(f"\n优化完成！邻区对 offset 分布：")
            offsets_arr = list(self.offsets.values())
            print(f"  均值：{np.mean(offsets_arr):.2f} dB")
            print(f"  范围：[{min(offsets_arr):.1f}, {max(offsets_arr):.1f}] dB")
            print(f"  各邻区对 offset：")
            for pair, offset in sorted(self.offsets.items()):
                print(f"    BS{pair[0]} → BS{pair[1]}: {offset:.1f} dB")

    def _evaluate_score(self, trajectories: List[TrajectoryData]) -> float:
        """
        计算当前 offset 配置在轨迹集上的综合得分

        得分 = 平均 SINR - ho_penalty × 平均切换次数 - rlf_penalty × RLF次数
        """
        total_sinr = 0.0
        total_ho = 0
        total_rlf = 0
        total_slots = 0

        for traj in trajectories:
            serving = self.run(traj)
            result = evaluate_trajectory(traj, serving)
            total_sinr += result.mean_sinr_db * traj.num_slots
            total_ho += result.ho_count
            total_rlf += result.rlf_count
            total_slots += traj.num_slots

        mean_sinr = total_sinr / max(total_slots, 1)
        mean_ho = total_ho / max(len(trajectories), 1)

        score = (self.cfg.sinr_weight * mean_sinr
                 - self.cfg.ho_count_penalty * mean_ho
                 - self.cfg.rlf_penalty * total_rlf)
        return score

    def evaluate_on_trajectories(
        self,
        trajectories: List[TrajectoryData],
        algorithm_name: str = "A3",
    ) -> AlgorithmResult:
        """
        在轨迹集上评估 A3 策略

        参数：
            trajectories:   轨迹列表
            algorithm_name: 算法名称（用于结果标注）

        返回：
            AlgorithmResult
        """
        traj_results = []
        for traj in tqdm(trajectories, desc=f"评估 {algorithm_name}"):
            serving = self.run(traj)
            result = evaluate_trajectory(traj, serving)
            traj_results.append(result)

        return aggregate_results(algorithm_name, traj_results)

    def save_offsets(self, path: str) -> None:
        """保存优化后的 offset 到文件"""
        import json
        data = {f"{k[0]},{k[1]}": v for k, v in self.offsets.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"A3 offset 已保存到：{path}")

    def load_offsets(self, path: str) -> None:
        """从文件加载 offset"""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.offsets = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data.items()
        }
        print(f"A3 offset 已从 {path} 加载")