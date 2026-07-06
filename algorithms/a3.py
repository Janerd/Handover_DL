"""
algorithms/a3.py — A3 切换算法（3GPP TS 36.331 / 38.331）

触发条件：
    RSRP_neighbor - RSRP_serving > offset(serving, neighbor) + hysteresis
持续满足 TTT 后执行切换。

优化：在训练集上对每个邻区对独立优化 offset，目标为最大化 SINR。
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from config import A3Config, HandoverConfig, A3_CFG, HO_CFG
from data_loader import TrajectoryData, get_neighbor_pairs
from evaluator import evaluate_trajectory, aggregate_results, AlgorithmResult


class A3HandoverPolicy:
    """A3 切换策略（每个邻区对独立 offset）"""

    def __init__(
        self,
        cfg: A3Config = A3_CFG,
        ho_cfg: HandoverConfig = HO_CFG,
        neighbor_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        self.cfg = cfg
        self.ho_cfg = ho_cfg
        self.neighbor_pairs = neighbor_pairs or get_neighbor_pairs()
        self.offsets: Dict[Tuple[int, int], float] = {
            p: cfg.default_offset_db for p in self.neighbor_pairs
        }
        self._ttt_slots = max(1, int(cfg.ttt_ms / 40.0))

    def run(self, traj: TrajectoryData) -> np.ndarray:
        """在单条轨迹上执行 A3 切换策略，返回服务小区序列 [T]。"""
        T = traj.num_slots
        C = traj.num_cells
        rsrp = traj.rsrp_l3

        serving = np.zeros(T, dtype=np.int32)
        serving[0] = int(np.argmax(rsrp[0]))
        ttt = np.zeros(C, dtype=np.int32)
        cooldown = 0

        for t in range(1, T):
            cur = serving[t - 1]
            cur_rsrp = rsrp[t, cur]

            for nb in range(C):
                if nb == cur:
                    ttt[nb] = 0
                    continue
                offset = self.offsets.get((cur, nb), self.cfg.default_offset_db)
                if rsrp[t, nb] > cur_rsrp + offset + self.cfg.hysteresis_db:
                    ttt[nb] += 1
                else:
                    ttt[nb] = 0

            best_nb = -1
            best_rsrp = -np.inf
            if cooldown <= 0:
                for nb in range(C):
                    if nb != cur and ttt[nb] >= self._ttt_slots:
                        if rsrp[t, nb] > best_rsrp:
                            best_rsrp = rsrp[t, nb]
                            best_nb = nb

            if best_nb >= 0:
                serving[t] = best_nb
                ttt[:] = 0
                cooldown = self.ho_cfg.ho_cooldown_slots
            else:
                serving[t] = cur
                if cooldown > 0:
                    cooldown -= 1

        return serving

    def optimize(self, train_trajectories: List[TrajectoryData]) -> None:
        """在训练集上优化每个邻区对的 offset。"""
        search = np.arange(
            self.cfg.offset_search_range[0],
            self.cfg.offset_search_range[1] + 1e-9,
            self.cfg.offset_search_step,
        )
        for pair in tqdm(self.neighbor_pairs, desc="优化 A3 offset"):
            best_offset = self.cfg.default_offset_db
            best_score = -np.inf
            for val in search:
                self.offsets[pair] = float(val)
                score = self._score(train_trajectories)
                if score > best_score:
                    best_score = score
                    best_offset = float(val)
            self.offsets[pair] = best_offset

    def _score(self, trajectories: List[TrajectoryData]) -> float:
        total_sinr, total_ho, total_rlf, total_slots = 0.0, 0, 0, 0
        for traj in trajectories:
            r = evaluate_trajectory(traj, self.run(traj))
            total_sinr += r.mean_sinr_db * traj.num_slots
            total_ho += r.ho_count
            total_rlf += r.rlf_count
            total_slots += traj.num_slots
        mean_sinr = total_sinr / max(total_slots, 1)
        mean_ho = total_ho / max(len(trajectories), 1)
        return (self.cfg.sinr_weight * mean_sinr
                - self.cfg.ho_count_penalty * mean_ho
                - self.cfg.rlf_penalty * total_rlf)

    def evaluate_on_trajectories(
        self, trajectories: List[TrajectoryData], name: str = "A3"
    ) -> AlgorithmResult:
        results = [evaluate_trajectory(traj, self.run(traj))
                   for traj in tqdm(trajectories, desc=f"评估 {name}")]
        return aggregate_results(name, results)

    def save_offsets(self, path: str) -> None:
        data = {f"{k[0]},{k[1]}": v for k, v in self.offsets.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_offsets(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.offsets = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data.items()
        }