"""
evaluator.py — 切换策略质量评估

评估指标（独立于切换决策，不参与训练）：
  - 均值 SINR [dB]
  - 5th percentile SINR [dB]
  - 切换次数
  - RLF 次数（SINR 持续低于阈值）
  - 乒乓率（1s 内来回切换）
  - 推理时延 [μs]（深度学习模型）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import EvalConfig, HandoverConfig, EVAL_CFG, HO_CFG
from data_loader import TrajectoryData


@dataclass
class TrajectoryResult:
    """单条轨迹的评估结果"""
    traj_id: int
    speed_kmh: float
    traj_type: str
    num_slots: int
    serving_cells: np.ndarray   # [T] 切换策略决定的服务小区序列
    serving_sinr: np.ndarray    # [T] 每时隙的服务小区 SINR
    mean_sinr_db: float = 0.0
    p5_sinr_db: float = 0.0
    ho_count: int = 0
    rlf_count: int = 0
    ping_pong_rate: float = 0.0
    inference_time_us: float = 0.0


@dataclass
class AlgorithmResult:
    """算法在所有测试轨迹上的汇总结果"""
    algorithm_name: str
    traj_results: List[TrajectoryResult] = field(default_factory=list)
    mean_sinr_db: float = 0.0
    p5_sinr_db: float = 0.0
    mean_ho_count: float = 0.0
    total_rlf_count: int = 0
    mean_ping_pong_rate: float = 0.0
    mean_inference_time_us: float = 0.0
    by_speed: Dict[float, Dict] = field(default_factory=dict)


def evaluate_trajectory(
    traj: TrajectoryData,
    serving_cells: np.ndarray,
    inference_time_us: float = 0.0,
    eval_cfg: EvalConfig = EVAL_CFG,
    ho_cfg: HandoverConfig = HO_CFG,
) -> TrajectoryResult:
    """评估单条轨迹的切换策略质量。"""
    T = traj.num_slots
    dt = 0.04  # 40ms per slot

    serving_sinr = np.array([
        traj.sinr[t, serving_cells[t]] for t in range(T)
    ], dtype=np.float32)

    # RLF：SINR 持续低于阈值超过 rlf_duration_ms
    rlf_slots = int(ho_cfg.rlf_duration_ms / (dt * 1000))
    rlf_count = 0
    consecutive = 0
    in_rlf = False
    for t in range(T):
        if serving_sinr[t] < ho_cfg.rlf_sinr_threshold_db:
            consecutive += 1
            if consecutive >= rlf_slots and not in_rlf:
                rlf_count += 1
                in_rlf = True
        else:
            consecutive = 0
            in_rlf = False

    # 乒乓：切换后在 pp_window 内切换回来
    pp_window = eval_cfg.ping_pong_window_slots
    pp_count = 0
    total_ho = 0
    for t in range(1, T):
        if serving_cells[t] != serving_cells[t - 1]:
            total_ho += 1
            prev = serving_cells[t - 1]
            if prev in serving_cells[t:min(t + pp_window, T)]:
                pp_count += 1

    return TrajectoryResult(
        traj_id=traj.traj_id,
        speed_kmh=traj.speed_kmh,
        traj_type=traj.traj_type,
        num_slots=T,
        serving_cells=serving_cells,
        serving_sinr=serving_sinr,
        mean_sinr_db=float(np.mean(serving_sinr)),
        p5_sinr_db=float(np.percentile(serving_sinr, 5)),
        ho_count=int(np.sum(np.diff(serving_cells) != 0)),
        rlf_count=rlf_count,
        ping_pong_rate=pp_count / max(total_ho, 1),
        inference_time_us=inference_time_us,
    )


def aggregate_results(
    algorithm_name: str,
    traj_results: List[TrajectoryResult],
) -> AlgorithmResult:
    """汇总所有轨迹的评估结果。"""
    if not traj_results:
        return AlgorithmResult(algorithm_name=algorithm_name)

    all_sinr = np.concatenate([r.serving_sinr for r in traj_results])

    by_speed = {}
    for speed in sorted(set(r.speed_kmh for r in traj_results)):
        rs = [r for r in traj_results if abs(r.speed_kmh - speed) < 1.0]
        s = np.concatenate([r.serving_sinr for r in rs])
        by_speed[speed] = {
            "mean_sinr_db": float(np.mean(s)),
            "p5_sinr_db": float(np.percentile(s, 5)),
            "mean_ho_count": float(np.mean([r.ho_count for r in rs])),
            "rlf_count": int(sum(r.rlf_count for r in rs)),
            "ping_pong_rate": float(np.mean([r.ping_pong_rate for r in rs])),
            "num_trajs": len(rs),
        }

    return AlgorithmResult(
        algorithm_name=algorithm_name,
        traj_results=traj_results,
        mean_sinr_db=float(np.mean(all_sinr)),
        p5_sinr_db=float(np.percentile(all_sinr, 5)),
        mean_ho_count=float(np.mean([r.ho_count for r in traj_results])),
        total_rlf_count=int(sum(r.rlf_count for r in traj_results)),
        mean_ping_pong_rate=float(np.mean([r.ping_pong_rate for r in traj_results])),
        mean_inference_time_us=float(np.mean([r.inference_time_us for r in traj_results])),
        by_speed=by_speed,
    )


def print_result(result: AlgorithmResult) -> None:
    """打印单个算法的评估结果。"""
    print(f"\n{'='*55}")
    print(f"算法：{result.algorithm_name}")
    print(f"{'='*55}")
    print(f"  均值 SINR：      {result.mean_sinr_db:+.2f} dB")
    print(f"  P5 SINR：        {result.p5_sinr_db:+.2f} dB")
    print(f"  平均切换次数：   {result.mean_ho_count:.1f}")
    print(f"  总 RLF 次数：    {result.total_rlf_count}")
    print(f"  乒乓率：         {result.mean_ping_pong_rate*100:.1f}%")
    if result.mean_inference_time_us > 0:
        print(f"  推理时延：       {result.mean_inference_time_us:.0f} μs")
    if result.by_speed:
        print(f"  按速度：")
        for spd, m in sorted(result.by_speed.items()):
            print(f"    {spd:.0f} km/h: SINR={m['mean_sinr_db']:+.2f} dB, "
                  f"HO={m['mean_ho_count']:.1f}, RLF={m['rlf_count']}, "
                  f"PP={m['ping_pong_rate']*100:.1f}%")


def compare_results(results: List[AlgorithmResult]) -> None:
    """对比多个算法的评估结果。"""
    print(f"\n{'='*75}")
    print("算法对比")
    print(f"{'='*75}")
    print(f"{'算法':<20} {'均值SINR':>9} {'P5 SINR':>9} {'切换次数':>9} {'RLF':>5} {'乒乓率':>7} {'推理时延':>9}")
    print("-" * 75)

    best_sinr = max(r.mean_sinr_db for r in results)
    min_ho = min(r.mean_ho_count for r in results)
    min_rlf = min(r.total_rlf_count for r in results)

    for r in results:
        s = "★" if abs(r.mean_sinr_db - best_sinr) < 0.01 else " "
        h = "★" if abs(r.mean_ho_count - min_ho) < 0.01 else " "
        f = "★" if r.total_rlf_count == min_rlf else " "
        t = f"{r.mean_inference_time_us:.0f}μs" if r.mean_inference_time_us > 0 else "N/A"
        print(f"{r.algorithm_name:<20} {r.mean_sinr_db:>+8.2f}{s} "
              f"{r.p5_sinr_db:>+8.2f}  {r.mean_ho_count:>8.1f}{h} "
              f"{r.total_rlf_count:>4}{f} {r.mean_ping_pong_rate*100:>6.1f}% {t:>9}")

    print("-" * 75)
    print("★ = 最优")


def save_results(results: List[AlgorithmResult], output_path: str) -> None:
    """保存评估结果到 JSON 文件。"""
    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        r.algorithm_name: {
            "mean_sinr_db": r.mean_sinr_db,
            "p5_sinr_db": r.p5_sinr_db,
            "mean_ho_count": r.mean_ho_count,
            "total_rlf_count": r.total_rlf_count,
            "mean_ping_pong_rate": r.mean_ping_pong_rate,
            "mean_inference_time_us": r.mean_inference_time_us,
            "by_speed": r.by_speed,
        }
        for r in results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)