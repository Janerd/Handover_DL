"""
evaluator.py
============
切换策略质量评估

评估指标（独立于切换决策，避免循环论证）：
  - 平均 SINR [dB]：服务小区的 SINR 均值
  - 5th percentile SINR [dB]：最差 5% 时刻的 SINR
  - 切换次数：轨迹中发生切换的次数
  - RLF 次数：无线链路失败次数（SINR 持续低于阈值）
  - 乒乓率：1s 内来回切换的比例
  - 推理时延 [μs]：DL 模型的单次推理时间

注意：
  SINR 来自 Sionna 射线追踪（物理量），不是决策输入（RSRP），
  因此用 SINR 评估不构成循环论证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import EvalConfig, HandoverConfig, EVAL_CFG, HO_CFG
from data_loader import TrajectoryData


# =========================================================
# 评估结果数据结构
# =========================================================

@dataclass
class TrajectoryResult:
    """单条轨迹的评估结果"""
    traj_id: int
    speed_kmh: float
    traj_type: str
    num_slots: int

    # 服务小区序列（由切换策略决定）[T]
    serving_cells: np.ndarray

    # 每时隙的服务小区 SINR [T]
    serving_sinr: np.ndarray

    # 评估指标
    mean_sinr_db: float = 0.0
    p5_sinr_db: float = 0.0
    ho_count: int = 0
    rlf_count: int = 0
    ping_pong_rate: float = 0.0
    inference_time_us: float = 0.0  # 仅 DL 模型有效


@dataclass
class AlgorithmResult:
    """算法在所有测试轨迹上的汇总结果"""
    algorithm_name: str
    traj_results: List[TrajectoryResult] = field(default_factory=list)

    # 汇总指标（跨所有轨迹）
    mean_sinr_db: float = 0.0
    p5_sinr_db: float = 0.0
    mean_ho_count: float = 0.0
    total_rlf_count: int = 0
    mean_ping_pong_rate: float = 0.0
    mean_inference_time_us: float = 0.0

    # 按速度分组的指标
    by_speed: Dict[float, Dict] = field(default_factory=dict)


# =========================================================
# 单条轨迹评估
# =========================================================

def evaluate_trajectory(
    traj: TrajectoryData,
    serving_cells: np.ndarray,
    inference_time_us: float = 0.0,
    eval_cfg: EvalConfig = EVAL_CFG,
    ho_cfg: HandoverConfig = HO_CFG,
) -> TrajectoryResult:
    """
    评估单条轨迹的切换策略质量

    参数：
        traj:              轨迹数据（包含 SINR 等信道数据）
        serving_cells:     [T] 切换策略决定的服务小区序列
        inference_time_us: DL 模型的平均推理时延 [μs]
        eval_cfg:          评估配置
        ho_cfg:            切换配置

    返回：
        TrajectoryResult
    """
    T = traj.num_slots
    C = traj.num_cells
    dt = 0.04  # 40ms per slot

    # 确保 serving_cells 长度正确
    assert len(serving_cells) == T, \
        f"serving_cells 长度 {len(serving_cells)} 与轨迹长度 {T} 不匹配"

    # ---- 1. 每时隙的服务小区 SINR ----
    serving_sinr = np.array([
        traj.sinr[t, serving_cells[t]] for t in range(T)
    ], dtype=np.float32)

    # ---- 2. 平均 SINR ----
    mean_sinr = float(np.mean(serving_sinr))

    # ---- 3. 5th percentile SINR ----
    p5_sinr = float(np.percentile(serving_sinr, 5))

    # ---- 4. 切换次数 ----
    ho_count = int(np.sum(np.diff(serving_cells) != 0))

    # ---- 5. RLF 次数 ----
    # RLF：SINR 持续低于阈值超过 rlf_duration_ms
    rlf_threshold = ho_cfg.rlf_sinr_threshold_db
    rlf_duration_slots = int(ho_cfg.rlf_duration_ms / (dt * 1000))

    rlf_count = 0
    below_threshold = serving_sinr < rlf_threshold
    consecutive = 0
    in_rlf = False

    for t in range(T):
        if below_threshold[t]:
            consecutive += 1
            if consecutive >= rlf_duration_slots and not in_rlf:
                rlf_count += 1
                in_rlf = True
        else:
            consecutive = 0
            in_rlf = False

    # ---- 6. 乒乓率 ----
    # 乒乓：在 ping_pong_window_slots 内，切换到某小区后又切换回来
    pp_window = eval_cfg.ping_pong_window_slots
    ping_pong_count = 0
    total_ho = 0

    for t in range(1, T):
        if serving_cells[t] != serving_cells[t - 1]:
            total_ho += 1
            # 检查 pp_window 内是否切换回来
            prev_cell = serving_cells[t - 1]
            end = min(t + pp_window, T)
            future_cells = serving_cells[t:end]
            if prev_cell in future_cells:
                ping_pong_count += 1

    ping_pong_rate = ping_pong_count / max(total_ho, 1)

    return TrajectoryResult(
        traj_id=traj.traj_id,
        speed_kmh=traj.speed_kmh,
        traj_type=traj.traj_type,
        num_slots=T,
        serving_cells=serving_cells,
        serving_sinr=serving_sinr,
        mean_sinr_db=mean_sinr,
        p5_sinr_db=p5_sinr,
        ho_count=ho_count,
        rlf_count=rlf_count,
        ping_pong_rate=ping_pong_rate,
        inference_time_us=inference_time_us,
    )


# =========================================================
# 汇总评估
# =========================================================

def aggregate_results(
    algorithm_name: str,
    traj_results: List[TrajectoryResult],
) -> AlgorithmResult:
    """
    汇总所有轨迹的评估结果

    参数：
        algorithm_name: 算法名称
        traj_results:   各轨迹的评估结果

    返回：
        AlgorithmResult
    """
    if not traj_results:
        return AlgorithmResult(algorithm_name=algorithm_name)

    # 全局汇总
    all_sinr = np.concatenate([r.serving_sinr for r in traj_results])
    mean_sinr = float(np.mean(all_sinr))
    p5_sinr = float(np.percentile(all_sinr, 5))
    mean_ho = float(np.mean([r.ho_count for r in traj_results]))
    total_rlf = int(np.sum([r.rlf_count for r in traj_results]))
    mean_pp = float(np.mean([r.ping_pong_rate for r in traj_results]))
    mean_infer = float(np.mean([r.inference_time_us for r in traj_results]))

    # 按速度分组
    by_speed = {}
    speeds = set(r.speed_kmh for r in traj_results)
    for speed in sorted(speeds):
        speed_results = [r for r in traj_results if abs(r.speed_kmh - speed) < 1.0]
        speed_sinr = np.concatenate([r.serving_sinr for r in speed_results])
        by_speed[speed] = {
            "mean_sinr_db": float(np.mean(speed_sinr)),
            "p5_sinr_db": float(np.percentile(speed_sinr, 5)),
            "mean_ho_count": float(np.mean([r.ho_count for r in speed_results])),
            "rlf_count": int(np.sum([r.rlf_count for r in speed_results])),
            "ping_pong_rate": float(np.mean([r.ping_pong_rate for r in speed_results])),
            "num_trajs": len(speed_results),
        }

    return AlgorithmResult(
        algorithm_name=algorithm_name,
        traj_results=traj_results,
        mean_sinr_db=mean_sinr,
        p5_sinr_db=p5_sinr,
        mean_ho_count=mean_ho,
        total_rlf_count=total_rlf,
        mean_ping_pong_rate=mean_pp,
        mean_inference_time_us=mean_infer,
        by_speed=by_speed,
    )


# =========================================================
# 结果打印和比较
# =========================================================

def print_result(result: AlgorithmResult) -> None:
    """打印单个算法的评估结果"""
    print(f"\n{'='*60}")
    print(f"算法：{result.algorithm_name}")
    print(f"{'='*60}")
    print(f"  平均 SINR：        {result.mean_sinr_db:+.2f} dB")
    print(f"  5th percentile SINR：{result.p5_sinr_db:+.2f} dB")
    print(f"  平均切换次数：     {result.mean_ho_count:.1f} 次/轨迹")
    print(f"  总 RLF 次数：      {result.total_rlf_count} 次")
    print(f"  乒乓率：           {result.mean_ping_pong_rate*100:.1f}%")
    if result.mean_inference_time_us > 0:
        print(f"  推理时延：         {result.mean_inference_time_us:.1f} μs")

    if result.by_speed:
        print(f"\n  按速度分组：")
        for speed, metrics in sorted(result.by_speed.items()):
            print(f"    {speed:.0f} km/h ({metrics['num_trajs']} 条)："
                  f" SINR={metrics['mean_sinr_db']:+.2f} dB,"
                  f" HO={metrics['mean_ho_count']:.1f},"
                  f" RLF={metrics['rlf_count']},"
                  f" PP={metrics['ping_pong_rate']*100:.1f}%")


def compare_results(results: List[AlgorithmResult]) -> None:
    """对比多个算法的评估结果（表格形式）"""
    if not results:
        return

    print(f"\n{'='*80}")
    print("算法对比汇总")
    print(f"{'='*80}")

    # 表头
    header = f"{'算法':<20} {'均值SINR':>10} {'P5 SINR':>10} {'切换次数':>10} {'RLF':>6} {'乒乓率':>8} {'推理时延':>10}"
    print(header)
    print("-" * 80)

    # 找到最优值（用于标注）
    best_sinr = max(r.mean_sinr_db for r in results)
    best_p5 = max(r.p5_sinr_db for r in results)
    min_ho = min(r.mean_ho_count for r in results)
    min_rlf = min(r.total_rlf_count for r in results)
    min_pp = min(r.mean_ping_pong_rate for r in results)

    for r in results:
        sinr_mark = "★" if abs(r.mean_sinr_db - best_sinr) < 0.01 else " "
        p5_mark = "★" if abs(r.p5_sinr_db - best_p5) < 0.01 else " "
        ho_mark = "★" if abs(r.mean_ho_count - min_ho) < 0.01 else " "
        rlf_mark = "★" if r.total_rlf_count == min_rlf else " "
        pp_mark = "★" if abs(r.mean_ping_pong_rate - min_pp) < 0.001 else " "

        infer_str = f"{r.mean_inference_time_us:.1f} μs" if r.mean_inference_time_us > 0 else "N/A"

        print(f"{r.algorithm_name:<20} "
              f"{r.mean_sinr_db:>+9.2f}{sinr_mark} "
              f"{r.p5_sinr_db:>+9.2f}{p5_mark} "
              f"{r.mean_ho_count:>9.1f}{ho_mark} "
              f"{r.total_rlf_count:>5}{rlf_mark} "
              f"{r.mean_ping_pong_rate*100:>7.1f}%{pp_mark} "
              f"{infer_str:>10}")

    print("-" * 80)
    print("★ = 最优值")

    # 按速度分组对比
    all_speeds = set()
    for r in results:
        all_speeds.update(r.by_speed.keys())

    if all_speeds:
        print(f"\n按速度分组（均值 SINR [dB]）：")
        speed_header = f"{'算法':<20}" + "".join(f" {s:.0f}km/h" for s in sorted(all_speeds))
        print(speed_header)
        print("-" * (20 + 8 * len(all_speeds)))
        for r in results:
            row = f"{r.algorithm_name:<20}"
            for speed in sorted(all_speeds):
                if speed in r.by_speed:
                    row += f" {r.by_speed[speed]['mean_sinr_db']:>+6.2f}"
                else:
                    row += f" {'N/A':>6}"
            print(row)


def save_results(
    results: List[AlgorithmResult],
    output_path: str,
) -> None:
    """保存评估结果到 npz 文件"""
    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for r in results:
        summary[r.algorithm_name] = {
            "mean_sinr_db": r.mean_sinr_db,
            "p5_sinr_db": r.p5_sinr_db,
            "mean_ho_count": r.mean_ho_count,
            "total_rlf_count": r.total_rlf_count,
            "mean_ping_pong_rate": r.mean_ping_pong_rate,
            "mean_inference_time_us": r.mean_inference_time_us,
            "by_speed": r.by_speed,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到：{output_path}")