"""
evaluate.py
===========
切换策略综合评估脚本

流程：
  1. 加载测试集轨迹数据
  2. 运行 A3 算法（优化后的 offset）
  3. 运行 GRU / TCN / Transformer（加载训练好的模型）
  4. 用 SINR / 切换次数 / RLF / 乒乓率 / 推理时延 对比所有算法
  5. 保存结果和可视化图

使用方法：
    python evaluate.py
    python evaluate.py --speed 30    # 只评估 30 km/h 轨迹
    python evaluate.py --no-a3-opt  # 跳过 A3 优化（使用默认 offset）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, HO_CFG
from data_loader import load_trajectories, get_neighbor_pairs
from evaluator import compare_results, print_result, save_results, AlgorithmResult
from algorithms.a3 import A3HandoverPolicy
from algorithms.gru import GRUHandoverPolicy, GRUConfig
from algorithms.tcn import TCNHandoverPolicy, TCNConfig
from algorithms.transformer import TransformerHandoverPolicy, TransformerConfig


# =========================================================
# 可视化
# =========================================================

def _setup_font():
    chinese_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()


def plot_comparison(results: List[AlgorithmResult], save_dir: Path) -> None:
    """生成算法对比图"""
    save_dir.mkdir(parents=True, exist_ok=True)

    names = [r.algorithm_name for r in results]
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"][:len(results)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Handover Policy Comparison", fontsize=14, fontweight="bold")

    # ---- 1. 平均 SINR ----
    ax = axes[0, 0]
    sinr_vals = [r.mean_sinr_db for r in results]
    bars = ax.bar(names, sinr_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("Mean SINR [dB]", fontsize=12)
    ax.set_ylabel("SINR [dB]")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, sinr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # ---- 2. 5th Percentile SINR ----
    ax = axes[0, 1]
    p5_vals = [r.p5_sinr_db for r in results]
    bars = ax.bar(names, p5_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("5th Percentile SINR [dB]", fontsize=12)
    ax.set_ylabel("SINR [dB]")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, p5_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # ---- 3. 平均切换次数 ----
    ax = axes[0, 2]
    ho_vals = [r.mean_ho_count for r in results]
    bars = ax.bar(names, ho_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("Mean Handover Count", fontsize=12)
    ax.set_ylabel("Count per trajectory")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, ho_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # ---- 4. RLF 次数 ----
    ax = axes[1, 0]
    rlf_vals = [r.total_rlf_count for r in results]
    bars = ax.bar(names, rlf_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("Total RLF Count", fontsize=12)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, rlf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}", ha="center", va="bottom", fontsize=9)

    # ---- 5. 乒乓率 ----
    ax = axes[1, 1]
    pp_vals = [r.mean_ping_pong_rate * 100 for r in results]
    bars = ax.bar(names, pp_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("Ping-Pong Rate [%]", fontsize=12)
    ax.set_ylabel("Rate [%]")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, pp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # ---- 6. 推理时延（仅 DL 模型）----
    ax = axes[1, 2]
    dl_results = [r for r in results if r.mean_inference_time_us > 0]
    if dl_results:
        dl_names = [r.algorithm_name for r in dl_results]
        dl_colors = [colors[i] for i, r in enumerate(results) if r.mean_inference_time_us > 0]
        infer_vals = [r.mean_inference_time_us for r in dl_results]
        bars = ax.bar(dl_names, infer_vals, color=dl_colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.set_title("Inference Latency [μs]", fontsize=12)
        ax.set_ylabel("Latency [μs]")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, infer_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No DL models evaluated",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Inference Latency [μs]", fontsize=12)

    plt.tight_layout()
    save_path = save_dir / "comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"对比图已保存到：{save_path}")
    plt.close()

    # ---- 按速度分组的 SINR 对比 ----
    all_speeds = set()
    for r in results:
        all_speeds.update(r.by_speed.keys())

    if len(all_speeds) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_speeds))
        speeds = sorted(all_speeds)
        width = 0.8 / len(results)

        for i, r in enumerate(results):
            sinr_by_speed = [r.by_speed.get(s, {}).get("mean_sinr_db", 0) for s in speeds]
            offset = (i - len(results) / 2 + 0.5) * width
            bars = ax.bar(x + offset, sinr_by_speed, width * 0.9,
                          label=r.algorithm_name, color=colors[i], alpha=0.8,
                          edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Speed [km/h]", fontsize=12)
        ax.set_ylabel("Mean SINR [dB]", fontsize=12)
        ax.set_title("Mean SINR by Speed", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f} km/h" for s in speeds])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = save_dir / "sinr_by_speed.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"速度分组对比图已保存到：{save_path}")
        plt.close()


def plot_sinr_cdf(results: List[AlgorithmResult], save_dir: Path) -> None:
    """绘制 SINR CDF 曲线"""
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"][:len(results)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for r, color in zip(results, colors):
        all_sinr = np.concatenate([tr.serving_sinr for tr in r.traj_results])
        sorted_sinr = np.sort(all_sinr)
        cdf = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)
        ax.plot(sorted_sinr, cdf, label=r.algorithm_name, color=color, linewidth=2)

    ax.set_xlabel("SINR [dB]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("SINR CDF Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="5th percentile")
    ax.set_xlim([-20, 40])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_path = save_dir / "sinr_cdf.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"SINR CDF 图已保存到：{save_path}")
    plt.close()


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="切换策略综合评估")
    parser.add_argument("--speed", type=float, default=None,
                        help="速度筛选 [km/h]（默认：全部）")
    parser.add_argument("--no-a3-opt", action="store_true",
                        help="跳过 A3 优化（使用默认 offset=3dB）")
    parser.add_argument("--model-dir", default=str(OUTPUT_DIR / "models"),
                        help="DL 模型目录")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR / "evaluation"),
                        help="评估结果保存目录")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    save_dir = Path(args.outdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载轨迹数据 ----
    print("加载轨迹数据...")
    train_trajs = load_trajectories(split="train", speed_filter=args.speed)
    test_trajs = load_trajectories(split="test", speed_filter=args.speed)

    if not test_trajs:
        print("错误：没有找到测试集轨迹")
        return

    print(f"训练集：{len(train_trajs)} 条，测试集：{len(test_trajs)} 条")

    all_results = []

    # ---- A3 算法 ----
    print("\n" + "="*60)
    print("评估 A3 算法")
    print("="*60)

    neighbor_pairs = get_neighbor_pairs()
    a3_policy = A3HandoverPolicy(neighbor_pairs=neighbor_pairs)

    a3_offset_path = save_dir / "a3_offsets.json"

    if not args.no_a3_opt:
        if a3_offset_path.exists():
            print(f"加载已优化的 A3 offset：{a3_offset_path}")
            a3_policy.load_offsets(str(a3_offset_path))
        else:
            print("在训练集上优化 A3 offset...")
            a3_policy.optimize(train_trajs)
            a3_policy.save_offsets(str(a3_offset_path))
    else:
        print("使用默认 A3 offset（3 dB）")

    a3_result = a3_policy.evaluate_on_trajectories(test_trajs, algorithm_name="A3 (optimized)")
    print_result(a3_result)
    all_results.append(a3_result)

    # ---- DL 模型 ----
    dl_models = [
        ("GRU", GRUHandoverPolicy, GRUConfig, "GRU_best.pt"),
        ("TCN", TCNHandoverPolicy, TCNConfig, "TCN_best.pt"),
        ("Transformer", TransformerHandoverPolicy, TransformerConfig, "Transformer_best.pt"),
    ]

    for model_name, PolicyClass, ConfigClass, model_file in dl_models:
        model_path = model_dir / model_file
        if not model_path.exists():
            print(f"\n跳过 {model_name}（模型文件不存在：{model_path}）")
            print(f"  请先运行：python train.py --model {model_name.lower()}")
            continue

        print(f"\n{'='*60}")
        print(f"评估 {model_name}")
        print(f"{'='*60}")

        cfg = ConfigClass()
        policy = PolicyClass(cfg=cfg)
        policy.build_model()
        policy.load(str(model_path))

        result = policy.evaluate_on_trajectories(test_trajs, algorithm_name=model_name)
        print_result(result)
        all_results.append(result)

    # ---- 汇总对比 ----
    if len(all_results) > 1:
        compare_results(all_results)

    # ---- 保存结果 ----
    save_results(all_results, str(save_dir / "results.json"))

    # ---- 可视化 ----
    if all_results:
        print("\n生成可视化图...")
        plot_comparison(all_results, save_dir)
        plot_sinr_cdf(all_results, save_dir)

    print(f"\n评估完成！结果保存到：{save_dir}")


if __name__ == "__main__":
    main()