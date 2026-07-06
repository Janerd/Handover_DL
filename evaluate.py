"""
evaluate.py — 切换策略综合评估

使用方法：
    python evaluate.py           # 评估所有算法
    python evaluate.py --speed 30  # 只评估 30 km/h 轨迹
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR, MODEL_DIR
from data_loader import load_trajectories, get_neighbor_pairs
from evaluator import compare_results, print_result, save_results, AlgorithmResult
from algorithms.a3 import A3HandoverPolicy
from algorithms.gru import GRUHandoverPolicy, GRUConfig
from algorithms.tcn import TCNHandoverPolicy, TCNConfig
from algorithms.transformer import TransformerHandoverPolicy, TransformerConfig


def _setup_font():
    for font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()


def plot_comparison(results: list, save_dir: Path) -> None:
    """生成算法对比图（6 指标）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    names = [r.algorithm_name for r in results]
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"][:len(results)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Handover Policy Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("均值 SINR [dB]", [r.mean_sinr_db for r in results]),
        ("P5 SINR [dB]", [r.p5_sinr_db for r in results]),
        ("平均切换次数", [r.mean_ho_count for r in results]),
        ("总 RLF 次数", [r.total_rlf_count for r in results]),
        ("乒乓率 [%]", [r.mean_ping_pong_rate * 100 for r in results]),
    ]

    for ax, (title, vals) in zip(axes.flat, metrics):
        bars = ax.bar(names, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # 推理时延（仅 DL 模型）
    ax = axes[1, 2]
    dl = [r for r in results if r.mean_inference_time_us > 0]
    if dl:
        dl_colors = [colors[i] for i, r in enumerate(results) if r.mean_inference_time_us > 0]
        bars = ax.bar([r.algorithm_name for r in dl],
                      [r.mean_inference_time_us for r in dl],
                      color=dl_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title("推理时延 [μs]", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, r in zip(bars, dl):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{r.mean_inference_time_us:.0f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "无 DL 模型", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("推理时延 [μs]", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sinr_cdf(results: list, save_dir: Path) -> None:
    """绘制 SINR CDF 曲线。"""
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"][:len(results)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, c in zip(results, colors):
        sinr = np.sort(np.concatenate([tr.serving_sinr for tr in r.traj_results]))
        ax.plot(sinr, np.arange(1, len(sinr) + 1) / len(sinr), label=r.algorithm_name, color=c, linewidth=2)
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SINR [dB]")
    ax.set_ylabel("CDF")
    ax.set_title("SINR CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-20, 40])
    plt.tight_layout()
    plt.savefig(save_dir / "sinr_cdf.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="切换策略综合评估")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--outdir", default=str(OUTPUT_DIR / "evaluation"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    save_dir = Path(args.outdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载轨迹数据
    train_trajs = load_trajectories(split="train")
    test_trajs = load_trajectories(split="test")
    if args.speed is not None:
        train_trajs = [t for t in train_trajs if abs(t.speed_kmh - args.speed) < 1.0]
        test_trajs = [t for t in test_trajs if abs(t.speed_kmh - args.speed) < 1.0]

    if not test_trajs:
        print("未找到测试集轨迹")
        return

    print(f"训练集：{len(train_trajs)} 条，测试集：{len(test_trajs)} 条")
    all_results = []

    # A3 算法
    a3 = A3HandoverPolicy(neighbor_pairs=get_neighbor_pairs())
    offset_cache = save_dir / "a3_offsets.json"
    if offset_cache.exists():
        a3.load_offsets(str(offset_cache))
        print("已加载 A3 offset 缓存")
    else:
        print("优化 A3 offset...")
        a3.optimize(train_trajs)
        a3.save_offsets(str(offset_cache))
    r = a3.evaluate_on_trajectories(test_trajs, name="A3 (optimized)")
    print_result(r)
    all_results.append(r)

    # DL 模型
    for name, PolicyClass, ConfigClass, fname in [
        ("GRU", GRUHandoverPolicy, GRUConfig, "GRU_best.pt"),
        ("TCN", TCNHandoverPolicy, TCNConfig, "TCN_best.pt"),
        ("Transformer", TransformerHandoverPolicy, TransformerConfig, "Transformer_best.pt"),
    ]:
        pt = model_dir / fname
        if not pt.exists():
            print(f"跳过 {name}（模型文件不存在：{pt}）")
            continue
        p = PolicyClass(cfg=ConfigClass())
        p.build_model()
        p.load(str(pt))
        r = p.evaluate_on_trajectories(test_trajs, name=name)
        print_result(r)
        all_results.append(r)

    if len(all_results) > 1:
        compare_results(all_results)

    save_results(all_results, str(save_dir / "results.json"))
    plot_comparison(all_results, save_dir)
    plot_sinr_cdf(all_results, save_dir)
    print(f"\n结果已保存：{save_dir}")


if __name__ == "__main__":
    main()