"""
run_experiment.py
=================
一键运行完整实验流程

使用方法：
    # 家用电脑（有 Sionna，GPU 训练）
    python run_experiment.py --mode home

    # 公司电脑（无 Sionna，CPU 评估）
    python run_experiment.py --mode work

    # 只重建数据集 + 训练（不评估）
    python run_experiment.py --mode home --skip-eval

    # 只评估（已有训练好的模型）
    python run_experiment.py --mode work --skip-train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: str, desc: str = "") -> int:
    """运行命令，打印描述，返回退出码"""
    print(f"\n{'='*60}")
    if desc:
        print(f">>> {desc}")
    print(f">>> {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="一键运行完整实验流程")
    parser.add_argument(
        "--mode", choices=["home", "work"], default="work",
        help="运行模式：home=家用电脑（GPU训练），work=公司电脑（CPU评估）"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="跳过训练（直接评估已有模型）"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="跳过评估（只重建数据集和训练）"
    )
    parser.add_argument(
        "--ho-cost", type=float, default=1.0,
        help="切换代价 [dB]（默认 1.0）"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="未来时间窗口 [slots]（默认 10 = 400ms）"
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    sionna_outputs = Path("C:/PC_Simu/Sionna/outputs") if args.mode == "home" \
        else Path("C:/Users/haojia/Sionna/outputs")

    dataset_name = f"dataset_benefit_h{args.horizon}_c{args.ho_cost:.1f}.npz"
    dataset_path = sionna_outputs / dataset_name
    model_dir = project_dir / "outputs" / "models_benefit"
    eval_dir = project_dir / "outputs" / "evaluation_benefit"

    print(f"\n实验配置：")
    print(f"  模式：{args.mode}")
    print(f"  数据集：{dataset_path}")
    print(f"  模型目录：{model_dir}")
    print(f"  评估目录：{eval_dir}")

    # ---- 步骤 1：重建数据集（仅家用电脑）----
    if args.mode == "home" and not args.skip_train:
        if not dataset_path.exists():
            rc = run(
                f"python {project_dir}/rebuild_dataset.py "
                f"--mode benefit --horizon {args.horizon} --ho-cost {args.ho_cost}",
                desc=f"步骤 1/3：重建数据集（切换收益标签，ho_cost={args.ho_cost} dB）"
            )
            if rc != 0:
                print("❌ 数据集重建失败，终止")
                sys.exit(1)
        else:
            print(f"\n✅ 数据集已存在，跳过重建：{dataset_path}")

    # ---- 步骤 2：训练（仅家用电脑）----
    if args.mode == "home" and not args.skip_train:
        if not dataset_path.exists():
            print(f"❌ 数据集不存在：{dataset_path}")
            sys.exit(1)

        rc = run(
            f"python {project_dir}/train.py "
            f"--model all "
            f"--dataset {dataset_path} "
            f"--outdir {model_dir}",
            desc="步骤 2/3：训练 GRU / TCN / Transformer"
        )
        if rc != 0:
            print("❌ 训练失败，终止")
            sys.exit(1)

    # ---- 步骤 3：评估 ----
    if not args.skip_eval:
        # 检查模型文件是否存在
        missing = []
        for name in ["GRU_best.pt", "TCN_best.pt", "Transformer_best.pt"]:
            if not (model_dir / name).exists():
                missing.append(name)

        if missing:
            print(f"\n⚠️  以下模型文件不存在：{missing}")
            print(f"   请先在家用电脑上训练，然后将 {model_dir} 同步到公司电脑")
            if args.mode == "work":
                sys.exit(1)

        rc = run(
            f"python {project_dir}/evaluate.py "
            f"--model-dir {model_dir} "
            f"--outdir {eval_dir}",
            desc="步骤 3/3：综合评估（A3 + GRU + TCN + Transformer）"
        )
        if rc != 0:
            print("❌ 评估失败")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("✅ 实验完成！")
    if not args.skip_eval:
        print(f"   结果：{eval_dir}/results.json")
        print(f"   图表：{eval_dir}/comparison.png")
        print(f"         {eval_dir}/sinr_cdf.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()