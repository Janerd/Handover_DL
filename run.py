"""
run.py — 一键运行入口

使用方法：
    python run.py              # 评估所有算法（使用预训练模型）
    python run.py --train      # 重新训练后评估
    python run.py --check      # 仅检测环境
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str, desc: str) -> int:
    print(f"\n{'─'*55}")
    print(f"▶ {desc}")
    print(f"{'─'*55}")
    return subprocess.run(cmd, shell=True).returncode


def main():
    parser = argparse.ArgumentParser(description="切换策略实验一键运行")
    parser.add_argument("--train", action="store_true", help="重新训练模型")
    parser.add_argument("--check", action="store_true", help="仅检测环境")
    parser.add_argument("--speed", type=float, default=None,
                        help="只评估指定速度的轨迹（30 / 60 / 120 km/h）")
    args = parser.parse_args()

    proj = Path(__file__).parent

    # 环境检测
    rc = run_cmd(f"python {proj}/setup_check.py", "环境检测")
    if rc != 0 or args.check:
        sys.exit(rc)

    # 训练（可选）
    if args.train:
        rc = run_cmd(f"python {proj}/train.py", "训练模型（GRU / TCN / Transformer）")
        if rc != 0:
            print("❌ 训练失败")
            sys.exit(1)

    # 评估
    speed_arg = f"--speed {args.speed}" if args.speed else ""
    rc = run_cmd(
        f"python {proj}/evaluate.py {speed_arg}",
        "综合评估（A3 + GRU + TCN + Transformer）"
    )
    if rc != 0:
        print("❌ 评估失败")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("✅ 完成！结果保存在 outputs/evaluation/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()