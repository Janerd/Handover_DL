"""
rebuild_dataset.py
==================
从已有的 trajectory_data.npz 重建数据集，使用区间平均 SINR 作为标签。

新标签定义：
    Y = argmax(mean(SINR[t : t + pred_horizon]))
    即：未来 pred_horizon slots 内，平均 SINR 最高的小区

优势：
    - 过滤短暂波动，只有持续更好的小区才会被选为目标
    - 不需要重新运行 Sionna 仿真（约 5 小时）
    - 只需约 1 分钟重建数据集

使用方法：
    python rebuild_dataset.py
    python rebuild_dataset.py --horizon 10   # 使用 10 slots（400ms）平均
    python rebuild_dataset.py --horizon 20   # 使用 20 slots（800ms）平均
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import TRAJECTORY_DATA_PATH, SIONNA_OUTPUT_DIR, DATA_CFG
from data_loader import TrajectoryData, normalize_X


def build_dataset_avg_label(
    trajectory_data_path: Path,
    window_size: int = 10,
    pred_horizon: int = 5,
    avg_horizon: int = 10,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict:
    """
    从轨迹数据重建数据集，使用区间平均 SINR 作为标签。

    参数：
        trajectory_data_path: trajectory_data.npz 路径
        window_size:          特征窗口大小（历史时隙数）
        pred_horizon:         预测起始偏移（从当前时隙往后多少步开始计算平均）
        avg_horizon:          平均 SINR 的时间窗口长度（时隙数）
        train_ratio:          训练集比例
        val_ratio:            验证集比例

    返回：
        包含 X_raw, Y_cell, Y_sinr, split_masks 的字典
    """
    print(f"加载轨迹数据：{trajectory_data_path}")
    raw = np.load(trajectory_data_path, allow_pickle=True)
    num_traj = len(raw["traj_ids"])
    print(f"  总轨迹数：{num_traj}")
    print(f"  标签定义：未来 [{pred_horizon}, {pred_horizon + avg_horizon}) slots 的平均 SINR 最优小区")

    W = window_size
    C = DATA_CFG.num_cells

    # 预估样本总数
    total_samples = 0
    for i in range(num_traj):
        T = int(raw["num_slots"][i])
        total_samples += max(0, T - W - pred_horizon - avg_horizon + 1)

    print(f"  预估样本总数：{total_samples}")

    X_raw = np.zeros((total_samples, W, DATA_CFG.num_features), dtype=np.float32)
    Y_cell = np.zeros(total_samples, dtype=np.int64)
    Y_sinr = np.zeros((total_samples, C), dtype=np.float32)
    meta_speed = np.zeros(total_samples, dtype=np.float32)
    meta_traj = np.zeros(total_samples, dtype=np.int32)
    meta_split = np.empty(total_samples, dtype=object)

    sample_idx = 0

    for i in range(num_traj):
        traj_split = str(raw["splits"][i])
        traj_speed = float(raw["speed_kmh"][i])
        T = int(raw["num_slots"][i])

        # 构建 TrajectoryData 对象（用于 get_feature_matrix）
        traj = TrajectoryData(
            traj_id=int(raw["traj_ids"][i]),
            speed_kmh=traj_speed,
            traj_type=str(raw["traj_types"][i]),
            split=traj_split,
            num_slots=T,
            rsrp_l3=raw["RSRP_l3"][i].astype(np.float32),
            sinr=raw["SINR"][i].astype(np.float32),
            rsrq=raw["RSRQ"][i].astype(np.float32),
            doppler=raw["doppler"][i].astype(np.float32),
            beam_id=raw["beam_id"][i].astype(np.int32),
            delay_spread=raw["delay_spread"][i].astype(np.float32),
            k_factor=raw["k_factor"][i].astype(np.float32),
            min_tau=raw["min_tau"][i].astype(np.float32),
            pos=raw["pos"][i].astype(np.float32),
            serving_l3=raw["serving_l3"][i].astype(np.int32),
        )

        # 特征矩阵（已归一化）[T, F]
        feat = traj.get_feature_matrix()
        sinr_seq = traj.sinr  # [T, C]

        for slot in range(W, T - pred_horizon - avg_horizon + 1):
            # 特征窗口：[slot-W, slot)
            X_raw[sample_idx] = feat[slot - W : slot]

            # 标签：未来 [slot+pred_horizon, slot+pred_horizon+avg_horizon) 的平均 SINR
            future_start = slot + pred_horizon
            future_end = slot + pred_horizon + avg_horizon
            avg_sinr = np.mean(sinr_seq[future_start:future_end], axis=0)  # [C]

            Y_cell[sample_idx] = int(np.argmax(avg_sinr))
            Y_sinr[sample_idx] = avg_sinr

            meta_speed[sample_idx] = traj_speed
            meta_traj[sample_idx] = i
            meta_split[sample_idx] = traj_split
            sample_idx += 1

    actual_samples = sample_idx
    X_raw = X_raw[:actual_samples]
    Y_cell = Y_cell[:actual_samples]
    Y_sinr = Y_sinr[:actual_samples]
    meta_speed = meta_speed[:actual_samples]
    meta_traj = meta_traj[:actual_samples]
    meta_split = meta_split[:actual_samples]

    print(f"  实际样本数：{actual_samples}")

    # 按 split 字段划分（与原始数据集保持一致）
    train_mask = meta_split == "train"
    val_mask = meta_split == "val"
    test_mask = meta_split == "test"

    print(f"  训练集：{train_mask.sum()} 样本")
    print(f"  验证集：{val_mask.sum()} 样本")
    print(f"  测试集：{test_mask.sum()} 样本")

    # 标签分布
    print(f"\n  标签分布（测试集）：")
    y_test = Y_cell[test_mask]
    for c in range(C):
        count = int(np.sum(y_test == c))
        ratio = count / max(len(y_test), 1) * 100
        print(f"    cell_{c}: {count} ({ratio:.1f}%)")

    return {
        "X_raw": X_raw,
        "Y_cell": Y_cell,
        "Y_sinr": Y_sinr,
        "split_train": train_mask,
        "split_val": val_mask,
        "split_test": test_mask,
        "meta_speed": meta_speed,
        "meta_traj": meta_traj,
        "avg_horizon": avg_horizon,
        "pred_horizon": pred_horizon,
        "window_size": window_size,
    }


def main():
    parser = argparse.ArgumentParser(description="重建数据集（区间平均 SINR 标签）")
    parser.add_argument("--horizon", type=int, default=10,
                        help="平均 SINR 的时间窗口长度（时隙数，默认 10 = 400ms）")
    parser.add_argument("--pred-offset", type=int, default=1,
                        help="预测起始偏移（默认 1，即从下一时隙开始）")
    parser.add_argument("--outdir", default=None,
                        help="输出目录（默认：Sionna outputs 目录）")
    args = parser.parse_args()

    if args.outdir is None:
        out_dir = SIONNA_OUTPUT_DIR
    else:
        out_dir = Path(args.outdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"dataset_avg{args.horizon}.npz"

    print(f"\n重建数据集（区间平均 SINR，窗口={args.horizon} slots = {args.horizon*40}ms）")
    print(f"输出路径：{save_path}")

    result = build_dataset_avg_label(
        trajectory_data_path=TRAJECTORY_DATA_PATH,
        window_size=DATA_CFG.window_size,
        pred_horizon=args.pred_offset,
        avg_horizon=args.horizon,
    )

    np.savez_compressed(
        save_path,
        X_raw=result["X_raw"],
        Y_cell=result["Y_cell"],
        Y_sinr=result["Y_sinr"],
        split_train=result["split_train"],
        split_val=result["split_val"],
        split_test=result["split_test"],
        meta_speed=result["meta_speed"],
        meta_traj=result["meta_traj"],
        num_cells=np.array(DATA_CFG.num_cells),
        num_features=np.array(DATA_CFG.num_features),
        window_size=np.array(result["window_size"]),
        pred_horizon=np.array(result["pred_horizon"]),
        avg_horizon=np.array(result["avg_horizon"]),
    )

    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"\n数据集已保存：{save_path}（{size_mb:.1f} MB）")
    print(f"\n下一步：使用新数据集训练")
    print(f"  python train.py --model all --dataset {save_path}")


if __name__ == "__main__":
    main()