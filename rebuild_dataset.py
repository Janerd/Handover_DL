"""
rebuild_dataset.py
==================
从已有的 trajectory_data.npz 重建数据集。

支持两种标签定义：

模式 1（默认）：切换收益最大的动作（推荐，学术上干净）
    Y = argmax(SINR_future_avg[k] - SINR_future_avg[current] - ho_cost)
    即：切换到哪个小区的净收益最大（如果最大收益 <= 0，则保持当前）

    优势：
    - SINR 在时间上分离（输入=过去，标签=未来），无循环论证
    - 标签直接反映"是否切换"和"切换到哪里"
    - ho_cost 参数控制切换激进程度（类似 A3 的 offset）

模式 2（--mode avg）：区间平均 SINR 最优小区
    Y = argmax(mean(SINR[t : t + horizon]))
    即：未来 horizon slots 内，平均 SINR 最高的小区

使用方法：
    python rebuild_dataset.py                          # 切换收益标签，ho_cost=1dB
    python rebuild_dataset.py --ho-cost 3.0            # 更保守（类似 A3 offset=3dB）
    python rebuild_dataset.py --mode avg --horizon 10  # 区间平均 SINR 标签
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import TRAJECTORY_DATA_PATH, SIONNA_OUTPUT_DIR, DATA_CFG
from data_loader import TrajectoryData


# =========================================================
# 模式 1：切换收益标签
# =========================================================

def build_dataset_switch_benefit(
    trajectory_data_path: Path,
    window_size: int = 10,
    future_horizon: int = 10,
    ho_cost_db: float = 1.0,
) -> dict:
    """
    从轨迹数据重建数据集，使用切换收益最大的动作作为标签。

    标签定义：
        对每个时隙 t，计算切换到每个小区的净收益：
            benefit[k] = mean(SINR[t+1:t+H, k]) - mean(SINR[t+1:t+H, current]) - ho_cost
        标签 = argmax(benefit)，如果 max(benefit) <= 0 则保持当前小区

    参数：
        trajectory_data_path: trajectory_data.npz 路径
        window_size:          特征窗口大小（历史时隙数）
        future_horizon:       未来 SINR 平均窗口长度（时隙数）
        ho_cost_db:           切换代价 [dB]（控制切换激进程度）

    返回：
        包含 X_raw, Y_cell, Y_sinr, split_masks 的字典
    """
    print(f"加载轨迹数据：{trajectory_data_path}")
    raw = np.load(trajectory_data_path, allow_pickle=True)
    num_traj = len(raw["traj_ids"])
    print(f"  总轨迹数：{num_traj}")
    print(f"  标签定义：切换收益最大的动作（future_horizon={future_horizon} slots，ho_cost={ho_cost_db} dB）")

    W = window_size
    H = future_horizon
    C = DATA_CFG.num_cells

    # 预估样本总数
    total_samples = 0
    for i in range(num_traj):
        T = int(raw["num_slots"][i])
        total_samples += max(0, T - W - H)

    print(f"  预估样本总数：{total_samples}")

    X_raw = np.zeros((total_samples, W, DATA_CFG.num_features), dtype=np.float32)
    Y_cell = np.zeros(total_samples, dtype=np.int64)
    Y_sinr = np.zeros((total_samples, C), dtype=np.float32)
    meta_speed = np.zeros(total_samples, dtype=np.float32)
    meta_traj = np.zeros(total_samples, dtype=np.int32)
    meta_split = np.empty(total_samples, dtype=object)

    stay_count = 0
    switch_count = 0
    sample_idx = 0

    for i in range(num_traj):
        traj_split = str(raw["splits"][i])
        traj_speed = float(raw["speed_kmh"][i])
        T = int(raw["num_slots"][i])

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

        feat = traj.get_feature_matrix()   # [T, F]，已归一化
        sinr_seq = traj.sinr               # [T, C]，原始 SINR [dB]
        serving_l3 = traj.serving_l3       # [T]，当前服务小区

        for slot in range(W, T - H):
            X_raw[sample_idx] = feat[slot - W : slot]

            current_cell = int(serving_l3[slot])

            # 未来 H slots 的平均 SINR [C]
            future_sinr_avg = np.mean(sinr_seq[slot + 1 : slot + 1 + H], axis=0)

            # 切换收益：目标小区未来 SINR - 当前小区未来 SINR - 切换代价
            current_future_sinr = future_sinr_avg[current_cell]
            switch_benefit = future_sinr_avg - current_future_sinr - ho_cost_db
            switch_benefit[current_cell] = 0.0  # 保持当前的收益定义为 0

            best_cell = int(np.argmax(switch_benefit))
            if switch_benefit[best_cell] <= 0:
                Y_cell[sample_idx] = current_cell
                stay_count += 1
            else:
                Y_cell[sample_idx] = best_cell
                switch_count += 1

            Y_sinr[sample_idx] = future_sinr_avg
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
    print(f"  标签分布：保持当前={stay_count}（{stay_count/actual_samples*100:.1f}%），"
          f"切换={switch_count}（{switch_count/actual_samples*100:.1f}%）")

    train_mask = meta_split == "train"
    val_mask = meta_split == "val"
    test_mask = meta_split == "test"

    print(f"  训练集：{train_mask.sum()} 样本")
    print(f"  验证集：{val_mask.sum()} 样本")
    print(f"  测试集：{test_mask.sum()} 样本")

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
        "future_horizon": future_horizon,
        "ho_cost_db": ho_cost_db,
        "window_size": window_size,
        "label_mode": "switch_benefit",
    }


# =========================================================
# 模式 2：区间平均 SINR 标签
# =========================================================

def build_dataset_avg_label(
    trajectory_data_path: Path,
    window_size: int = 10,
    pred_horizon: int = 1,
    avg_horizon: int = 10,
) -> dict:
    """
    从轨迹数据重建数据集，使用区间平均 SINR 最优小区作为标签（模式 2）。
    """
    print(f"加载轨迹数据：{trajectory_data_path}")
    raw = np.load(trajectory_data_path, allow_pickle=True)
    num_traj = len(raw["traj_ids"])
    print(f"  总轨迹数：{num_traj}")
    print(f"  标签定义：未来 [{pred_horizon}, {pred_horizon + avg_horizon}) slots 的平均 SINR 最优小区")

    W = window_size
    C = DATA_CFG.num_cells

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

        feat = traj.get_feature_matrix()
        sinr_seq = traj.sinr

        for slot in range(W, T - pred_horizon - avg_horizon + 1):
            X_raw[sample_idx] = feat[slot - W : slot]

            future_start = slot + pred_horizon
            future_end = slot + pred_horizon + avg_horizon
            avg_sinr = np.mean(sinr_seq[future_start:future_end], axis=0)

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

    train_mask = meta_split == "train"
    val_mask = meta_split == "val"
    test_mask = meta_split == "test"

    print(f"  训练集：{train_mask.sum()} 样本")
    print(f"  验证集：{val_mask.sum()} 样本")
    print(f"  测试集：{test_mask.sum()} 样本")

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
        "label_mode": "avg",
    }


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="重建数据集（切换收益标签 or 区间平均 SINR 标签）")
    parser.add_argument(
        "--mode", choices=["benefit", "avg"], default="benefit",
        help="标签模式：benefit=切换收益（默认），avg=区间平均 SINR"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="未来时间窗口长度（时隙数，默认 10 = 400ms）"
    )
    parser.add_argument(
        "--ho-cost", type=float, default=1.0,
        help="切换代价 [dB]（仅 benefit 模式，默认 1.0）"
    )
    parser.add_argument(
        "--pred-offset", type=int, default=1,
        help="预测起始偏移（仅 avg 模式，默认 1）"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="输出目录（默认：Sionna outputs 目录）"
    )
    args = parser.parse_args()

    if args.outdir is None:
        out_dir = SIONNA_OUTPUT_DIR
    else:
        out_dir = Path(args.outdir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "benefit":
        save_path = out_dir / f"dataset_benefit_h{args.horizon}_c{args.ho_cost:.1f}.npz"
        print(f"\n重建数据集（切换收益标签，horizon={args.horizon} slots，ho_cost={args.ho_cost} dB）")
        print(f"输出路径：{save_path}")

        result = build_dataset_switch_benefit(
            trajectory_data_path=TRAJECTORY_DATA_PATH,
            window_size=DATA_CFG.window_size,
            future_horizon=args.horizon,
            ho_cost_db=args.ho_cost,
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
            future_horizon=np.array(result["future_horizon"]),
            ho_cost_db=np.array(result["ho_cost_db"]),
            label_mode=np.array(result["label_mode"]),
        )

    else:  # avg
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