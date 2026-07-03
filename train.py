"""
train.py
========
训练深度学习切换策略模型（GRU / TCN / Transformer）

使用方法：
    python train.py --model gru
    python train.py --model tcn
    python train.py --model transformer
    python train.py --model all   # 训练所有模型
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    GRUConfig, TCNConfig, TransformerConfig,
    GRU_CFG, TCN_CFG, TRANSFORMER_CFG, OUTPUT_DIR,
)
from data_loader import load_dataset_splits, compute_class_weights, normalize_X, DATASET_PATH
from algorithms.gru import GRUHandoverPolicy
from algorithms.tcn import TCNHandoverPolicy
from algorithms.transformer import TransformerHandoverPolicy


# =========================================================
# 训练函数
# =========================================================

def train_model(
    policy,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> dict:
    """
    训练单个 DL 模型

    参数：
        policy:     DL 切换策略对象（已调用 build_model()）
        X_train:    [N, W, F] 训练特征
        Y_train:    [N] 训练标签
        X_val:      [N, W, F] 验证特征
        Y_val:      [N] 验证标签
        model_name: 模型名称（用于保存）
        save_dir:   模型保存目录

    返回：
        训练历史字典
    """
    cfg = policy.cfg
    device = policy.device

    print(f"\n{'='*60}")
    print(f"训练模型：{model_name}")
    print(f"{'='*60}")
    print(f"  设备：{device}")
    policy.count_parameters()

    # ---- 数据集 ----
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).long()

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=0,
    )

    # ---- 损失函数（加权，处理标签不均衡）----
    if cfg.use_class_weights:
        class_weights = compute_class_weights(Y_train, cfg.num_classes)
        weight_tensor = torch.from_numpy(class_weights).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # 学习率调度：余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01,
    )

    # ---- 训练循环 ----
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    save_path = save_dir / f"{model_name}_best.pt"
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- 训练阶段 ----
        policy.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            logits = policy.model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(policy.model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * len(Y_batch)
            train_correct += (logits.argmax(1) == Y_batch).sum().item()
            train_total += len(Y_batch)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- 验证阶段 ----
        policy.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                logits = policy.model(X_batch)
                loss = criterion(logits, Y_batch)
                val_loss += loss.item() * len(Y_batch)
                val_correct += (logits.argmax(1) == Y_batch).sum().item()
                val_total += len(Y_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ---- 打印进度 ----
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{cfg.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.0f}s")

        # ---- Early Stopping ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            policy.save(str(save_path))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch} (best: epoch {best_epoch}, val_acc={best_val_acc*100:.1f}%)")
                break

    total_time = time.time() - start_time
    print(f"\n  训练完成！最佳验证准确率：{best_val_acc*100:.1f}%（epoch {best_epoch}）")
    print(f"  总训练时间：{total_time:.0f}s")
    print(f"  模型已保存到：{save_path}")

    # 加载最佳模型
    policy.load(str(save_path))

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["total_time_s"] = total_time

    return history


def evaluate_classification(
    policy,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    在测试集上评估分类性能（Accuracy + Macro-F1）

    注意：这是分类任务的评估，不是切换策略的评估。
    切换策略评估（SINR/RLF/HO次数）在 evaluate.py 中进行。
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    policy.model.eval()
    device = policy.device

    X_tensor = torch.from_numpy(X_test).float()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            logits = policy.model(X_batch)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)

    Y_pred = np.concatenate(all_preds)

    acc = accuracy_score(Y_test, Y_pred)
    macro_f1 = f1_score(Y_test, Y_pred, average="macro")
    weighted_f1 = f1_score(Y_test, Y_pred, average="weighted")

    print(f"\n{model_name} 分类性能（测试集）：")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Macro-F1:    {macro_f1*100:.2f}%")
    print(f"  Weighted-F1: {weighted_f1*100:.2f}%")
    print(f"\n  详细报告：")
    print(classification_report(Y_test, Y_pred,
                                 target_names=[f"cell_{i}" for i in range(policy.cfg.num_classes)]))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="训练 DL 切换策略模型")
    parser.add_argument(
        "--model", choices=["gru", "tcn", "transformer", "all"],
        default="all", help="要训练的模型（默认：all）"
    )
    parser.add_argument(
        "--outdir", default=str(OUTPUT_DIR / "models"),
        help="模型保存目录"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数（覆盖配置文件）"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="数据集路径（默认使用 config 中的 DATASET_PATH）"
    )
    args = parser.parse_args()

    save_dir = Path(args.outdir)

    # ---- 加载数据 ----
    dataset_path = Path(args.dataset) if args.dataset else None
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset_splits(dataset_path)

    # ---- 判断是否需要归一化 ----
    # rebuild_dataset.py 生成的数据集已经归一化（特征来自 get_feature_matrix()）
    # 原始 dataset.npz 未归一化，需要手动归一化
    num_cells = X_train.shape[2] // 10  # F = 10 * C
    needs_norm = X_train[..., :num_cells].min() < -10.0  # 原始 RSRP 约 -155 dBm

    if needs_norm:
        print("\n归一化特征（原始 dataset.npz）...")
        X_train = normalize_X(X_train, num_cells)
        X_val = normalize_X(X_val, num_cells)
        X_test = normalize_X(X_test, num_cells)
        print(f"  归一化完成：RSRP 范围 [{X_train[..., :num_cells].min():.3f}, {X_train[..., :num_cells].max():.3f}]")
    else:
        print(f"\n特征已归一化（来自 rebuild_dataset.py），跳过归一化步骤")
        print(f"  RSRP 范围 [{X_train[..., :num_cells].min():.3f}, {X_train[..., :num_cells].max():.3f}]")

    # ---- 配置模型 ----
    models_to_train = []

    if args.model in ("gru", "all"):
        cfg = GRUConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        policy = GRUHandoverPolicy(cfg=cfg)
        policy.build_model()
        models_to_train.append(("GRU", policy))

    if args.model in ("tcn", "all"):
        cfg = TCNConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        policy = TCNHandoverPolicy(cfg=cfg)
        policy.build_model()
        models_to_train.append(("TCN", policy))

    if args.model in ("transformer", "all"):
        cfg = TransformerConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        policy = TransformerHandoverPolicy(cfg=cfg)
        policy.build_model()
        models_to_train.append(("Transformer", policy))

    # ---- 训练 ----
    all_histories = {}
    for model_name, policy in models_to_train:
        history = train_model(
            policy, X_train, Y_train, X_val, Y_val,
            model_name=model_name, save_dir=save_dir,
        )
        cls_metrics = evaluate_classification(policy, X_test, Y_test, model_name)
        history.update(cls_metrics)
        all_histories[model_name] = history

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("训练汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'最佳Val Acc':>12} {'Test Acc':>10} {'Macro-F1':>10} {'训练时间':>10}")
    print("-" * 60)
    for name, h in all_histories.items():
        print(f"{name:<15} {h['best_val_acc']*100:>11.1f}% "
              f"{h['accuracy']*100:>9.1f}% "
              f"{h['macro_f1']*100:>9.1f}% "
              f"{h['total_time_s']:>8.0f}s")

    # 保存训练历史
    import json
    history_path = save_dir / "training_history.json"
    # 转换 numpy 类型为 Python 原生类型
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {
        k: {kk: convert(vv) if not isinstance(vv, list) else [convert(x) for x in vv]
            for kk, vv in v.items()}
        for k, v in all_histories.items()
    }
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n训练历史已保存到：{history_path}")


if __name__ == "__main__":
    main()