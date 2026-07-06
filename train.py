"""
train.py — 训练深度学习切换策略模型

使用方法：
    python train.py              # 训练所有模型（GRU / TCN / Transformer）
    python train.py --model gru  # 只训练 GRU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import GRUConfig, TCNConfig, TransformerConfig, MODEL_DIR
from data_loader import load_dataset, normalize_X, compute_class_weights
from algorithms.gru import GRUHandoverPolicy
from algorithms.tcn import TCNHandoverPolicy
from algorithms.transformer import TransformerHandoverPolicy


def train_one(policy, X_train, Y_train, X_val, Y_val, name: str, save_dir: Path) -> dict:
    """训练单个模型，返回训练历史。"""
    cfg = policy.cfg
    device = policy.device

    print(f"\n{'='*55}\n训练：{name}  |  设备：{device}  |  参数量：{policy.count_parameters():,}\n{'='*55}")

    # 数据集
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long()),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long()),
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0,
    )

    # 损失函数（加权处理标签不均衡）
    if cfg.use_class_weights:
        w = torch.from_numpy(compute_class_weights(Y_train, cfg.num_classes)).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(policy.model.parameters(),
                                   lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01)

    save_path = save_dir / f"{name}_best.pt"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        # 训练
        policy.model.train()
        tl, tc, tt = 0.0, 0, 0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            logits = policy.model(X_b)
            loss = criterion(logits, Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(Y_b)
            tc += (logits.argmax(1) == Y_b).sum().item()
            tt += len(Y_b)
        scheduler.step()

        # 验证
        policy.model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                logits = policy.model(X_b)
                vl += criterion(logits, Y_b).item() * len(Y_b)
                vc += (logits.argmax(1) == Y_b).sum().item()
                vt += len(Y_b)

        train_acc = tc / tt
        val_acc = vc / vt
        history["train_loss"].append(tl / tt)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(vl / vt)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.num_epochs} | "
                  f"Train {train_acc*100:.1f}% | Val {val_acc*100:.1f}% | "
                  f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            policy.save(str(save_path))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping（最佳：epoch {best_epoch}，val_acc={best_val_acc*100:.1f}%）")
                break

    print(f"  完成：最佳 val_acc={best_val_acc*100:.1f}%，耗时 {time.time()-t0:.0f}s")
    policy.load(str(save_path))
    history.update({"best_val_acc": best_val_acc, "best_epoch": best_epoch,
                    "total_time_s": time.time() - t0})
    return history


def main():
    parser = argparse.ArgumentParser(description="训练切换策略模型")
    parser.add_argument("--model", choices=["gru", "tcn", "transformer", "all"],
                        default="all")
    parser.add_argument("--outdir", default=str(MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    save_dir = Path(args.outdir)

    # 加载并归一化数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()
    num_cells = X_train.shape[2] // 10
    # 检测是否需要归一化（原始物理量范围 < -10）
    if X_train[..., :num_cells].min() < -10.0:
        print("归一化特征...")
        X_train = normalize_X(X_train, num_cells)
        X_val = normalize_X(X_val, num_cells)
        X_test = normalize_X(X_test, num_cells)

    models_to_train = []
    if args.model in ("gru", "all"):
        cfg = GRUConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        p = GRUHandoverPolicy(cfg=cfg)
        p.build_model()
        models_to_train.append(("GRU", p))
    if args.model in ("tcn", "all"):
        cfg = TCNConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        p = TCNHandoverPolicy(cfg=cfg)
        p.build_model()
        models_to_train.append(("TCN", p))
    if args.model in ("transformer", "all"):
        cfg = TransformerConfig()
        if args.epochs:
            cfg.num_epochs = args.epochs
        p = TransformerHandoverPolicy(cfg=cfg)
        p.build_model()
        models_to_train.append(("Transformer", p))

    all_history = {}
    for name, policy in models_to_train:
        h = train_one(policy, X_train, Y_train, X_val, Y_val, name, save_dir)
        all_history[name] = h

    # 保存训练历史
    def _cvt(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, list):
            return [_cvt(x) for x in v]
        return v

    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: {kk: _cvt(vv) for kk, vv in v.items()}
                   for k, v in all_history.items()}, f, indent=2)

    print(f"\n训练历史已保存：{history_path}")


if __name__ == "__main__":
    main()