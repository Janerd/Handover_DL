# Handover_DL — 深度学习切换策略实验

基于 Sionna RT 仿真数据，对比 A3 算法与深度学习切换策略（GRU / TCN / Transformer）。

## 项目结构

```
Handover_DL/
├── config.py           # 全局配置（路径、模型超参数、评估指标）
├── data_loader.py      # 加载 Sionna 仿真数据
├── evaluator.py        # 质量评估（SINR / 切换次数 / RLF / 乒乓率）
├── train.py            # 训练 DL 模型
├── evaluate.py         # 综合评估与对比
├── algorithms/
│   ├── a3.py           # A3 切换算法（3GPP 标准，每邻区对独立 offset）
│   ├── base.py         # DL 策略基类（滑动窗口 + 推理时延测量）
│   ├── gru.py          # GRU 模型
│   ├── tcn.py          # TCN 模型（因果膨胀卷积）
│   └── transformer.py  # 小型 Transformer（因果掩码 + CLS token）
└── outputs/
    ├── models/         # 训练好的模型权重
    └── evaluation/     # 评估结果和可视化图
```

## 数据来源

Sionna RT 仿真（慕尼黑场景，7 基站，3 种速度）：
- `C:/PC_Simu/Sionna/outputs/trajectory_data.npz`（轨迹级数据）
- `C:/PC_Simu/Sionna/outputs/dataset.npz`（样本级数据，用于 DL 训练）

## 使用流程

### 步骤 1：训练 DL 模型

```bash
cd C:\Users\haojia\Handover_DL

# 训练所有模型
python train.py --model all

# 只训练 GRU
python train.py --model gru

# 指定训练轮数
python train.py --model all --epochs 30
```

### 步骤 2：综合评估

```bash
# 评估所有算法（A3 + GRU + TCN + Transformer）
python evaluate.py

# 只评估 30 km/h 轨迹
python evaluate.py --speed 30

# 跳过 A3 优化（使用默认 offset=3dB）
python evaluate.py --no-a3-opt
```

## 算法说明

### A3 算法（基线）
- **决策依据**：RSRP（L3 滤波后）
- **触发条件**：邻区 RSRP > 服务小区 RSRP + offset + hysteresis，持续 TTT=80ms
- **优化**：在训练集上，对每个邻区对独立优化 offset，目标：最大化 SINR

### GRU / TCN / Transformer（深度学习）
- **决策依据**：过去 10 slots（400ms）的 70 维特征（RSRP/RSRQ/SINR/Doppler/K_factor 等）
- **输出**：预测未来 200ms 的最优服务小区
- **切换执行**：滑动窗口（连续 3 次预测同一邻区才执行切换，类似 TTT）

## 评估指标

| 指标 | 说明 | 避免循环论证 |
|---|---|---|
| 平均 SINR [dB] | 服务小区的 SINR 均值 | ✅ SINR 来自射线追踪，不是决策输入 |
| 5th percentile SINR | 最差 5% 时刻的 SINR | ✅ 同上 |
| 切换次数 | 每条轨迹的切换次数 | ✅ 统计量 |
| RLF 次数 | SINR < -6dB 持续 200ms | ✅ 基于 SINR |
| 乒乓率 | 1s 内来回切换的比例 | ✅ 统计量 |
| 推理时延 [μs] | DL 模型单次推理时间 | ✅ 实测 |

## 依赖

```bash
pip install torch numpy scikit-learn tqdm matplotlib