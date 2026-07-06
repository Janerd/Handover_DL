# Handover Decision with Deep Learning

基于 Sionna RT 物理层仿真数据，对比 A3 算法与深度学习切换策略（GRU / TCN / Transformer）在 5G 弱覆盖场景下的性能。

## 项目结构

```
Handover_DL/
├── run.py              # 一键运行入口
├── setup_check.py      # 环境检测
├── train.py            # 模型训练
├── evaluate.py         # 综合评估
├── config.py           # 全局配置
├── data_loader.py      # 数据加载
├── evaluator.py        # 评估指标
├── algorithms/
│   ├── a3.py           # A3 切换算法（3GPP 标准）
│   ├── gru.py          # GRU 模型
│   ├── tcn.py          # TCN 模型
│   └── transformer.py  # Transformer 模型
├── data/               # 数据文件（需手动放入）
│   ├── trajectory_data.npz
│   └── dataset.npz
└── outputs/
    └── models/         # 预训练模型权重
        ├── GRU_best.pt
        ├── TCN_best.pt
        └── Transformer_best.pt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**已安装 PyTorch 的情况：**

- 已安装 GPU 版（`torch+cu*`）：无需重新安装，代码会自动检测并使用 GPU
- 已安装 CPU 版（`torch+cpu`）：无需重新安装，直接运行即可
- 版本要求：`torch >= 2.0.0`，低于此版本请升级

**未安装 PyTorch 的情况（CPU 版，推荐）：**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn tqdm matplotlib packaging
```

**测试环境**：Python 3.12.10，PyTorch 2.12.0+cpu，Windows 11

### 2. 检测环境

```bash
python setup_check.py
```

### 3. 运行评估

```bash
python run.py
```

评估结果保存在 `outputs/evaluation/`：
- `results.json`：数值结果
- `comparison.png`：6 指标对比图
- `sinr_cdf.png`：SINR CDF 曲线

### 4. 可选：重新训练

```bash
python run.py --train
```

## 算法说明

### A3（基线）
3GPP 标准切换算法，基于 RSRP 触发。对每个邻区对独立优化偏置参数，目标为最大化 SINR。

### GRU / TCN / Transformer
基于过去 10 个时隙（400ms）的信道特征序列，预测切换收益最大的目标小区。

**输入特征**（10 类 × 7 基站 = 70 维）：
RSRP、RSRQ、SINR、Doppler、BeamID、RSRP 变化率、BeamID 变化、时延扩展、K 因子、最短路径时延

**标签定义**：未来 400ms 内，切换收益（目标小区 SINR - 当前小区 SINR - 切换代价）最大的动作。

**切换执行**：滑动窗口（连续 3 次预测同一邻区才执行切换，类似 TTT 机制）。

## 评估指标

| 指标 | 说明 |
|---|---|
| 均值 SINR [dB] | 服务小区 SINR 均值（主指标） |
| P5 SINR [dB] | 最差 5% 时刻的 SINR |
| 切换次数 | 每条轨迹的平均切换次数 |
| RLF 次数 | 无线链路失败次数（SINR < -6 dB 持续 200ms） |
| 乒乓率 | 1s 内来回切换的比例 |
| 推理时延 [μs] | 深度学习模型的单次推理时间 |

## 仿真环境

- 场景：慕尼黑市中心（1km × 1km）
- 基站：7 个，ISD ≈ 307m，载波频率 3.5 GHz
- UE 速度：30 / 60 / 120 km/h
- 信道：Sionna RT 射线追踪（真实 3D 建筑物几何）