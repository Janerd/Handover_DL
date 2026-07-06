# 基于深度学习的 5G 切换决策研究

## 摘要

本文研究 5G 弱覆盖场景下的切换决策问题，基于 Sionna RT 物理层射线追踪仿真生成慕尼黑城市场景数据集，对比 A3 算法与三种深度学习模型（GRU、TCN、Transformer）的切换性能。实验结果表明，Transformer 在均值 SINR 上超过 A3 基线 0.22 dB，TCN 在切换稳定性（乒乓率 6.1%）上表现最优。

---

## 一、问题定义

### 1.1 切换决策的形式化

UE 在移动过程中，每个时隙 $t$（时长 40ms）需要决定服务小区：

$$c_t \in \{0, 1, \ldots, C-1\}, \quad C = 7$$

**目标**：在保证服务质量（SINR）的同时，减少不必要的切换次数。

**状态空间**：过去 $W = 10$ 个时隙的信道观测序列：

$$\mathbf{S}_t = [\mathbf{x}_{t-W+1}, \mathbf{x}_{t-W+2}, \ldots, \mathbf{x}_t] \in \mathbb{R}^{W \times F}$$

其中 $F = 70$（10 类特征 × 7 基站）。

### 1.2 评估指标

| 指标 | 定义 | 说明 |
|---|---|---|
| 均值 SINR | $\bar{\gamma} = \frac{1}{T}\sum_{t=1}^{T} \gamma_{t, c_t}$ | 主指标，反映平均服务质量 |
| P5 SINR | $\gamma_{5\%}$ | 最差 5% 时刻，反映覆盖鲁棒性 |
| 切换次数 | $N_{\text{HO}} = \sum_{t=2}^{T} \mathbf{1}[c_t \neq c_{t-1}]$ | 网络信令开销 |
| RLF 次数 | $\gamma_{t,c_t} < -6$ dB 持续 200ms | 无线链路失败 |
| 乒乓率 | 1s 内来回切换比例 | 切换稳定性 |
| 推理时延 | 单次前向传播时间 [μs] | UE 侧实时性 |

**注**：SINR 仅用于评估，不参与训练，避免循环论证。

---

## 二、数据集

### 2.1 仿真环境

- **仿真工具**：Sionna RT v2.0.1（基于 Mitsuba 3 的物理层射线追踪仿真器）
- **场景**：慕尼黑市中心 1km × 1km，真实 3D 建筑物几何（可行走比例 56.8%）
- **基站**：$C = 7$，ISD ≈ 307m，载波频率 $f_c = 3.5$ GHz，基站高度 $h_{\text{BS}} = 10$ m
- **UE**：120 条轨迹，速度 $v \in \{30, 60, 120\}$ km/h，时隙时长 40ms
- **天线**：SISO，全向天线（iso 方向图）

### 2.2 信道特征

每个时隙提取 10 类特征，每类 7 个基站，共 70 维：

| 索引 | 特征名 | 物理含义 | 归一化方式 |
|---|---|---|---|
| [0:7] | RSRP_l3 | L3 滤波参考信号接收功率 [dBm] | min-max，$[-140, -40]$ dB |
| [7:14] | RSRQ | 参考信号接收质量 [dB] | min-max，$[-30, 0]$ dB |
| [14:21] | SINR | 信干噪比 [dB] | min-max，$[-20, 40]$ dB |
| [21:28] | Doppler | 多普勒频移 [Hz] | 对称归一化，$\pm 500$ Hz |
| [28:35] | BeamID | 最强路径波束 ID | 正值归一化，$/6$ |
| [35:42] | RSRP_diff | RSRP 变化率 [dB/slot] | 对称归一化，$\pm 5$ dB |
| [42:49] | BeamID_diff | 波束 ID 变化 | 对称归一化，$\pm 6$ |
| [49:56] | DelaySpread | RMS 时延扩展 [s] | 正值归一化，$/10^{-6}$ |
| [56:63] | K_factor | Ricean K 因子 | 正值归一化，$/30$ |
| [63:70] | min_tau | 最短路径时延 [s] | 正值归一化，$/2\times10^{-6}$ |

**L3 滤波**（指数加权平均）：

$$\text{RSRP}_{\text{l3}}(t) = (1-\alpha) \cdot \text{RSRP}_{\text{l3}}(t-1) + \alpha \cdot \text{RSRP}_{\text{raw}}(t), \quad \alpha = 1/16$$

### 2.3 标签定义

**切换收益标签**：

$$Y_t = \arg\max_{k \in \{0,\ldots,C-1\}} \left[ \frac{1}{H}\sum_{i=1}^{H} \gamma_{t+i,k} - \frac{1}{H}\sum_{i=1}^{H} \gamma_{t+i,c_t} - \delta \right]$$

其中 $H = 10$（400ms），$\delta = 3$ dB 为切换代价。若最大收益 $\leq 0$，标签为保持当前小区 $c_t$。

**设计意义**：
1. 使用**未来** SINR 定义标签，**当前** SINR 作为输入特征，时间上分离，避免循环论证
2. 切换代价 $\delta$ 过滤短暂波动，只有持续更好的小区才被标注为切换目标
3. 标签包含"保持当前"选项，模型学习"何时不切换"

### 2.4 数据集划分

| 划分 | 轨迹数 | 样本数 |
|---|---|---|
| 训练集 | 72 条（60%） | 153,576 |
| 验证集 | 24 条（20%） | 51,192 |
| 测试集 | 24 条（20%） | 51,192 |

---

## 三、A3 算法

### 3.1 标准 A3 事件（3GPP TS 36.331）

A3 事件触发条件：

$$M_n(t) - M_s(t) > \text{offset}_{(s,n)} + \text{Hys}$$

其中：
- $M_s(t)$：服务小区 $s$ 的 RSRP 测量值（L3 滤波后）
- $M_n(t)$：邻区 $n$ 的 RSRP 测量值
- $\text{offset}_{(s,n)}$：邻区对 $(s,n)$ 的切换偏置（Cell Individual Offset，CIO）
- $\text{Hys} = 2$ dB：迟滞量，防止边界附近频繁触发

**TTT 机制**：条件必须持续满足 TTT = 80ms（2 个时隙）才触发切换命令。若中途不满足，计时器重置。

**切换执行**：收到切换命令后，UE 接入目标小区，设置冷却时间 200ms（5 个时隙）防止立即再次切换。

### 3.2 每邻区对独立优化

**动机**：不同邻区对的路径损耗差异、建筑物遮挡程度不同，统一的 offset 无法适应所有情况。

**优化目标**：

$$\text{offset}^*_{(s,n)} = \arg\max_{\delta \in \mathcal{D}} \mathcal{J}(\delta; \mathcal{T}_{\text{train}})$$

$$\mathcal{J} = \bar{\gamma} - \lambda_1 \cdot \bar{N}_{\text{HO}} - \lambda_2 \cdot N_{\text{RLF}}$$

其中 $\lambda_1 = 0.1$，$\lambda_2 = 5.0$，搜索集合 $\mathcal{D} = \{-3.0, -2.5, \ldots, 9.0\}$ dB（步长 0.5 dB，共 25 个候选值）。

**优化算法**（坐标下降）：

```
初始化：所有邻区对 offset = 3.0 dB（默认值）
对每个邻区对 (s, n)：
    for δ in [-3.0, -2.5, ..., 9.0]:
        临时设置 offset(s,n) = δ
        在训练集上运行 A3，计算 J(δ)
    offset(s,n) ← argmax J(δ)
```

本实验共 42 个邻区对，每对搜索 25 个候选值，总计 1050 次评估。

---

## 四、深度学习模型

### 4.1 训练设置

| 超参数 | 值 |
|---|---|
| 批大小 | 512 |
| 最大轮数 | 50 |
| 优化器 | AdamW，$\text{lr}=10^{-3}$，$\text{wd}=10^{-4}$ |
| 学习率调度 | 余弦退火，$\text{lr}_{\min} = 10^{-5}$ |
| 梯度裁剪 | max_norm = 1.0 |
| Early Stopping | patience = 10 |
| 损失函数 | 加权交叉熵 |

**类别权重**（处理标签不均衡）：

$$w_c = \frac{N}{C \cdot N_c}, \quad \text{归一化使} \bar{w} = 1$$

### 4.2 GRU 模型

**结构**：

$$\mathbf{h}_t = \text{GRU}(\mathbf{x}_t, \mathbf{h}_{t-1}), \quad \hat{y} = \text{FC}(\text{Dropout}(\mathbf{h}_W))$$

**GRU 单元**：

$$z_t = \sigma(W_z [\mathbf{h}_{t-1}; \mathbf{x}_t] + b_z) \quad \text{（更新门）}$$

$$r_t = \sigma(W_r [\mathbf{h}_{t-1}; \mathbf{x}_t] + b_r) \quad \text{（重置门）}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h [r_t \odot \mathbf{h}_{t-1}; \mathbf{x}_t] + b_h)$$

$$\mathbf{h}_t = (1 - z_t) \odot \mathbf{h}_{t-1} + z_t \odot \tilde{\mathbf{h}}_t$$

**超参数**：隐状态维度 128，2 层，Dropout 0.3，单向。

**参数量**：约 100K。

### 4.3 TCN 模型

**结构**：

$$\mathbf{X}' = \text{Linear}_{70 \to 64}(\mathbf{X}) \in \mathbb{R}^{B \times 10 \times 64}$$

$$\mathbf{Z} = \text{TCNBlock}_{d=4}(\text{TCNBlock}_{d=2}(\text{TCNBlock}_{d=1}(\mathbf{X}'^T)))$$

$$\hat{y} = \text{FC}(\text{Dropout}(\mathbf{Z}_{:,:,-1}))$$

**因果膨胀卷积**（kernel=3）：

$$(\mathbf{x} *_d \mathbf{w})[t] = \sum_{k=0}^{K-1} \mathbf{w}[k] \cdot \mathbf{x}[t - d \cdot k]$$

其中 $d$ 为膨胀因子，卷积只访问 $t$ 及之前的时刻（因果性）。

**感受野**：三层叠加后，$t=10$ 时刻可覆盖全部 10 个历史时隙：

$$\text{RF} = 1 + \sum_{i=0}^{L-1} (K-1) \cdot 2^i = 1 + 2 \cdot (1 + 2 + 4) = 15 > 10 \checkmark$$

**TCN 残差块**：

$$\text{TCNBlock}(x) = \text{ReLU}(\text{BN}(\text{CausalConv}_2(\text{Dropout}(\text{ReLU}(\text{BN}(\text{CausalConv}_1(x)))))) + x)$$

**超参数**：通道数 $[64, 128, 128]$，kernel=3，Dropout 0.2。

**参数量**：约 200K。

### 4.4 Transformer 模型

**结构**：

$$\mathbf{X}' = \text{Linear}_{70 \to 64}(\mathbf{X}) \in \mathbb{R}^{B \times 10 \times 64}$$

$$\mathbf{Z} = \text{TransformerEncoder}([\mathbf{e}_{\text{cls}}; \mathbf{X}'] + \text{PE}) \in \mathbb{R}^{B \times 11 \times 64}$$

$$\hat{y} = \text{FC}(\text{LayerNorm}(\text{Dropout}(\mathbf{Z}_{:,0,:})))$$

**Multi-Head Self-Attention**（4 头，$d_k = 16$）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中 $M$ 为因果掩码（上三角为 $-\infty$，CLS token 行全为 0）。

**因果掩码**（$T=10$，含 CLS 共 11 个位置）：

$$M_{ij} = \begin{cases} 0 & i = 0 \text{（CLS 行）} \\ 0 & j \leq i \\ -\infty & j > i \end{cases}$$

**Pre-LN Transformer 层**（训练更稳定）：

$$\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LN}(\mathbf{x}))$$

$$\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$$

**超参数**：$d_{\text{model}}=64$，4 头，2 层，FFN=128，Dropout 0.1。

**参数量**：约 50K（最小）。

---

## 五、切换执行策略

### 5.1 滑动窗口机制

模型每时隙输出预测目标小区 $\hat{c}_t$，通过滑动窗口过滤：

```
初始化：计数器 counter = {}，冷却 cooldown = 0

for t = 1, 2, ..., T:
    if cooldown > 0:
        c_t = c_{t-1}，cooldown -= 1
        continue
    
    if ĉ_t ≠ c_{t-1}:
        counter[ĉ_t] += 1
    else:
        counter = {}
    
    if counter[ĉ_t] >= N and ĉ_t ≠ c_{t-1}:
        c_t = ĉ_t（执行切换）
        counter = {}，cooldown = 5
    else:
        c_t = c_{t-1}（保持当前）
```

**参数**：$N = 3$（连续 3 次，即 120ms），冷却时间 5 个时隙（200ms）。

**与 A3 TTT 的对比**：

| | A3 TTT | DL 滑动窗口 |
|---|---|---|
| 触发条件 | RSRP 差值 > offset | 模型预测连续 N 次一致 |
| 时间过滤 | 80ms（固定） | $N \times 40$ms（可调） |
| 信息来源 | 当前 RSRP | 历史 10 slots 全部特征 |

---

## 六、实验结果

### 6.1 总体对比

| 算法 | 均值 SINR [dB] | P5 SINR [dB] | 切换次数 | RLF | 乒乓率 | 推理时延 |
|---|---|---|---|---|---|---|
| A3 (optimized) | +9.14 | -26.08 | **2.8** | **151** | **0.6%** | N/A |
| GRU | +9.06 | -26.08 | 8.0 | 162 | 16.7% | 572 μs |
| TCN | +9.23 | -26.08 | 6.6 | 160 | **6.1%** | 2284 μs |
| **Transformer** | **+9.36** | -26.08 | 6.7 | 153 | 12.3% | **575 μs** |

### 6.2 按速度分组

| 算法 | 30 km/h | 60 km/h | 120 km/h |
|---|---|---|---|
| A3 | +9.03 dB | +17.62 dB | -7.40 dB |
| GRU | +8.90 dB | +17.95 dB | -8.03 dB |
| TCN | +9.07 dB | +17.92 dB | -7.51 dB |
| Transformer | +9.22 dB | +17.99 dB | -7.37 dB |

### 6.3 RSRP-SINR 相关性分析

为验证实验框架的合理性，分析 RSRP 最强小区与 SINR 最高小区的一致率：

- **一致率**：72.8%（即 27.2% 的时隙中，RSRP 最强 ≠ SINR 最高）
- **RSRP-SINR 相关系数**：$r = 0.897$（强相关）
- **RSRP 决策导致 SINR 损失的时隙比例**：3.4%
- **不一致时的平均 SINR 损失**：12.3 dB

**结论**：在 7 基站低干扰场景下，RSRP 和 SINR 高度相关，A3 的 RSRP 决策在 96.6% 的时隙是正确的，这解释了 A3 在切换次数和稳定性上的优势。DL 方法的理论提升空间约为 $3.4\% \times 12.3 \approx 0.4$ dB，与实验结果（Transformer 超过 A3 约 0.22 dB）基本吻合。

### 6.4 结论

1. **Transformer 在 SINR 质量上最优**（+9.36 dB），超过 A3 基线 0.22 dB，推理时延最短（575 μs）
2. **TCN 在切换稳定性上最优**（乒乓率 6.1%），接近 A3（0.6%）
3. **A3 在切换效率上最优**（切换次数 2.8，乒乓率 0.6%），无需训练和推理资源
4. **DL 方法的代价**：切换次数（6-8 次）是 A3（2.8 次）的 2-3 倍，乒乓率（6-17%）远高于 A3（0.6%）
5. **场景局限性**：在少基站低干扰场景下，RSRP-SINR 高度相关，A3 已接近最优。DL 方法的优势在高干扰、多基站、RSRP-SINR 解耦的复杂场景中更为显著

---

## 参考文献

1. 3GPP TS 36.331, "Evolved Universal Terrestrial Radio Access (E-UTRA); Radio Resource Control (RRC)," 3rd Generation Partnership Project, 2023.
2. Hoydis, J., et al., "Sionna: An Open-Source Library for Next-Generation Physical Layer Research," arXiv:2203.11854, 2022.
3. Bai, S., Kolter, J. Z., and Koltun, V., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018.
4. Vaswani, A., et al., "Attention Is All You Need," NeurIPS, 2017.
5. Cho, K., et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," EMNLP, 2014.