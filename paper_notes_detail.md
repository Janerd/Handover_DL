# 算法原理详细说明

---

## 一、A3 算法：最优门限寻找

### 1.0 整体思路

A3 算法是 3GPP 标准定义的切换触发机制，其核心思想是**基于当前时刻的 RSRP 差值判断是否切换**。当邻区的信号强度持续超过服务小区一定门限时，触发切换命令。"持续"由 TTT（Time-To-Trigger）保证——条件必须连续满足一段时间才执行切换，从而过滤掉短暂的信号波动，避免乒乓切换。

A3 的决策逻辑极为简单：**只看当前时刻的 RSRP，不看历史，不预测未来**。这是它的优势（计算开销极低、无需训练），也是它的局限（无法区分遮挡导致的短暂信号下降和真正的覆盖边界）。

本文对 A3 的改进在于：将每个邻区对的切换偏置（offset）从统一值改为**独立优化**。不同邻区对之间的路径损耗差异、建筑物遮挡程度不同，统一的 offset 无法适应所有情况。通过在训练集上对每个邻区对独立搜索最优 offset，使 A3 在当前场景下达到最佳性能，作为深度学习方法的强基线。

由于 A3 的切换决策是阶跃函数（不可微），无法使用梯度方法，因此采用**坐标下降 + 网格搜索**：每次固定其他邻区对的 offset，对当前邻区对在离散搜索集合上穷举，选择使综合得分最高的值。

### 1.1 问题建模

设网络中有 $C$ 个基站，邻区关系集合为 $\mathcal{E} = \{(s, n) \mid s \neq n, n \in \mathcal{N}(s)\}$，共 $|\mathcal{E}| = 42$ 个有向邻区对。

每个邻区对 $(s, n)$ 有独立的切换偏置参数 $\theta_{sn} \in \mathbb{R}$，参数向量为：

$$\boldsymbol{\theta} = \{\theta_{sn}\}_{(s,n) \in \mathcal{E}} \in \mathbb{R}^{42}$$

**切换决策函数**：给定参数 $\boldsymbol{\theta}$ 和轨迹 $\tau$，A3 算法产生服务小区序列：

$$\mathbf{c}(\boldsymbol{\theta}, \tau) = [c_1, c_2, \ldots, c_T] \in \{0,\ldots,C-1\}^T$$

**目标函数**：在训练集 $\mathcal{T}_{\text{train}} = \{\tau_1, \ldots, \tau_{72}\}$ 上最大化：

$$\mathcal{J}(\boldsymbol{\theta}) = \underbrace{\frac{\sum_{\tau} \sum_{t} \gamma_{t, c_t(\boldsymbol{\theta}, \tau)}}{\sum_{\tau} T_\tau}}_{\text{加权均值 SINR}} - \lambda_1 \underbrace{\frac{\sum_{\tau} N_{\text{HO}}(\boldsymbol{\theta}, \tau)}{|\mathcal{T}|}}_{\text{每轨迹平均切换次数}} - \lambda_2 \underbrace{\sum_{\tau} N_{\text{RLF}}(\boldsymbol{\theta}, \tau)}_{\text{总 RLF 次数}}$$

其中 $\lambda_1 = 0.1$，$\lambda_2 = 5.0$。均值 SINR 按轨迹长度加权；切换次数取每轨迹平均；RLF 次数取总和（$\lambda_2 = 5.0$ 已足够大，无需归一化）。

### 1.2 为什么不能用梯度方法

$\mathcal{J}(\boldsymbol{\theta})$ 关于 $\boldsymbol{\theta}$ **不可微**，原因：

A3 的切换决策是一个阶跃函数：

$$c_t = \begin{cases} n^* & \text{if } M_{n^*}(t) - M_s(t) > \theta_{sn^*} + \text{Hys 且持续 TTT} \\ c_{t-1} & \text{otherwise} \end{cases}$$

当 $\theta_{sn}$ 连续变化时，$c_t$ 在某个临界点发生跳变，导致 $\mathcal{J}$ 关于 $\theta_{sn}$ 是分段常数函数，梯度几乎处处为零或不存在。

因此必须使用**无梯度优化**方法。

### 1.3 坐标下降 + 网格搜索

**坐标下降**（Coordinate Descent）：每次只优化一个参数，其余固定：

$$\theta_{sn}^{(k+1)} = \arg\max_{\delta \in \mathcal{D}} \mathcal{J}(\theta_1^{(k)}, \ldots, \theta_{s,n-1}^{(k)}, \delta, \theta_{s,n+1}^{(k)}, \ldots)$$

**网格搜索**：搜索集合 $\mathcal{D} = \{-3.0, -2.5, -2.0, \ldots, 8.5, 9.0\}$，共 25 个候选值。

**完整算法**：

$$\theta_{sn}^* = \arg\max_{\delta \in \mathcal{D}} \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathcal{J}_\tau(\boldsymbol{\theta}_{-sn}, \delta)$$

其中 $\boldsymbol{\theta}_{-sn}$ 表示除 $(s,n)$ 外所有参数保持当前最优值。

**计算复杂度**：$|\mathcal{E}| \times |\mathcal{D}| \times |\mathcal{T}| = 42 \times 25 \times 72 = 75{,}600$ 次轨迹评估。

### 1.4 A3 决策的完整数学描述

**状态变量**：
- $c_t$：当前服务小区
- $\text{TTT}_{n}(t)$：邻区 $n$ 的 TTT 计数器（单位：时隙）
- $\text{cool}(t)$：切换冷却计数器

**每时隙更新**（$t = 1, 2, \ldots, T$）：

**Step 1**：更新 TTT 计数器：

$$\text{TTT}_n(t) = \begin{cases} \text{TTT}_n(t-1) + 1 & \text{if } M_n(t) - M_{c_t}(t) > \theta_{c_t, n} + \text{Hys} \\ 0 & \text{otherwise} \end{cases}, \quad \forall n \neq c_t$$

**Step 2**：确定切换目标（若冷却结束）：

$$n^*(t) = \arg\max_{n: \text{TTT}_n(t) \geq \lceil \text{TTT}_{\text{ms}} / \Delta t \rceil} M_n(t)$$

其中 $\text{TTT}_{\text{ms}} = 80$ ms，$\Delta t = 40$ ms，故阈值为 2 个时隙。

**Step 3**：执行切换：

$$c_{t+1} = \begin{cases} n^*(t) & \text{if } n^*(t) \text{ 存在且 } \text{cool}(t) = 0 \\ c_t & \text{otherwise} \end{cases}$$

若发生切换，重置 $\text{TTT}_n(t) = 0, \forall n$，设置 $\text{cool} = 5$（200ms）。

---

## 二、GRU 模型

### 2.0 整体思路

GRU（Gated Recurrent Unit，门控循环单元）是一种循环神经网络，专为处理时序数据设计。其核心思想是**通过门控机制选择性地保留或遗忘历史信息**，从而在有限的隐状态空间中压缩整个历史序列的关键信息。

在切换决策问题中，UE 的信道状态随时间变化，当前时刻的最优切换决策不仅取决于当前的信道测量值，还取决于信道的变化趋势。例如，RSRP 正在持续下降（可能即将离开覆盖区）和 RSRP 短暂下降后回升（建筑物遮挡）是两种截然不同的情况，需要不同的切换策略。GRU 通过两个门来建模这种时序依赖：

- **更新门** $\mathbf{z}_t$：决定当前时刻的信息有多少应该更新到隐状态中。当信道发生突变时，更新门开大，让新信息主导；当信道稳定时，更新门关小，保留历史信息。
- **重置门** $\mathbf{r}_t$：决定历史信息有多少应该被遗忘。当需要"重新开始"（如切换后进入新小区）时，重置门关小，忽略之前的历史。

两层 GRU 的设计使模型能够学习不同抽象层次的时序特征：第一层提取局部的信道变化模式（如 RSRP 的短期趋势），第二层整合更长时间跨度的趋势信息（如是否正在接近小区边界）。最终取最后时隙的隐状态，经过全连接层输出 7 个小区的切换收益预测。

GRU 的主要优势是**对特征尺度变化鲁棒**，且能自然处理变长序列。其局限是顺序计算（无法并行），且隐状态维度有限，对非常长的历史依赖可能存在遗忘。

### 2.1 输入输出

- **输入**：$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_W] \in \mathbb{R}^{B \times W \times F}$，$W=10$，$F=70$
- **输出**：$\hat{\mathbf{y}} \in \mathbb{R}^{B \times C}$，$C=7$（logits，未经 softmax）

### 2.2 GRU 单元数学推导

GRU 是 LSTM 的简化版本，用两个门替代三个门。

设 $\mathbf{h}_{t-1} \in \mathbb{R}^H$（上一时刻隐状态），$\mathbf{x}_t \in \mathbb{R}^F$（当前输入），$H=128$。

**更新门**（决定保留多少历史信息）：

$$\mathbf{z}_t = \sigma\left(W_z \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_z\right) \in [0,1]^H$$

$W_z \in \mathbb{R}^{H \times (H+F)}$，$\sigma$ 为 sigmoid 函数。

**重置门**（决定遗忘多少历史信息）：

$$\mathbf{r}_t = \sigma\left(W_r \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_r\right) \in [0,1]^H$$

**候选隐状态**（基于重置后的历史和当前输入）：

$$\tilde{\mathbf{h}}_t = \tanh\left(W_h \begin{bmatrix} \mathbf{r}_t \odot \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_h\right) \in [-1,1]^H$$

$\odot$ 为逐元素乘法（Hadamard 积）。

**最终隐状态**（更新门的插值）：

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**直觉**：
- $\mathbf{z}_t \to 0$：保留历史 $\mathbf{h}_{t-1}$（信道稳定，无需更新）
- $\mathbf{z}_t \to 1$：用新信息 $\tilde{\mathbf{h}}_t$ 替换历史（信道突变，需要更新）
- $\mathbf{r}_t \to 0$：忽略历史（重新开始）
- $\mathbf{r}_t \to 1$：完全利用历史

### 2.3 两层 GRU 的信息流

$$\mathbf{h}_t^{(1)} = \text{GRU}^{(1)}(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)})$$

$$\mathbf{h}_t^{(2)} = \text{GRU}^{(2)}(\mathbf{h}_t^{(1)}, \mathbf{h}_{t-1}^{(2)})$$

$$\hat{\mathbf{y}} = W_{\text{out}} \cdot \text{Dropout}(\mathbf{h}_W^{(2)}) + \mathbf{b}_{\text{out}}$$

**参数量**：
- GRU Layer 1：$3 \times H \times (F + H) = 3 \times 128 \times 198 = 76{,}032$
- GRU Layer 2：$3 \times H \times (H + H) = 3 \times 128 \times 256 = 98{,}304$
- FC：$H \times C = 128 \times 7 = 896$
- 总计：约 175K 参数

---

## 三、TCN 模型

### 3.0 整体思路

TCN（Temporal Convolutional Network，时序卷积网络）用卷积代替循环结构来处理时序数据。其核心思想是**通过因果膨胀卷积在保证实时性的同时，以指数级扩大感受野**。

"因果"意味着卷积核只访问当前时刻及之前的时刻，不看未来，保证模型可以在线部署（每个时隙只需要历史数据即可推理）。"膨胀"意味着卷积核的采样间隔随层数指数增长（$d = 1, 2, 4$），使得三层卷积就能覆盖全部 10 个历史时隙，而不需要像普通卷积那样堆叠很多层。

残差连接解决了深层网络的梯度消失问题，使训练更稳定。当输入输出通道数不同时，用 $1\times1$ 卷积做维度对齐。

TCN 的主要优势是**并行计算**：不同时刻的卷积操作相互独立，可以同时计算，训练速度比 GRU 快。其局限是感受野固定（由网络结构决定），无法像 Transformer 那样动态关注任意位置；BatchNorm 在推理时需要使用训练集统计量，导致推理时延较高。

### 3.1 因果卷积的数学定义

标准 1D 卷积（非因果）：

$$(\mathbf{x} * \mathbf{w})[t] = \sum_{k=-(K-1)/2}^{(K-1)/2} \mathbf{w}[k] \cdot \mathbf{x}[t+k]$$

**因果卷积**（只看历史）：

$$(\mathbf{x} *_{\text{causal}} \mathbf{w})[t] = \sum_{k=0}^{K-1} \mathbf{w}[k] \cdot \mathbf{x}[t-k]$$

实现方式：在序列左侧填充 $K-1$ 个零，然后做标准卷积，截去右侧多余输出。

**膨胀因果卷积**（dilation $d$）：

$$(\mathbf{x} *_{d} \mathbf{w})[t] = \sum_{k=0}^{K-1} \mathbf{w}[k] \cdot \mathbf{x}[t - d \cdot k]$$

感受野（单层）：$1 + (K-1) \cdot d$

### 3.2 感受野分析

三层 TCN，$K=3$，$d \in \{1, 2, 4\}$：

| 层 | 膨胀因子 $d$ | 单层感受野 | 累积感受野 |
|---|---|---|---|
| 1 | 1 | $1 + 2 \times 1 = 3$ | 3 |
| 2 | 2 | $1 + 2 \times 2 = 5$ | $3 + (5-1) = 7$ |
| 3 | 4 | $1 + 2 \times 4 = 9$ | $7 + (9-1) = 15$ |

最终感受野 15 > 序列长度 10，覆盖全部历史。

### 3.3 TCN 残差块的完整数学描述

设输入 $\mathbf{x} \in \mathbb{R}^{C_{\text{in}} \times T}$，输出 $\mathbf{y} \in \mathbb{R}^{C_{\text{out}} \times T}$：

$$\mathbf{a}_1 = \text{Dropout}(\text{ReLU}(\text{BN}(\text{CausalConv}_{d}^{C_{\text{in}} \to C_{\text{out}}}(\mathbf{x}))))$$

$$\mathbf{a}_2 = \text{Dropout}(\text{ReLU}(\text{BN}(\text{CausalConv}_{d}^{C_{\text{out}} \to C_{\text{out}}}(\mathbf{a}_1))))$$

$$\mathbf{y} = \text{ReLU}\left(\mathbf{a}_2 + \underbrace{W_{1\times1} \mathbf{x}}_{\text{残差投影（若 } C_{\text{in}} \neq C_{\text{out}}\text{）}}\right)$$

**BatchNorm**（训练时）：

$$\text{BN}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

其中 $\mu_B, \sigma_B^2$ 为当前 batch 的均值和方差，$\gamma, \beta$ 为可学习参数。

### 3.4 完整前向传播

$$\mathbf{X}' = \text{Linear}_{F \to 64}(\mathbf{X}) \in \mathbb{R}^{B \times T \times 64}$$

$$\mathbf{Z}^{(0)} = \mathbf{X}'^{\top} \in \mathbb{R}^{B \times 64 \times T} \quad \text{（转置为 channel-first）}$$

$$\mathbf{Z}^{(1)} = \text{TCNBlock}_{d=1}^{64 \to 64}(\mathbf{Z}^{(0)}) \in \mathbb{R}^{B \times 64 \times T}$$

$$\mathbf{Z}^{(2)} = \text{TCNBlock}_{d=2}^{64 \to 128}(\mathbf{Z}^{(1)}) \in \mathbb{R}^{B \times 128 \times T}$$

$$\mathbf{Z}^{(3)} = \text{TCNBlock}_{d=4}^{128 \to 128}(\mathbf{Z}^{(2)}) \in \mathbb{R}^{B \times 128 \times T}$$

$$\mathbf{z} = \mathbf{Z}^{(3)}_{:,:,-1} \in \mathbb{R}^{B \times 128} \quad \text{（取最后时刻）}$$

$$\hat{\mathbf{y}} = W_{\text{out}} \cdot \text{Dropout}(\mathbf{z}) + \mathbf{b}_{\text{out}} \in \mathbb{R}^{B \times 7}$$

---

## 四、Transformer 模型

### 4.0 整体思路

Transformer 通过 Self-Attention 机制处理时序数据，其核心思想是**让序列中的每个位置都能直接关注其他任意位置**，而不需要像 GRU 那样逐步传递信息，也不需要像 TCN 那样通过多层卷积扩大感受野。

在切换决策问题中，Self-Attention 的优势在于：模型可以直接学习"哪些历史时隙对当前预测最重要"。例如，如果 3 个时隙前发生了一次 RSRP 突降，模型可以直接关注那个时刻，而不需要通过 GRU 的隐状态逐步传递这个信息。

本文的 Transformer 设计有以下几个关键特点：

1. **因果掩码**：上三角掩码确保时隙 $t$ 的预测只能看到 $t$ 及之前的时隙，保证实时性。
2. **CLS token**：在序列开头插入一个可学习的特殊 token，经过 Transformer 编码后，CLS token 的输出聚合了整个序列的信息，用于最终分类。这避免了对所有时隙输出取平均的信息损失。
3. **Pre-LN**：先做 LayerNorm 再做注意力（与原始 Transformer 相反），训练更稳定，不需要 warmup。
4. **小型设计**：$d_{\text{model}}=64$，2 层，4 头，参数量约 70K，是三种模型中最小的，推理时延也最短（575 μs）。

Transformer 在本实验中取得了最高的均值 SINR（+9.36 dB），说明 Self-Attention 对信道时序特征的建模能力优于 GRU 和 TCN。其局限是对序列长度的计算复杂度为 $O(W^2)$，但在 $W=10$ 的短序列场景下不是瓶颈。

### 4.1 位置编码

正弦位置编码（Vaswani et al., 2017）：

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $pos \in \{0, 1, \ldots, W\}$（含 CLS token），$i \in \{0, 1, \ldots, d_{\text{model}}/2 - 1\}$，$d_{\text{model}} = 64$。

### 4.2 Multi-Head Self-Attention

设输入序列 $\mathbf{Z} \in \mathbb{R}^{(W+1) \times d_{\text{model}}}$（含 CLS token，$W+1=11$），头数 $h=4$，每头维度 $d_k = d_{\text{model}}/h = 16$。

**第 $i$ 个注意力头**：

$$Q_i = \mathbf{Z} W_i^Q, \quad K_i = \mathbf{Z} W_i^K, \quad V_i = \mathbf{Z} W_i^V$$

其中 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$。

**缩放点积注意力**（含因果掩码）：

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i \in \mathbb{R}^{(W+1) \times d_k}$$

**因果掩码**：

$$M_{pq} = \begin{cases} 0 & p = 0 \text{（CLS 行，可看所有位置）} \\ 0 & q \leq p \text{（当前及历史位置）} \\ -\infty & q > p \text{（未来位置，softmax 后为 0）} \end{cases}$$

**多头拼接**：

$$\text{MHA}(\mathbf{Z}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$。

### 4.3 Pre-LN Transformer 层

与原始 Transformer（Post-LN）不同，本文使用 Pre-LN（先归一化）：

**Self-Attention 子层**：

$$\mathbf{Z}' = \mathbf{Z} + \text{MHA}(\text{LN}_1(\mathbf{Z}))$$

**Feed-Forward 子层**：

$$\mathbf{Z}'' = \mathbf{Z}' + \text{FFN}(\text{LN}_2(\mathbf{Z}'))$$

其中 FFN 为两层全连接：

$$\text{FFN}(\mathbf{x}) = W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，$d_{\text{ff}} = 128$。

**Pre-LN 的优势**：梯度直接通过残差连接传播，训练更稳定，不需要 warmup 学习率调度。

### 4.4 CLS Token 的作用

CLS token $\mathbf{e}_{\text{cls}} \in \mathbb{R}^{d_{\text{model}}}$ 是可学习参数，初始化为截断正态分布 $\mathcal{N}(0, 0.02^2)$。

拼接后的输入序列：

$$\mathbf{Z}^{(0)} = [\mathbf{e}_{\text{cls}}; \mathbf{x}_1'; \mathbf{x}_2'; \ldots; \mathbf{x}_W'] + \text{PE} \in \mathbb{R}^{(W+1) \times d_{\text{model}}}$$

经过 $L=2$ 层 Transformer 后，CLS token 的输出 $\mathbf{Z}^{(L)}_{0,:}$ 通过注意力机制聚合了整个序列的信息（因为 CLS 行的掩码全为 0，可以看到所有位置）。

**分类头**：

$$\hat{\mathbf{y}} = W_{\text{out}} \cdot \text{Dropout}(\text{LN}(\mathbf{Z}^{(L)}_{0,:})) + \mathbf{b}_{\text{out}} \in \mathbb{R}^C$$

### 4.5 完整前向传播

$$\mathbf{X}' = \text{Linear}_{70 \to 64}(\mathbf{X}) \in \mathbb{R}^{B \times 10 \times 64}$$

$$\mathbf{Z}^{(0)} = \text{Dropout}\left([\mathbf{e}_{\text{cls}}^{\text{expand}}; \mathbf{X}'] + \text{PE}\right) \in \mathbb{R}^{B \times 11 \times 64}$$

$$\mathbf{Z}^{(1)} = \text{TransformerLayer}_1(\mathbf{Z}^{(0)}, M) \in \mathbb{R}^{B \times 11 \times 64}$$

$$\mathbf{Z}^{(2)} = \text{TransformerLayer}_2(\mathbf{Z}^{(1)}, M) \in \mathbb{R}^{B \times 11 \times 64}$$

$$\hat{\mathbf{y}} = W_{\text{out}} \cdot \text{Dropout}(\text{LN}(\mathbf{Z}^{(2)}_{:,0,:})) + \mathbf{b}_{\text{out}} \in \mathbb{R}^{B \times 7}$$

**参数量**：
- 输入投影：$70 \times 64 = 4{,}480$
- CLS token：$64$
- 位置编码：不可学习（固定）
- 每层 Transformer：$4 \times d_k \times d_{\text{model}} \times 4 + 2 \times d_{\text{model}} \times d_{\text{ff}} = 4 \times 16 \times 64 \times 4 + 2 \times 64 \times 128 = 16{,}384 + 16{,}384 = 32{,}768$
- 2 层：$65{,}536$
- 分类头：$64 \times 7 = 448$
- 总计：约 70K 参数（最小）

---

## 五、三种模型的本质区别

| 维度 | GRU | TCN | Transformer |
|---|---|---|---|
| 时序建模方式 | 递归（顺序处理） | 卷积（并行处理） | 注意力（全局依赖） |
| 长程依赖 | 通过门控传递，有遗忘 | 通过膨胀扩大感受野 | 直接建模任意距离 |
| 并行计算 | ❌（依赖前一时刻） | ✅（各时刻独立） | ✅（矩阵运算） |
| 参数量 | ~175K | ~200K | ~70K |
| 推理时延 | 572 μs | 2284 μs | 575 μs |
| 对序列长度的复杂度 | $O(W)$ | $O(W \log W)$ | $O(W^2)$ |

**TCN 推理慢的原因**：BatchNorm 在推理时需要使用训练集统计量，且卷积操作的内存访问模式不如矩阵乘法高效。

**Transformer 参数最少但效果最好的原因**：Self-Attention 可以直接建模任意两个时隙之间的依赖，不需要像 GRU 那样逐步传递信息，也不需要像 TCN 那样通过多层卷积扩大感受野。在序列长度较短（$W=10$）时，Transformer 的 $O(W^2)$ 复杂度不是瓶颈。