
---

# **基于动态性能包络与地形感知的无人机协同巡检系统建模**
**(Optimization of UAV Inspection Systems with Dynamic Performance Envelope & Terrain Awareness)**

---

## **I. 符号与变量定义 (Nomenclature)**

为了确保数学描述的准确性与一致性，本节定义了模型中使用的所有集合、物理参数及决策变量。

### **A. 集合与索引 (Sets & Indices)**
*   $\mathcal{L}$: **候选机巢位置集合**，索引 $l$。由聚类算法生成，位于风机密集区中心。
*   $\mathcal{K}$: **待巡检风机集合**，索引 $k$。
*   $\mathcal{N}_l$: **机巢 $l$ 的局部节点网络**，$\mathcal{N}_l = \mathcal{K}_l \cup \{O_l\}$，其中 $O_l$ 为机巢 $l$ 的基地节点。
*   $\mathcal{T}$: **机巢类型集合**，索引 $\tau \in \{1, 2, 3, 4\}$。
    *   *说明*：$\tau$ 的数值直接代表该类型机巢可容纳的无人机数量。
*   $\mathcal{V}_{l,\tau}$: **无人机集合**，索引 $v \in \{1, \dots, \tau\}$，表示隶属于机巢 $l$ (类型 $\tau$) 的无人机。
*   $\mathcal{R}$: **航次集合**，索引 $r \in \{1, \dots, R_{max}\}$，表示单架无人机的多轮飞行任务。

### **B. 物理与工程参数 (Physical & Engineering Parameters)**

#### **1. 动力学与气动参数**
*   $m$: 无人机总质量 (kg)。
*   $g$: 重力加速度 ($9.8 \text{ m/s}^2$)。
*   $\rho$: 空气密度 ($1.225 \text{ kg/m}^3$)。
*   $A$: 旋翼桨盘总面积 (m$^2$)。
*   $v_0$: 悬停诱导速度 (m/s)，$v_0 = \sqrt{mg / 2\rho A}$。
*   $U_{tip}$: 旋翼叶尖速度 (m/s)。
*   $d_0, C_{d0}$: 废阻系数与叶片型阻系数。
*   $\kappa$: 诱导功率修正因子。

#### **2. 飞行性能包络 (Performance Envelope)**
*   **$U_{max}$**: **最大允许速度** (m/s)。
    *   *定义*：无人机的安全操作极限。它既是**最大允许地速**（防止顺风失控），也是**最大允许空速**（防止逆风过载）。
*   $V_{up}, V_{down}$: 垂直爬升与下降速率 (m/s)。
*   $\mathbf{W}$: 风速矢量 $(W_x, W_y)$ (m/s)。
*   $E_{bat}$: 电池总可用能量 (J)。
*   $H_{safe}$: 地形跟随的安全高度裕度 (m)。
*   $T_{serv}$: 单个风机的悬停作业时间 (s)。

#### **3. 经济参数**
*   $C_{base}^{\tau}$: 类型为 $\tau$ 的机巢**基准建设成本**。体现规模效应 ($C_{base}^4 < 4 C_{base}^1$)。
*   $C_{uav}$: 单架无人机的购置成本。
*   $\eta$: 单架无人机最大服务风机数 (Phase I 估算用)。

### **C. 决策变量 (Decision Variables)**

#### **Phase I 变量 (选址)**
*   $x_{l, \tau} \in \{0, 1\}$: **选址变量**。若在候选点 $l$ 建设类型为 $\tau$ 的机巢，则为 1。
*   $y_{l, k} \in \{0, 1\}$: **指派变量**。若风机 $k$ 指派给机巢 $l$，则为 1。

#### **Phase II 变量 (路径)**
*   $z_{i,j,v,r}^{l, \tau} \in \{0, 1\}$: **路由变量**。无人机 $v$ (航次 $r$) 是否从 $i$ 飞往 $j$。
*   $a_{i,v,r}^{l, \tau} \geq 0$: **到达时间**。无人机 $v$ (航次 $r$) 到达节点 $i$ 的时刻。
*   $e_{i,v,r}^{l, \tau} \geq 0$: **剩余能量**。无人机 $v$ (航次 $r$) 到达节点 $i$ 时的电池能量。
*   $Z_{MS} \geq 0$: **最大完工时间 (Makespan)**。

---

## **II. 物理层建模：动态飞行策略与能耗解析**
**(Physics Layer: Dynamic Flight Strategy & Energy Analysis)**

本节建立底层模型，推导在风场 $\mathbf{W}$ 和速度限制 $U_{max}$ 下，无人机在任意两点间的**实际飞行时间**与**能耗**。这是两个优化阶段的基础。

### **1. 动态速度策略推导 (Dynamic Velocity Strategy)**

对于给定的航向角 $\theta$，定义单位方向矢量 $\mathbf{d} = [\sin\theta, \cos\theta]$。
根据速度矢量三角形：$\mathbf{V}_{air} = \mathbf{V}_{ground} - \mathbf{W}$。设地速大小为 $V_g$，则 $\mathbf{V}_{ground} = V_g \mathbf{d}$。

我们采用 **"最大性能包络 (Max Performance Envelope)"** 策略，分两种情形讨论：

#### **情形 A：地速受限 (顺风/侧风)**
当风力有助于飞行或侧向吹时，若全速飞行可能导致地速过快，危及安全。
*   **尝试假设**：设定地速 $V_g = U_{max}$。
*   **检验空速**：计算此时的空速模长 $V_{try} = \| U_{max} \mathbf{d} - \mathbf{W} \|$。
*   **判定**：若 $V_{try} \le U_{max}$，则假设成立。
    *   **结果**：实际地速 $V_g^* = U_{max}$，实际空速 $V_{air}^* = V_{try}$。

#### **情形 B：空速受限 (强逆风)**
当遭遇强逆风时，若强行保持 $U_{max}$ 的地速，所需空速将超过物理极限。此时必须限制空速，牺牲地速。
*   **条件**：情形 A 中的 $V_{try} > U_{max}$。
*   **修正**：强制设定空速模长 $\|\mathbf{V}_{air}\| = U_{max}$。
*   **求解地速**：
    将矢量方程平方：$\| \mathbf{V}_{air} \|^2 = \| V_g \mathbf{d} - \mathbf{W} \|^2$
    代入极限条件：$U_{max}^2 = (V_g \mathbf{d} - \mathbf{W}) \cdot (V_g \mathbf{d} - \mathbf{W})$
    展开得到关于 $V_g$ 的**一元二次方程**：
    $$
    V_g^2 - \underbrace{2(\mathbf{d} \cdot \mathbf{W})}_{b} V_g + \underbrace{(\|\mathbf{W}\|^2 - U_{max}^2)}_{c} = 0 \quad (Eq. 1)
    $$
*   **结果**：取物理意义的正根：
    $$ V_g^* = \frac{-b + \sqrt{b^2 - 4c}}{2}, \quad V_{air}^* = U_{max} $$

### **2. 混合功率模型 (Hybrid Power Model)**

基于上述计算得到的实际空速 $V_{air}^*$，代入气动功率公式：
$$
P_{cruise}(V_{air}^*) = \underbrace{P_{hover} \sqrt{\sqrt{1+\frac{(V_{air}^*)^4}{4v_0^4}} - \frac{(V_{air}^*)^2}{2v_0^2}}}_{\text{诱导功率}} + \underbrace{\frac{\rho \sigma A U_{tip}^3 C_{d0}}{8} \left(1+\frac{3(V_{air}^*)^2}{U_{tip}^2}\right)}_{\text{型阻功率}} + \underbrace{\frac{1}{2} d_0 \rho A (V_{air}^*)^3}_{\text{废阻功率}} \quad (Eq. 2)
$$

### **3. 地形感知的三维路径代价 (Terrain-Aware Path Cost)**

对于任意两点 $i, j$，基于数字高程模型 (DEM) 提取路径上的最高海拔，计算安全高度 $H_{path}^{i,j}$：
$$ H_{path}^{i,j} = \max_{(x,y) \in \text{Path}_{i,j}} \{ Z_{dem}(x,y) \} + H_{safe} $$

单程飞行时间 $T_{i,j}$ 和飞行能耗 $E_{flight}^{i,j}$ 计算如下：

$$
T_{i,j} = \frac{D_{xy}^{i,j}}{V_g^*} + \frac{\Delta H_{up}}{V_{up}} + \frac{\Delta H_{down}}{V_{down}} \quad (Eq. 3)
$$

$$
E_{flight}^{i,j} = \frac{D_{xy}^{i,j}}{V_g^*} P_{cruise}(V_{air}^*) + \frac{\Delta H_{up}}{V_{up}} (P_{hover} + mgV_{up}) + \frac{\Delta H_{down}}{V_{down}} (P_{hover} - mgV_{down}) \quad (Eq. 4)
$$
*   注：$\Delta H_{up} = \max(0, H_{path}^{i,j} - Z_i)$，$\Delta H_{down} = \max(0, H_{path}^{i,j} - Z_j)$。

### **4. 往返任务总能耗解析 (Round-Trip Mission Energy Derivation)**

这是计算物理可达性 $R_{l,k}$ 的核心依据。
假设执行一个从机巢 $l$ 出发，服务风机 $k$，然后返回机巢 $l$ 的闭环任务。总能耗 $E_{mission}(l,k)$ 由以下三部分组成：

**(1) 水平巡航能耗 ($E_{horiz}$)**
考虑风场影响，去程 ($l \to k$) 和回程 ($k \to l$) 的航向相反 ($\theta$ vs $\theta+\pi$)，导致 $V_g^*$ 和 $P_{cruise}$ 均不同：
$$
E_{horiz} = D_{xy}^{l,k} \left( \frac{P_{cruise}(\theta_{lk})}{V_g^*(\theta_{lk})} + \frac{P_{cruise}(\theta_{lk}+\pi)}{V_g^*(\theta_{lk}+\pi)} \right)
$$

**(2) 垂直机动能耗 ($E_{vert}$)**
无人机需从起点 $Z_l$ 爬升至安全高度 $H_{path}$ 再下降至目标 $Z_k$，回程同理。
总爬升高度 $\Delta H_{total\_up} = (H_{path} - Z_l) + (H_{path} - Z_k)$。
总下降高度 $\Delta H_{total\_down} = (H_{path} - Z_k) + (H_{path} - Z_l)$。
垂直总能耗为：
$$
E_{vert} = P_{climb} \cdot \frac{\Delta H_{total\_up}}{V_{up}} + P_{desc} \cdot \frac{\Delta H_{total\_down}}{V_{down}}
$$
代入 $P_{climb} \approx P_{hover} + mgV_{up}$ 和 $P_{desc} \approx P_{hover} - mgV_{down}$ 并整理得：
$$
E_{vert} = \left( \frac{P_{hover}}{V_{up}} + mg \right) \Delta H_{total\_up} + \left( \frac{P_{hover}}{V_{down}} - mg \right) \Delta H_{total\_down}
$$

**(3) 悬停作业能耗 ($E_{serv}$)**
$$
E_{serv} = P_{hover} \cdot T_{serv}
$$

**(4) 总公式**
$$
E_{mission}(l,k) = E_{horiz} + E_{vert} + E_{serv} \quad (Eq. 5)
$$

### **5. 物理可达性矩阵 ($R_{l,k}$)**
通过比较总能耗与电池容量，构建 Phase I 的硬约束参数：
$$
R_{l,k} = \begin{cases} 
1, & \text{若 } E_{mission}(l,k) \le E_{bat} \\
0, & \text{否则}
\end{cases} \quad (Eq. 6)
$$

---

## **III. 第一阶段模型：异构机巢选址与协同指派优化**
**(Phase I: Heterogeneous Nest Deployment & Collaborative Assignment Optimization)**

**目标**：在满足物理可达性、风机全覆盖和机巢容量限制的前提下，确定机巢的最佳位置和类型，以及每个风机的归属机巢，以最小化总建设成本和无人机总采购成本。

### **1. 决策变量 (Decision Variables)**
*   $x_{l, \tau} \in \{0, 1\}$: **选址变量**。如果候选位置 $l$ 建设了类型为 $\tau$ 的机巢，则为 1，否则为 0。
*   $y_{l, k} \in \{0, 1\}$: **指派变量**。如果风机 $k$ 被分配给机巢 $l$ 负责，则为 1，否则为 0。

### **2. 目标函数 (Objective Function)**
最小化包含基础设施与设备的**总经济成本 (Total Cost of Ownership)**：
$$
\min Z_1 = \sum_{l \in \mathcal{L}} \sum_{\tau \in \mathcal{T}} \left( C_{base}^{\tau} + \tau \cdot C_{uav} \right) \cdot x_{l,\tau} \quad (M1.1)
$$
*   **物理含义**：
    *   $C_{base}^{\tau}$：不同类型机巢的基准建设成本，体现了规模效应（例如 $C_{base}^4 < 4 \times C_{base}^1$）。
    *   $\tau \cdot C_{uav}$：根据机巢类型 $\tau$ 决定的无人机购置成本。

### **3. 约束条件 (Constraints)**

**(1) 选址唯一性约束 (Unique Nest Type per Location)**
每个候选位置 $l$ 最多只能建设一个类型的机巢（即不同类型互斥）：
$$ \sum_{\tau \in \mathcal{T}} x_{l,\tau} \le 1, \quad \forall l \in \mathcal{L} \quad (M1.2) $$

**(2) 风机全覆盖约束 (Full Turbine Coverage)**
每个风机 $k$ 必须且只能被分配给一个机巢进行巡检：
$$ \sum_{l \in \mathcal{L}} y_{l,k} = 1, \quad \forall k \in \mathcal{K} \quad (M1.3) $$

**(3) 建设依托约束 (Assignment Requires Nest Deployment)**
只有当候选位置 $l$ 建设了机巢（无论何种类型），才能将风机 $k$ 指派给它：
$$ y_{l,k} \le \sum_{\tau \in \mathcal{T}} x_{l,\tau}, \quad \forall l \in \mathcal{L}, k \in \mathcal{K} \quad (M1.4) $$
*   **物理含义**：确保没有“幽灵机巢”接收任务。

**(4) 物理可达性约束 (Physical Reachability Constraint - 核心连接)**
严禁将风机指派给物理上不可达（即往返能耗超过电池容量）的机巢。
$$ y_{l,k} \le R_{l,k}, \quad \forall l \in \mathcal{L}, k \in \mathcal{K} \quad (M1.5) $$
*   **物理含义**：$R_{l,k}$ 是基于物理层公式 (Eq. 6) 预先计算的硬约束。如果 $R_{l,k}=0$，则 $y_{l,k}$ 必须为 0。这是连接物理层与决策层的桥梁。

**(5) 机巢容量与服务上限约束 (Nest Capacity Constraint)**
分配给机巢 $l$ 的风机总数，不得超过该机巢类型 $\tau$ 所提供的总服务能力。
$$ \sum_{k \in \mathcal{K}} y_{l,k} \le \eta \cdot \sum_{\tau \in \mathcal{T}} (\tau \cdot x_{l,\tau}), \quad \forall l \in \mathcal{L} \quad (M1.6) $$

---

## **IV. 第二阶段模型：多航次路径规划与负载均衡**
**(Phase II: Multi-Trip Routing Optimization & Load Balancing)**

**目标**：基于第一阶段模型输出的已建设机巢 ($x_{l,\tau}=1$) 及其所辖的风机集合 ($\mathcal{K}_l$)，为每个机巢内的多架无人机规划具体的“多航次”路径，以最小化整个巡检任务的最大完工时间 ($Z_{MS}$)，实现机队间的负载均衡。

### **1. 决策变量 (Decision Variables)**
*   $z_{i,j,v,r}^{l, \tau} \in \{0, 1\}$: 若机巢 $l$ (类型 $\tau$) 的无人机 $v$ 在第 $r$ 航次中，从节点 $i$ 直接飞往节点 $j$，则为 1。
*   $a_{i,v,r}^{l, \tau} \geq 0$: 无人机 $v$ (第 $r$ 航次) **到达**节点 $i$ 时的时刻。
*   $e_{i,v,r}^{l, \tau} \geq 0$: 无人机 $v$ (第 $r$ 航次) **到达**节点 $i$ 时的剩余能量。
*   $Z_{MS} \geq 0$: 最大完工时间。

### **2. 目标函数 (Objective Function)**
最小化整个巡检任务的最大完工时间 (Makespan)：
$$
\min Z_{MS} \quad (M2.1)
$$
*   **物理含义**：此目标函数旨在优化整个无人机机队的效率，确保最忙碌的无人机也能尽早完成所有任务，从而缩短项目的总执行周期。

### **3. 约束条件 (Constraints)**

**(1) 任务覆盖与流守恒**
每个风机必须被访问一次：
$$
\sum_{v \in \mathcal{V}_{l,\tau}} \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_l} z_{k,j,v,r}^{l, \tau} = 1, \quad \forall l \in L^*, \tau \in \mathcal{T}, k \in \mathcal{K}_l \quad (M2.2)
$$
节点流平衡（进 = 出）：
$$
\sum_{i \in \mathcal{N}_l} z_{i,k,v,r}^{l, \tau} = \sum_{j \in \mathcal{N}_l} z_{k,j,v,r}^{l, \tau}, \quad \forall k \in \mathcal{K}_l \quad (M2.3)
$$

**(2) 航次闭环与时序**
每个航次必须从基地 $O_l$ 出发并返回基地 $O_l$（允许空航次）：
$$
\sum_{j \in \mathcal{N}_l} z_{O_l,j,v,r}^{l, \tau} = \sum_{i \in \mathcal{N}_l} z_{i,O_l,v,r}^{l, \tau} \leq 1 \quad (M2.4)
$$
航次 $r+1$ 必须在航次 $r$ 结束后才能开始（消除对称性）：
$$
\sum_{j \in \mathcal{N}_l} z_{O_l,j,v,r+1}^{l, \tau} \leq \sum_{i \in \mathcal{N}_l} z_{i,O_l,v,r}^{l, \tau} \quad (M2.5)
$$

**(3) 时间递推与 Makespan (MTZ约束)**
无人机到达节点 $j$ 的时间 $a_j$ 等于到达 $i$ 的时间加上作业和服务时间：
$$
a_{j,v,r}^{l, \tau} \geq a_{i,v,r}^{l, \tau} + T_{serv} + T_{i,j} - M(1 - z_{i,j,v,r}^{l, \tau}) \quad (M2.6)
$$
定义 Makespan 为最晚返回基地的时刻：
$$
Z_{MS} \geq \sum_{i \in \mathcal{N}_l} z_{i,O_l,v,r}^{l, \tau} \cdot a_{i,v,r}^{l, \tau} \quad (M2.7)
$$

**(4) 能量递推与限制**
无人机到达节点 $j$ 的剩余能量 $e_j$：
$$
e_{j,v,r}^{l, \tau} \leq e_{i,v,r}^{l, \tau} - E_{flight}^{i,j} - E_{serv} + M(1 - z_{i,j,v,r}^{l, \tau}) \quad (M2.8)
$$
初始满电约束：
$$
e_{O_l,v,r}^{l, \tau} = E_{bat} \quad (M2.9)
$$
能量非负约束：
$$
e_{i,v,r}^{l, \tau} \geq 0 \quad (M2.10)
$$

**(5) 动态安全返航 (Dynamic Safety Return)**
**关键安全约束**：无人机在任何风机节点 $i$ 完成作业后，剩余能量必须足以支持其直接飞回基地 $O_l$。
$$
e_{i,v,r}^{l, \tau} - E_{serv} \geq E_{flight}^{i,O_l} - M(1 - \sum_{j \in \mathcal{N}_l} z_{i,j,v,r}^{l, \tau}) \quad (M2.11)
$$
*   **物理含义**：防止无人机飞入“有去无回”的死角，确保在任意任务点中止时均可安全返航。