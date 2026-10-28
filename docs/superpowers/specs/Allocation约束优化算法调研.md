---
title: "Allocation 约束 SCF 的优化算法再调研"
subtitle: "从 Schur 补、缩减空间 Newton 到多保真预测—校正与分支跟踪"
---

# Allocation 约束 SCF 的优化算法再调研

\newpage

## 摘要 {-}

本报告以《Allocation 约束的不动点数学分析》（2026-10-28）[1] 为出发点，重新检索并评估可迁移到该问题的优化算法。与上一篇“前沿优化算法综述”不同，本文不做算法族的一般性介绍，而是围绕 allocation 约束的四个结构特征做选型：

1. **大状态、小约束**：密度 $\rho$ 的维度远大于约束数 $n_c$，约束 Jacobian 是一个 $n_c\times n_c$ 的 Schur 补；
2. **缺失密度响应**：固定密度/冻结 Hamiltonian 模型给出 $J_0=Q_\rho T_\mu$，精确 Jacobian 是 $J=Q_\rho(I-T_\rho)^{-1}T_\mu$，两者相差一整条响应通道；
3. **规范零空间**：$\sum_iA_i=I$ 导致 $J\mathbf 1=0$，约束乘子不唯一，LICQ 失效；
4. **非单调与折叠**：$q(\mu)$ 在金属、近简并、磁支附近可以有平台、跳变与多根，Fe$_2$ 的磁支翻转是直接例子。

结论是：**最匹配的算法族不是“更好的混合器”，而是“缩减空间（Schur）拟牛顿 + 全空间残差验收 + 多保真模型管理 + 分支延拓”**。固定密度子空间应当被降级为预条件器/预测器，而不是 Jacobian 或收敛判据。具体建议是四层求解器：内层 SCF（AA/Broyden）、响应层（线性化 SCF 解 $(I-T_\rho)^{-1}$）、外层缩减 Newton/Gauss–Newton/Levenberg–Marquardt（带规范投影与信赖域）、分支层（continuation/deflation）。当内层 SCF 成为瓶颈时，再考虑全耦合 KKT/鞍点预条件或非线性消除（ASPIN）。

本报告还给出：算法—结构特征能力矩阵、各算法的代价与失效边界、与之前两篇调研的衔接（哪些保留、哪些降级）、分阶段路线图与可验证的算法级测试计划。

**关键词**：约束 SCF；Schur 补；缩减空间 Newton；Gauss–Newton；增广拉格朗日；非线性预条件；多保真；space mapping；bilevel/MPEC；continuation；deflation；规范零空间

---

## 结论先行

在展开细节之前，先把结论按“该做什么、不该做什么、为什么”列清楚。

### 该做什么

1. **把外层变量取成 $\mu$，把内层状态取成 $\rho$，显式使用 Schur 补。** 外层不是“调 $\mu$ 让子空间读数等于目标”，而是“在电子不动点流形上求 $q(\rho^\star(\mu))=t$”。
2. **用线性化 SCF 计算响应，而不是用固定密度子空间代替响应。** 对每个约束方向 $v$，解
   $(I-T_\rho)y=T_\mu v$，再取 $Jv=Q_\rho y$。这一步的成本约等于一次线性化 SCF，可以复用现有混合器的预条件。
3. **用全空间 $q_{\rm full}$ 做唯一验收标准。** 固定密度读数只能用于产生初值、预测方向、构造预条件；不能用于 trust 半径、Jacobian 更新或收敛标志。
4. **对规范零空间做投影或固定参考乘子。** 在每次线性求解前投影掉 $\mathbf 1$ 方向，或在 Jacobian 上做 $J\leftarrow PJP$；否则 Broyden/Newton 会在无物理意义的方向上消耗步数。
5. **对非单调/折叠使用 continuation 或 Levenberg–Marquardt。** 单一 secant 没有全局保证；Fe$_2$ 的磁支翻转需要分支跟踪，而不是继续缩小步长。
6. **把固定密度模型当作多保真模型，而不是精确模型。** 用 output space mapping 做偏置校正，用 trust-region 模型管理保证收敛；低保真只提议，高保真验收。

### 不该做什么

1. **不要把 $J_0=Q_\rho T_\mu$ 当作精确 Jacobian。** 它缺的是 $Q_\rho(I-T_\rho)^{-1}T_\rho T_\mu$；当 $T_\rho$ 接近单位算符时，误差会被 $(I-T_\rho)^{-1}$ 放大。
2. **不要用固定密度子空间的 $q_{\rm sub}$ 作为外层收敛判据。** 子空间代数精确到 $10^{-14}$，只证明“在给定态上 $A_{i,\rm sub}$ 与 $A_i$ 一致”，不证明“投影模型等于自洽约束映射”。
3. **不要在同一个 trust-region 比值里混用两种残差。** 预测下降量来自哪个模型，实际下降量就必须来自同一个高保真模型；否则半径会不断收缩或过冲。
4. **不要在没有正则化的情况下对 $J$ 直接求逆。** 规范零空间、近线性相关的约束、弱响应都会让 $J$ 病态；用截断 SVD、Tikhonov、ALM 或伪逆。
5. **不要指望“多迭代几次固定密度内层”能消除模型误差。** 内层越精确地解一个错误的冻结模型，外层越容易得到与真实密度响应不匹配的 $\mu$。

### 为什么

因为 allocation 问题的数学核心是**约束的密度响应**，而不是约束算符的代数。$A_i$ 的单位分解、正性、低秩结构都没有问题；问题在于外层把“冻结模型”误当成了“精确模型”。所有推荐的算法，本质上都是在不同层次上补回这条响应通道：Schur 补补回它，JFNK 用 $Jv$ 补回它，非线性预条件用子问题补回它，多保真用高保真验收补回它，continuation 用分支跟踪补回它。

---

## 问题的精确数学定位

### 从约束 KS 到耦合残差

约束问题是

$$
\min_{\rho}\ E_{\rm KS}[\rho]
\quad\text{s.t.}\quad
q_i(\rho)=t_i,\quad i=1,\dots,n_c,
$$

拉格朗日函数与约束 Hamiltonian 为

$$
\mathcal L=E_{\rm KS}[\rho]-\sum_i\mu_i\bigl(q_i(\rho)-t_i\bigr),
\qquad
H[\rho;\mu]=H_{\rm KS}[\rho]+\sum_i\mu_iA_i,
\qquad
A_i=\frac{\delta q_i}{\delta\rho}.
$$

把一次电子迭代记为 $\rho'=T(\rho,\mu)$ [2]，则耦合残差是

$$
F(\rho,\mu)=
\begin{pmatrix}
\rho-T(\rho,\mu)\\
q(\rho)-t
\end{pmatrix}=0.
$$

在解附近线性化：

$$
\begin{pmatrix}
I-T_\rho & -T_\mu\\
Q_\rho & 0
\end{pmatrix}
\begin{pmatrix}
\delta\rho\\ \delta\mu
\end{pmatrix}
=-
\begin{pmatrix}
f_\rho\\ f_q
\end{pmatrix},
\qquad
T_\rho=\frac{\partial T}{\partial\rho},\quad
T_\mu=\frac{\partial T}{\partial\mu},\quad
Q_\rho=\frac{\partial q}{\partial\rho}.
$$

对密度块做 Schur 消元，得到约束方向的有效 Jacobian：

$$
J=Q_\rho(I-T_\rho)^{-1}T_\mu,
\qquad
\delta\mu=-J^{-1}f_q
\ \ (\text{当 }f_\rho=0).
$$

### 与非线性特征值问题的关系

固定 $\mu$ 的约束 KS 方程

$$
\left(H_{\rm KS}[\rho]+\sum_i\mu_iA_i\right)\psi=\epsilon S\psi,
\qquad
\rho=\sum_n f_n|\psi_n\rangle\langle\psi_n|
$$

是一个参数化非线性特征值问题（NEPv）：矩阵依赖密度，密度又依赖本征矢。SCF 是它的不动点迭代；约束外层则是在 NEPv 解流形上求 $q=t$。这个视角把“能隙—条件数”的联系显式化：能隙越小，本征矢的一阶响应 $1/(\epsilon_n-\epsilon_m)$ 越大，$T_\rho$ 越接近单位算符 [3,4]。

### 固定密度模型：零阶截断

若固定输入密度 $\rho_0$、只解冻结 Hamiltonian 的约束子问题，则

$$
q_0(\mu)=q\bigl(T(\rho_0,\mu)\bigr),
\qquad
J_0=Q_\rho T_\mu.
$$

精确 Jacobian 与它的差是

$$
J-J_0
=Q_\rho\left[(I-T_\rho)^{-1}-I\right]T_\mu
=Q_\rho(I-T_\rho)^{-1}T_\rho T_\mu.
$$

利用 Neumann 级数

$$
(I-T_\rho)^{-1}=I+T_\rho+T_\rho^2+\cdots,
$$

可以把精确 Jacobian 写成

$$
J=Q_\rho\left(I+T_\rho+T_\rho^2+\cdots\right)T_\mu,
\qquad
J_0=Q_\rho T_\mu.
$$

这不是一个“实现细节”，而是一个**截断阶数**的问题：$J_0$ 是零阶截断，$J$ 是所有阶的求和。后面会看到，这与机器学习里的“one-step unrolling vs implicit differentiation”是同一件事。

![Neumann 级数的截断：固定密度只是零阶近似](figures/fig1_neumann.png)

### 四个结构特征

| 特征 | 数学表现 | 直接后果 | 算法含义 |
|---|---|---|---|
| F1 大状态、小约束 | $\dim\rho\gg n_c$ | Schur 补只有 $n_c\times n_c$ | 缩减空间方法天然占优；显式 $J$ 可行 |
| F2 缺失密度响应 | $J-J_0=Q_\rho(I-T_\rho)^{-1}T_\rho T_\mu$ | 固定密度步长偏差 $1/(1-\alpha)$ | 必须解响应方程或做隐式微分 |
| F3 规范零空间 | $\sum_iA_i=I\Rightarrow J\mathbf 1=0$ | 乘子不唯一、KKT 奇异、LICQ 失效 | 投影/固定参考/正则化 |
| F4 非单调与折叠 | $q(\mu)$ 有平台、跳变、多根 | 单次 secant 跳支；trust 半径失效 | LM、continuation、deflation |
| F5 昂贵黑箱 | 每次 $q$ 求值含一次 SCF | 不能大量有限差分 | Broyden、JFNK、多保真 |
| F6 强内层耦合 | $T_\rho$ 谱半径接近 1 | 嵌套迭代慢；响应方程病态 | 全耦合 KKT、ASPIN、非线性消除 |

这张表是后文所有算法评估的坐标。一个算法是否“值得考虑”，取决于它是否直接针对 F1–F6 中的至少一个。

### 问题类别：MPEC、PDE 约束优化、bilevel 的精确关系

这个问题常被笼统地称为“bilevel optimization”，但严格说：

- **标准 bilevel**：内层是一个**有唯一解的极小化问题** $\rho^\star(\mu)=\arg\min_\rho E(\rho;\mu)$，外层用隐式微分求超梯度；
- **本文问题**：内层是一个**平衡条件**（不动点）$\rho^\star=T(\rho^\star;\mu)$，可能有多个解，且不要求是某个内层目标的全局极小。

因此它更准确的分类是 **equilibrium-constrained optimization**（平衡约束优化）或 **MPEC** [5,6]（mathematical program with equilibrium constraints）的一个连续、无互补约束的特例；同时它也是 **PDE-constrained optimization** 的一种 [7]（把 KS 方程看作状态方程，把 $\mu$ 看作控制/参数）。这三个类别的算法工具箱高度重叠，但理论假设不同：

| 类别 | 内层假设 | 主要工具 | 对本问题的适用性 |
|---|---|---|---|
| Bilevel | 内层唯一极小 | 隐式微分、超梯度、one-step unrolling | 结构相同，但“唯一极小”不成立 |
| MPEC | 内层是变分不等式/平衡 | 松弛、罚函数、SQP、C-stationarity | 连续平衡的特例；多解需额外处理 |
| PDE 约束优化 | 状态方程 $c(\rho,\mu)=0$ | 全耦合/缩减 Newton、伴随、Schur | 最贴近；可直接借用 LNKS/KKT 理论 |
| 约束根求解 | $r(\mu)=0$，$r$ 为黑箱 | Broyden、JFNK、LM、trust region | 最贴近外层；但没有用约束结构 |

结论：**用 PDE 约束优化的“缩减空间 Newton”作为主框架，用 MPEC/bilevel 的“隐式微分/多解”理论补足非唯一性，用黑箱根求解的 Broyden/LM 作为工程实现。**

---

## 算法族全景与能力矩阵

### 一张能力矩阵

下表把候选算法族与 F1–F6 对应起来。● 表示直接针对该特征，○ 表示部分有用，— 表示基本无关。

| 算法族 | F1 小约束 | F2 响应 | F3 零空间 | F4 折叠 | F5 黑箱 | F6 强耦合 |
|---|---|---|---|---|---|---|
| 缩减空间 Newton/GN/LM | ● | ● | ○ | ○ | ○ | ○ |
| 全耦合 KKT/鞍点预条件 | ○ | ● | ● | ○ | ○ | ● |
| SQP / 缩减 SQP | ● | ● | ○ | ○ | ○ | ○ |
| 增广拉格朗日 / 罚函数 | ○ | ○ | ● | ○ | ○ | ○ |
| 非线性预条件 / ASPIN | ○ | ● | ○ | — | ○ | ● |
| 多保真 / space mapping / TRMM | ● | ○ | — | ○ | ● | ○ |
| Bilevel / 隐式微分 | ● | ● | ○ | — | ○ | ○ |
| Continuation / deflation | ○ | — | — | ● | ○ | — |
| 零空间 / 约束预条件 | ○ | — | ● | — | — | ○ |
| Anderson/DIIS / Broyden（联合） | ● | — | ○ | ○ | ● | ○ |
| 无导数优化（DFO） | ● | — | — | ○ | ● | — |

### 三条主线

从矩阵可以看出三条主线，它们不是互斥的，而是分别回答不同问题：

1. **“怎么算响应”**：缩减空间 Newton、JFNK、非线性预条件、隐式微分。它们解决 F2。
2. **“怎么处理约束的退化”**：零空间投影、ALM、SQP、鞍点预条件。它们解决 F3。
3. **“怎么不被折叠骗”**：LM、trust region、continuation、deflation、多保真。它们解决 F4/F5。

一个成熟的求解器应当三条主线都有，而不是只在其中一条上堆算法。

### 之前调研中哪些仍然有效

上一篇综述中的算法并不是全部适用。按本问题的结构重新评估：

| 之前的方法 | 在本问题中的角色 | 判断 |
|---|---|---|
| Anderson/DIIS | 内层 SCF；联合 $(\rho,\mu)$ 的低风险外层 | 保留，但必须用全空间残差并加权 |
| Broyden | 外层 $J$ 的低秩更新；避免重复响应求解 | 保留，需规范投影与 trust region |
| JFNK/Newton–Krylov | 计算 $Jv$、解响应方程、全耦合 KKT | 保留，是核心工具 |
| 三次正则化/信赖域 | 外层全局化；处理非单调 | 保留，但 Hessian 成本高，优先 trust region+LM |
| 流形优化 | 波函数规范、占据-空带混合 | 降级为几何语言，不是主求解器 |
| 随机化/低秩 | 响应矩阵、Schur 补、$Jv$ 的压缩 | 保留，尤其 $n_c$ 大时 |
| Deflation/continuation | 多解/磁支 | 保留，Fe$_2$ 必需 |
| 学习型初猜/预条件 | 降低内层 SCF 迭代 | 可选，风险高 |
| 隐式微分/DEQ | 可微 DFT 的伴随、精确 Schur | 保留，与主线一致 |
| SDP/ADMM | 约束子问题、非光滑项 | 保留为子问题工具，不是主框架 |

---


---

## 缩减空间方法：把小的 Schur 补算清楚

### 为什么“小约束”是最大的结构优势

密度 $\rho$ 的维度可以是 $10^5$–$10^9$，而约束数 $n_c$ 通常只是“几个磁矩/几个片段电荷”的量级（几十到几百）。Schur 补 $J$ 只有 $n_c\times n_c$，这意味着：

- 可以**显式形成** $J$，用稠密线性代数（LU、SVD、伪逆）处理；
- 可以用 $n_c$ 次响应求解构造 $J$ 的每一列；
- 可以用低秩/随机化方法压缩 $J$，而不会损失太多；
- 可以把 $J$ 的奇异值分解直接用于诊断秩亏、零空间与折叠。

这与“大状态”的直觉相反：真正难的从来不是 $n_c$ 维的线性代数，而是每一次 $Jv$ 都要解一个**线性化 SCF**。因此，缩减空间方法的设计重点不是“怎么解 $n_c\times n_c$”，而是“怎么少算几次 $Jv$”。

### 三种构造 Jacobian 的方式

**方式 A：全空间有限差分。**
对每个约束方向 $e_i$，做一次完整的 $q_{\rm full}(\mu+h e_i)$ 求值，差分得到 $J$ 的第 $i$ 列。优点是简单、无需线性响应代码；缺点是每列一次完整 SCF，$n_c$ 大时不可接受。它适合作为**验证基准**，不适合生产。

**方式 B：线性化 SCF（推荐）。**
利用

$$
Jv=Q_\rho(I-T_\rho)^{-1}T_\mu v,
$$

分三步：

1. 计算“裸”响应 $b=T_\mu v$：在冻结 $H$ 下，把约束算符 $A_i$ 的扰动加进去，做一次线性响应/子空间求解；
2. 解响应方程 $(I-T_\rho)y=b$：这是一个**线性化 SCF**（非精确 Newton 的 forcing term 可自适应控制线性求解精度 [8,9]），$T_\rho$ 就是 SCF 迭代矩阵；
3. 读出 $Jv=Q_\rho y$。

关键观察：**第 2 步和普通 SCF 用的是同一个算子。** 因此可以复用现有的 Kerker 预条件、Anderson/Broyden 历史，甚至直接把 SCF 混合器当作线性求解器（对线性问题，AA 等价于 GMRES [10,11]）。这意味着“算响应”不需要一套全新的基础设施，而是把内层 SCF 的求解器**复用**到线性化问题上。

**方式 C：Broyden 更新。**
维护 $J$ 或 $J^{-1}$ 的低秩更新：

$$
J_{k+1}=J_k+\frac{(y_k-J_k s_k)s_k^{T}}{s_k^{T}s_k},
\qquad
s_k=\mu_{k+1}-\mu_k,\quad
y_k=r_{k+1}-r_k,
$$

其中 $r=q_{\rm full}(\mu)-t$。每步只需一次全空间 $q$ 求值，不需要响应求解；每隔 $K$ 步用方式 B 重新刷新一次 $J$，防止旧信息污染。对昂贵问题，这是最实用的折中。

### 三种方式的代价对照

设一次完整 SCF 的成本为 $C_{\rm SCF}$，一次线性化 SCF（解 $(I-T_\rho)y=b$）的成本为 $C_{\rm lin}\approx \kappa C_{\rm SCF}$，其中 $\kappa$ 是线性求解的迭代数（通常 5–30）。则：

| 方式 | 构造 $J$ 的成本 | 每次外层迭代的成本 | 适用 |
|---|---|---|---|
| A 全空间差分 | $n_c C_{\rm SCF}$ | $n_c C_{\rm SCF}$ | 验证、$n_c\le3$ |
| B 线性化 SCF | $n_c C_{\rm lin}$ | $n_c C_{\rm lin}$ | $n_c$ 小–中，追求精确 |
| C Broyden | 0 | $C_{\rm SCF}$ | $n_c$ 大、响应昂贵 |
| B+C 混合 | 每 $K$ 步 $n_c C_{\rm lin}$ | 多数步 $C_{\rm SCF}$ | 生产默认 |

如果 $n_c=10$、$\kappa=10$，则一次完整 $J$ 约等于 100 次“Hamiltonian 构建”，而一次 SCF 约 30 次；也就是说，一次 $J$ 约等于 3 次 SCF 的成本。用 3 次 SCF 换掉几十次盲目的外层 secant 迭代，通常是划算的。这就是“缩减空间方法”在经济上的核心论据。

### Gauss–Newton 与 Levenberg–Marquardt

外层残差是 $r(\mu)=q_{\rm full}(\mu)-t$。若 $J$ 方阵且满秩，Newton 步是

$$
\delta\mu=-J^{-1}r.
$$

但 $J$ 可能：

- 秩亏（规范零空间、近线性相关约束）；
- 非方（$n_c$ 与约束数不一致，或有冗余约束）；
- 在折叠附近奇异或变号；
- 尺度差异大（不同约束的量纲/量级不同）。

因此更稳健的是 **Gauss–Newton/Levenberg–Marquardt**：

$$
\left(J^{T}J+\lambda\,\mathrm{diag}(J^{T}J)\right)\delta\mu=-J^{T}r,
$$

其中 $\lambda\ge0$ 是阻尼参数 [12]。$\lambda\to0$ 退化为 Gauss–Newton；$\lambda$ 大时退化为最速下降。$\lambda$ 按增益比更新：

$$
\varrho_k=\frac{\|r(\mu_k)\|^2-\|r(\mu_k+\delta\mu_k)\|^2}
{\|r(\mu_k)\|^2-\|r(\mu_k)+J_k\delta\mu_k\|^2}.
$$

$\varrho_k$ 接近 1：模型可信，接受步、减小 $\lambda$；$\varrho_k$ 太小：模型不可信，拒绝步、增大 $\lambda$。这套逻辑与 trust region 等价 [13]，但实现更简单，且对秩亏/非方问题天然稳健。

**LM 的局限**：在折叠点，$J$ 奇异，LM 会给出最小范数步，能停在折叠附近但不能“越过”折叠；要跟踪分支，仍需 continuation（第 10 章）。

### 截断阶数：从零阶到精确

Neumann 级数给出一个自然的“模型层级”：

$$
J_K=Q_\rho\left(\sum_{k=0}^{K}T_\rho^k\right)T_\mu,
\qquad
J_0=Q_\rho T_\mu,\quad
J_\infty=J.
$$

- $K=0$：固定密度模型，零阶截断；
- $K=1$：包含一次密度反馈；
- $K$ 增大：逐步逼近精确 Jacobian；
- $K=\infty$：精确 Schur 补。

这为“多保真”提供了一个**连续的保真度旋钮**，而不是只有“冻结/精确”两档。实际实现中，$K=1$ 或 $2$ 可能已经足够好，因为 $T_\rho$ 的谱半径在多数绝缘体中远小于 1；在金属/近简并体系中才需要更大的 $K$ 或精确解。这个观察可以直接用于设计“分层 Jacobian”：先用 $J_0$ 产生方向，再用 $J_1$ 修正，最后用精确 $J$ 验收。

### 数值细节

- **尺度化**：把 $\mu$ 无量纲化，使各约束的灵敏度同量级；否则 $J$ 的条件数会被量纲污染。
- **有限差分步长**：$h\sim\sqrt{\epsilon_{\rm mach}}\max(1,|\mu_i|)$；对噪声较大的 $q$，用中心差分或更大步长。
- **秩亏处理**：对 $J$ 做 SVD，截断小于 $\tau\sigma_{\max}$ 的奇异值；用伪逆求最小范数步。
- **规范投影**：在求逆前做 $J\leftarrow PJP$，$P=I-\mathbf 1\mathbf 1^{T}/n_c$；或固定一个参考乘子。
- **非方情形**：用 $J^{T}J$ 的 LM 形式，或 QR 分解求最小二乘。
- **线搜索/信赖域**：步长用 $\|\delta\mu\|_\infty$ 或 $\|\delta\mu\|_2$ 限制；merit 用全空间残差 $\|r\|^2$。

---

## 全耦合 KKT 与鞍点预条件：什么时候不要嵌套

### KKT 系统

把密度和乘子放在一起，线性化系统是

$$
\underbrace{
\begin{pmatrix}
I-T_\rho & -T_\mu\\
Q_\rho & 0
\end{pmatrix}
}_{\mathcal K}
\begin{pmatrix}
\delta\rho\\ \delta\mu
\end{pmatrix}
=-
\begin{pmatrix}
f_\rho\\ f_q
\end{pmatrix}.
$$

这是一个**块鞍点系统** [14]：$(1,1)$ 块是 SCF 响应算子，$(1,2)$ 与 $(2,1)$ 是耦合，$(2,2)=0$。它的结构与 PDE 约束优化中的 KKT 系统完全相同 [7,14]。

### 块预条件

直接求解 $\mathcal K$ 需要好的预条件。常用块三角预条件：

$$
\mathcal P=
\begin{pmatrix}
\widehat A & -T_\mu\\
0 & \widehat S
\end{pmatrix},
\qquad
\widehat A\approx I-T_\rho,
\quad
\widehat S\approx -Q_\rho\widehat A^{-1}T_\mu=-J.
$$

应用 $\mathcal P^{-1}$ 时：

1. 解 $\widehat A\,u=b_1$：这是一次**线性化 SCF**，可用 Kerker/AA 预条件；
2. 解 $\widehat S\,v=b_2-u$：这是 $n_c\times n_c$ 的稠密求解；
3. 回代得到 $\delta\mu$，再解 $\widehat A$ 得到 $\delta\rho$。

另一种是**约束预条件**：

$$
\mathcal P_c=
\begin{pmatrix}
\widehat A & -T_\mu\\
Q_\rho & 0
\end{pmatrix},
$$

它在每一步都保持约束线性化，适合等式约束；但要求 $\widehat A$ 的近似足够好。

### 什么时候 monolithic 优于嵌套

嵌套（reduced-space）把内层 SCF 收敛到很高精度，再更新 $\mu$。它的优点是模块化、复用现有 SCF；缺点是：

- 内层误差会污染外层 Jacobian；
- 当 $T_\rho$ 谱半径接近 1 时，内层收敛慢，外层要等很久；
- 强耦合时，内外层交替可能收敛很慢（块 Gauss–Seidel 的谱半径接近 1）。

Monolithic 同时更新 $\rho$ 和 $\mu$，理论上是“一次 Newton 解决所有变量”，对强耦合更稳；但它需要：

- 一个好的块预条件（否则 Krylov 迭代数爆炸）；
- 处理 $T_\rho$ 的非对称性（用 GMRES，而不是 MINRES）；
- 更大的实现复杂度。

**经验判断**：若内层 SCF 在 10–20 次迭代内收敛，嵌套更简单、更便宜；若内层 SCF 经常上百次、或 $n_c$ 很大，monolithic 的块预条件可能更划算。Fe$_2$ 的 82 次 vs 13 次提示：块 Gauss–Seidel 的性能高度依赖内层质量与耦合强度，不能仅凭“嵌套/全耦合”二分。

### 与 Schur 补的等价性

Monolithic 与 reduced-space 在数学上等价 [15]：对 $\mathcal K$ 做块消元就得到 $J$。差别在**数值路径**：

- reduced-space 显式形成 $J$，用稠密线性代数；
- monolithic 不形成 $J$，用 Krylov + 块预条件；
- 两者可以混合：用 reduced-space 的 $J$ 构造 $\widehat S$，再用 monolithic 的 Krylov 解 $\widehat A$。

这种“混合”往往是最好的：$n_c$ 小的时候显式 $J$ 很便宜；$\widehat A$ 的求解复用 SCF 预条件。

---

## 约束处理：SQP、增广拉格朗日、罚函数与 filter

### SQP 与缩减 SQP

序列二次规划（SQP）[16] 在每步解一个带线性化约束的二次规划：

$$
\min_{\delta\rho,\delta\mu}\ 
\frac12
\begin{pmatrix}\delta\rho\\ \delta\mu\end{pmatrix}^{T}
\mathcal H
\begin{pmatrix}\delta\rho\\ \delta\mu\end{pmatrix}
+\nabla E^{T}
\begin{pmatrix}\delta\rho\\ \delta\mu\end{pmatrix}
\quad\text{s.t.}\quad
Q_\rho\delta\rho=-(q-t),
$$

其中 $\mathcal H$ 是拉格朗日 Hessian 或其拟牛顿近似。对等式约束，SQP 与 Newton 的 KKT 系统结构相同；差别在于 $\mathcal H$ 可以用 BFGS/SR1 更新，而不必精确计算。

**缩减 SQP（rSQP）** 把变量分成“决策变量” $\mu$ 与“状态变量” $\rho$，用状态方程消去 $\rho$，得到一个只含 $\mu$ 的二次规划；其 Hessian 是**缩减 Hessian**，可以用拟牛顿更新 [17]。这与本文的 Schur 补方法是同一件事的优化语言版本：

- 状态方程 $\rho-T(\rho,\mu)=0$ 消去 $\delta\rho$；
- 缩减 Hessian 对应 $J^{T}J$ 或拉格朗日 Hessian 的 Schur 补；
- rSQP 的 merit function/trust region 对应本文的全空间残差/信赖域。

rSQP 的价值在于：它给出了一套成熟的**全局化与收敛理论**（merit function、信赖域、二阶校正），可以移植到 allocation 外层。

### 增广拉格朗日（ALM）

ALM [18,19] 把约束加入罚项，并同时更新乘子：

$$
\mathcal L_k(\rho,\mu)
=E_{\rm KS}[\rho]-\mu^{T}(q-t)
+\frac{k}{2}\|q-t\|^2,
$$

$$
\mu\leftarrow\mu-k(q-t).
$$

对本文问题，ALM 的作用有三个：

1. **正则化零空间**：罚项给 KKT 系统加上 $kQ_\rho^{T}Q_\rho$，使 $J$ 在规范方向上有正曲率，缓解 $J\mathbf 1=0$ 的奇异；
2. **处理秩亏/近线性相关约束**：即使 $J$ 不满秩，ALM 的子问题仍有唯一解；
3. **全局化**：ALM 的增广拉格朗日函数可以作为 merit function，配合线搜索/信赖域。

代价是：$k$ 需要逐步增大（或自适应），$k$ 太大时系统病态；乘子的收敛可能慢于原始残差。对“约束必须严格满足”的场景，ALM 通常作为**过渡**，最终仍需把 $q-t$ 压到阈值以下。

### 罚函数、内点与 filter

- **二次罚**：$E+\frac{k}{2}\|q-t\|^2$，实现简单，但 $k\to\infty$ 才能精确满足约束，且条件数随 $k$ 恶化；适合作为初值阶段。
- **精确罚（$L_1$）**：$E+k\|q-t\|_1$，有限 $k$ 即可精确满足，但在 $q=t$ 处非光滑；适合作为兜底。
- **内点/active-set**：如果约束是不等式（例如磁矩上下界），内点法或 active-set 更自然；等式约束下内点法退化为 Newton/ALM 的变体。
- **filter 方法**：把“约束违反”和“目标函数”分开处理，不构造单一 merit function；适合目标函数噪声大、或约束优先级高于能量的场景 [20]。

### 冗余约束与 LICQ 失效

因为 $\sum_iA_i=I$，约束梯度线性相关，LICQ（线性无关约束规范）不成立。标准 SQP 的收敛理论通常要求 LICQ [16]；失效的后果是：

- KKT 矩阵奇异，乘子不唯一；
- SQP 的二次规划可能无界或多解；
- 拟牛顿更新可能沿零空间漂移。

补救方式（按推荐顺序）：

1. **去掉一个冗余约束**：若 $\sum_iq_i=Q_{\rm tot}$ 且 $\sum_it_i=Q_{\rm tot}$，则第 $n_c$ 个约束是前 $n_c-1$ 个的推论；只保留 $n_c-1$ 个独立约束。
2. **固定参考乘子**：令 $\mu_{n_c}=0$，把其余乘子作为独立变量；这等价于在商空间求解。
3. **零空间投影**：用 $P=I-\mathbf 1\mathbf 1^{T}/n_c$ 投影 $J$ 和更新量。
4. **ALM/正则化**：用罚项或 Tikhonov 给零空间加曲率；乘子仍不唯一，但原始解稳定。

这四条不是互斥的；生产实现通常同时做 1 和 3。


---

## 非线性预条件与非线性消除：固定密度模型的正确位置

### 固定密度模型不是 Jacobian，而是非线性预条件器

上传文档的核心判断是：$J_0=Q_\rho T_\mu$ 缺少密度响应，不能作为外层 Jacobian。把这句话翻译成优化语言，可以得到一个更积极的结论：**固定密度子问题是一个合法的非线性预条件器。**

在非线性预条件（nonlinear preconditioning）框架中，求解 $F(x)=0$ 不直接对 $F$ 做 Newton，而是先构造一个“容易解的非线性子问题” $M(x)=0$，再对预条件后的残差做 Newton。ASPIN（additive Schwarz preconditioned inexact Newton）[21] 是这一类方法的代表：它把区域分解的子域求解当作非线性预条件，外层用 Newton 加速。其收敛性不要求预条件器精确，只要求它“well-defined”且预条件后的残差光滑。

对应到 allocation：

- “容易解的子问题” = 冻结 $H[\rho_k]$ 的约束子问题，变量是波函数系数/子空间，约束读数是 $q_{\rm sub}$；
- 外层变量 = $\mu$（以及必要时 $\rho$）；
- 固定密度子问题把“占据-空带混合”的快速非线性消除掉，留下“密度-约束耦合”的慢非线性给外层。

这正是 **非线性消除（nonlinear elimination）** 的结构：把变量分成两组，先精确解一组（波函数/子空间），再对剩下的组（$\mu$）做 Newton。区别只在于“精确”是相对的：固定 $H$ 的子问题可以解到很高精度，但它不是完整的状态方程。

### 为什么这比“把固定密度当 Jacobian”更合理

两者的差别在于**角色的分离**：

| 用法 | 数学身份 | 后果 |
|---|---|---|
| 把 $J_0$ 当 Jacobian | 近似导数 | 步长方向有偏；外层残差与模型不一致；trust 半径失真 |
| 把固定 $H$ 子问题当非线性预条件 | 变量变换/子问题求解 | 只改变迭代路径，不改变收敛目标；外层仍用全空间残差 |

非线性预条件的理论保证是：只要预条件后的映射 $G=M\circ F$ 在解附近光滑且非奇异，Newton 就能收敛；预条件器只需要“把困难的非线性集中到容易解的子问题里”，不需要等于 Jacobian。这正好允许固定密度子问题存在模型误差。

### ASPIN/ASPEN 的可用性

ASPIN 的典型形式是：

1. 对当前 $x$，做一次（或几次）非线性子域求解，得到预条件后的残差 $\tilde F(x)$；
2. 用 Newton/Krylov 解 $\tilde F(x)=0$；
3. 子域求解可以是近似的，但要在每次外层迭代中更新。

对应到 allocation，子域求解就是“冻结 $H$ 下的约束子问题”，外层是 $\mu$ 的 Newton。与 DeltaSpin 的块 Gauss–Seidel 相比，ASPIN 的关键改进是：**不是简单地把子问题的输出当作下一步，而是把子问题的残差拿去做 Newton。** 这通常能显著减少外层迭代数，尤其是在强耦合时。

代价与风险：

- 预条件后的残差需要额外的全空间评估（stage-2），否则无法定义；
- 如果子问题有多个解（磁支），预条件映射是多值的；必须先做分支选择；
- 实现比“块 Gauss–Seidel”复杂，需要把子问题求解器包装成可调用的非线性预条件器。

### 什么时候值得上非线性预条件

- 内层 SCF 是瓶颈（几十到上百次迭代），且外层约束更新被内层拖着走；
- 约束与密度强耦合（$T_\rho$ 谱半径接近 1），块 Gauss–Seidel 收敛慢；
- 约束数较多，显式 Schur 补的 $n_c^2$ 成本开始显著。

如果内层 SCF 本来就在 10 次内收敛，非线性预条件的收益有限，不如直接用缩减空间 Newton。

---

## 多保真、space mapping 与信赖域模型管理

### 两个保真度：子空间读数与全空间读数

上传文档已经把两个模型分得很清楚：

- **低保真**：冻结 $H[\rho_k]$，在子空间内求 $q_{\rm sub}(\mu)$；便宜，但缺少密度响应；
- **高保真**：每次 trial 后做 stage-2 全空间更新，得到 $q_{\rm full}(\mu)$；昂贵，但正确。

多保真优化的任务 [22] 不是“让低保真变精确”，而是**用低保真产生候选、用高保真验收**，并保证在低保真不可信时自动收缩搜索范围。

### 信赖域模型管理（TRMM）

Alexandrov、Dennis、Lewis 与 Torczon [23] 给出了一个严格框架：在信赖域内，用低保真模型 $m_k$ 预测下降，用高保真函数 $f$ 计算实际下降，比值

$$
\varrho_k=
\frac{f(x_k)-f(x_k+s_k)}
{m_k(x_k)-m_k(x_k+s_k)}
$$

决定接受/拒绝与半径更新。收敛性要求低保真模型在信赖域内是“fully linear”的：模型值与梯度的误差被 $O(\Delta_k)$ 控制。满足这个条件时，即使模型有偏，算法仍能收敛到高保真问题的驻点。

对应到 allocation，**merit function 应取全空间残差**：

$$
f(\mu)=\frac12\|q_{\rm full}(\mu)-t\|^2,
\qquad
m_k(\mu)=\frac12\|q_{\rm corr}(\mu)-t\|^2,
$$

其中 $q_{\rm corr}$ 是低保真模型加偏置校正。预测下降用 $m_k$，实际下降用 $f$；两者不能混。

### 输出空间映射：一个便宜的偏置校正

最简单的校正是一阶输出 space mapping [24]：

$$
q_{\rm corr}(\mu)=q_{\rm sub}(\mu)
+\left[q_{\rm full}(\mu_k)-q_{\rm sub}(\mu_k)\right].
$$

它在当前点严格匹配高保真值：$q_{\rm corr}(\mu_k)=q_{\rm full}(\mu_k)$。对下一步的预测，它把低保真的“形状”与高保真的“偏置”结合起来，通常比直接用 $q_{\rm sub}$ 好得多。

但**只校正值不够**：如果低保真的 Jacobian 也有偏，预测步方向仍会错。要做一阶一致的校正，需要同时校正 Jacobian：

$$
J_{\rm corr}(\mu_k)=J_{\rm sub}(\mu_k)
+\left[J_{\rm full}(\mu_k)-J_{\rm sub}(\mu_k)\right],
$$

其中 $J_{\rm full}$ 可以用一次响应求解或 Broyden 更新得到。实际操作中，通常不显式算 $J_{\rm full}$，而是用全空间残差的 Broyden 更新来逐步修正 $J_{\rm corr}$。这就是“多保真拟牛顿”：低保真提供初始 Hessian/Jacobian，高保真提供函数值与更新 [25,26]。

### 多保真拟牛顿

Bryson 与 Rumpfkeil 的多保真拟牛顿方法把低保真梯度用于构造/更新拟牛顿矩阵，把高保真函数值用于校正。对应到本文：

- 用 $J_0$（或 $J_1$）初始化 $J$；
- 每次全空间 $q_{\rm full}$ 求值后，用 Broyden 更新 $J$；
- 每隔 $K$ 步用一次精确响应刷新 $J$；
- trust 半径由全空间残差的实际下降决定。

这比“每步都算精确 $J$”便宜，比“只用 $J_0$”准确，是生产实现最现实的折中。

### 角色分离的三条规则

1. **低保真只产生方向**：$q_{\rm sub}$、$J_0$、$J_1$ 可以决定 trial 步，但不能决定接受/拒绝。
2. **高保真只做验收**：$q_{\rm full}$ 决定残差、trust 半径、收敛标志。
3. **历史不混用**：$q_{\rm sub}$ 与 $q_{\rm full}$ 的残差不能放进同一个 Broyden/AA 历史；否则历史向量不在同一空间，割线条件被污染。

这三条规则听起来像“工程纪律”，实际上有数学依据：多保真收敛理论要求模型误差在信赖域内可控；把两个模型的残差混在一起，等于让模型误差进入割线条件，破坏了“fully linear”假设。

---

## Bilevel、MPEC 与隐式微分：零阶展开

### 标准 bilevel 的超梯度

标准 bilevel 问题 [6]：

$$
\min_{\mu}\ \Phi(\mu,\rho^\star(\mu))
\quad\text{s.t.}\quad
\rho^\star(\mu)=\arg\min_{\rho}\ E(\rho;\mu).
$$

若内层强凸，隐式微分给出

$$
\frac{d\rho^\star}{d\mu}
=-\left(\frac{\partial^2E}{\partial\rho^2}\right)^{-1}
\frac{\partial^2E}{\partial\rho\partial\mu}.
$$

超梯度为

$$
\frac{d\Phi}{d\mu}
=\frac{\partial\Phi}{\partial\mu}
+\frac{\partial\Phi}{\partial\rho}\frac{d\rho^\star}{d\mu}.
$$

这与本文的 Schur 补结构完全相同：把内层 Hessian 换成 $I-T_\rho$，把交叉导数换成 $T_\mu$。

### 平衡约束：不是 argmin，而是不动点

本文的内层是 $\rho^\star=T(\rho^\star;\mu)$，不是某个内层目标的 argmin。因此：

- 内层可能有多解（磁支），$\rho^\star(\mu)$ 是多值映射；
- 内层的“Hessian”不是能量 Hessian，而是 $I-T_\rho$；
- 超梯度在分支点不存在或依赖路径。

这属于 **equilibrium-constrained optimization / MPEC** [5] 的连续特例。标准 bilevel 的“唯一内层解”假设不成立，必须额外做分支选择（第 10 章）。这是本文与机器学习 bilevel 文献最重要的区别，不能直接把“超梯度下降”套过来。

### 截断展开：固定密度近似的精确身份

把 $d\rho^\star/d\mu=(I-T_\rho)^{-1}T_\mu$ 展开：

$$
\frac{d\rho^\star}{d\mu}
=\left(I+T_\rho+T_\rho^2+\cdots\right)T_\mu.
$$

- 取 $K=0$：$d\rho^\star/d\mu\approx T_\mu$，对应 $J_0=Q_\rho T_\mu$；
- 取 $K$ 项：对应 $J_K$；
- 取无穷项：精确 $J$。

因此，**固定密度近似就是“one-step/zero-order unrolling”**。机器学习里的“truncated unrolling”与这里的“截断阶数”是同一件事；误差是 $O(T_\rho^{K+1})$。当 $T_\rho$ 的谱半径接近 1（金属、近简并、磁支），零阶截断的偏差会被放大——这与上传文档的 $1/(1-\alpha)$ 分析完全一致。

### 隐式微分的伴随系统

对可微 DFT，若损失函数是 $\mathcal L(\rho,\mu)$，反向传播需要解伴随方程

$$
\mathcal K^{T}
\begin{pmatrix}
\xi_\rho\\ \xi_\mu
\end{pmatrix}
=
\begin{pmatrix}
\partial\mathcal L/\partial\rho\\ \partial\mathcal L/\partial\mu
\end{pmatrix},
$$

其中 $\mathcal K$ 就是第 5 章的 KKT 矩阵。这与深度平衡模型/可微优化层的隐式微分同构 [27,28]。也就是说，**正向的 Schur/KKT 结构可以直接复用到反向**：同一个块预条件、同一个响应求解器、同一个规范投影。这比“展开所有 SCF 迭代再反传”省显存，也比“只做一步 unrolling”准确。

### 多解与不可微

在磁支翻转、Verwey 转变、f 电子组态转变附近，$\rho^\star(\mu)$ 不连续；隐式微分失效。可选处理：

- 在分支内做隐式微分，跨分支时标记梯度不可信；
- 用熵正则/展宽平滑转变；
- 用 continuation 跟踪分支，把“分支标签”作为额外状态；
- 对损失函数做路径平均/期望，而不是点值。

这不是纯算法问题，而是物理上“相变附近梯度无定义”的体现；任何可微 DFT 实现都必须面对。


---

## 分支、折叠与多解：continuation 与 deflation

### 为什么约束响应会折叠

$q(\mu)$ 不是单调函数。随着 $\mu$ 改变：

- 本征值移动，占据数改变，费米面移动；
- 金属中可能出现能带穿越/费米面重构；
- 近简并处本征矢响应 $\delta c_n\sim 1/(\epsilon_n-\epsilon_m)$ 发散；
- 磁支之间可能翻转（Fe$_2$ 的偏磁通道）。

这些效应会让 $q(\mu)$ 出现平台、拐点、跳变，甚至多个根。对单一 secant/线性插值来说，这意味着：

- 在平台区，$\partial q/\partial\mu\approx0$，Newton 步会爆炸；
- 在折叠点，$J$ 奇异，步长方向不确定；
- 在跳变处，$q$ 不连续，任何基于差商的 Jacobian 都失效。

### 伪弧长延拓

把目标 $t$ 或外场 $\lambda$ 作为延拓参数，把 $\mu$ 和 $\lambda$ 都当作曲线上的变量。伪弧长（pseudo-arclength）[29] 的增广系统是

$$
\begin{pmatrix}
r(\mu,\lambda)\\
\dot\mu^{T}(\mu-\mu_{\rm prev})
+\dot\lambda\,(\lambda-\lambda_{\rm prev})-\Delta s
\end{pmatrix}=0,
$$

其中 $(\dot\mu,\dot\lambda)$ 是切向量，$\Delta s$ 是弧长步长。它的优点是：

- 可以越过折叠点（$J$ 奇异处），因为增广系统仍然满秩；
- 可以跟踪“回头”的分支，而不是在 $\mu$ 的一维投影上丢解；
- 可以用切向量判断分支方向与稳定性变化。

对 Fe$_2$，建议把目标磁矩 $t$ 作为延拓参数：从容易收敛的小磁矩逐步加到目标值，保存每个点的 $\mu$、$\rho$、总能和磁支标签。这样即使中间出现折叠，也能跟踪同一分支，而不是跳支。

### Deflation：系统地找其他根

Deflation [30] 在已知根 $\mu_1$ 处引入极点：

$$
M(\mu;\mu_1)=\frac{1}{\|\mu-\mu_1\|^p}+\sigma,
\qquad
M(\mu;\mu_1)\,r(\mu)=0.
$$

从新的初猜出发，Newton 型迭代不会回到 $\mu_1$，从而找到其他根。结合延拓就是 **deflated continuation**：在分支的每个点做 deflation，寻找分岔出去的新分支。

在 allocation 中实现 deflation 的关键：

1. **在 $\mu$ 空间还是 $(\rho,\mu)$ 空间？** 推荐在 $\mu$ 空间（$n_c$ 小），用 $r(\mu)=q_{\rm full}(\mu)-t$；若在 $(\rho,\mu)$ 空间，距离要同时包含密度差，维度大且规范依赖。
2. **对称等价解**：全局自旋旋转、平移、点群操作会产生等价解；deflation 会把它们当作新解。应先在商空间（模掉对称群）中定义距离，或按对称性分类。
3. **求解器**：deflation 破坏不动点的压缩性，外层应改用 Newton/JFNK/LM，而不是普通 SCF。
4. **成本**：每个根一次完整 SCF；根的数量可能组合爆炸，需要预算与优先级。

### 分支选择与物理判据

找到多个根后，不能只看“哪个收敛得快”。建议：

- **能量**：比较总能；最低的未必是基态（可能是约束态/鞍点）。
- **稳定性**：检查约束 Hessian/响应矩阵的本征值；负本征值意味着鞍点。
- **物理量**：磁矩、占据数、能隙、态密度是否符合化学直觉。
- **参数连续性**：解是否随 $t$、$U$、展宽平滑变化；跳变可能意味着分支切换或数值伪影。

Fe$_2$ 的例子说明：固定密度预测与全空间读数差 $0.145\,\mu_B$，远大于 `constraint_thr=1e-4`；这不是“精度不够”，而是“模型错了”。分支跟踪的目标不是把这个差值压小，而是确保在正确的分支上收敛。

---

## 零空间、秩亏与约束预处理

### 规范零空间的来源

因为 $\sum_iA_i=I$，对所有乘子同时加常数 $c$：

$$
\mu_i\to\mu_i+c
\quad\Longrightarrow\quad
H\to H+cI.
$$

整体常数平移不改变本征矢与密度，因此 $q$ 不变。于是

$$
J\mathbf 1=0.
$$

这不是数值误差，而是精确的零空间。后果：

- $J$ 不可逆，Newton 步不唯一；
- KKT 矩阵奇异，直接求解会失败或给出巨大乘子；
- Broyden 更新可能沿零空间漂移，污染历史；
- trust region 会把步长浪费在无物理意义的公共分量上。

### 四种处理方式

**（1）去掉冗余约束。**
若 $\sum_iq_i=Q_{\rm tot}$ 且 $\sum_it_i=Q_{\rm tot}$，第 $n_c$ 个约束是前 $n_c-1$ 个的推论。只保留 $n_c-1$ 个独立约束，零空间自动消失。这是最干净的做法，但要求实现能识别全局求和关系。

**（2）固定参考乘子。**
令 $\mu_{n_c}=0$，其余 $n_c-1$ 个作为独立变量。等价于在商空间 $\mathbb R^{n_c}/\mathrm{span}(\mathbf 1)$ 中求解。实现简单，但参考约束的选择会影响数值条件。

**（3）零空间投影。**
定义

$$
P=I-\frac{\mathbf 1\mathbf 1^{T}}{\mathbf 1^{T}\mathbf 1},
\qquad
J\leftarrow PJP,
\qquad
\delta\mu\leftarrow P\,\delta\mu.
$$

投影后的 $J$ 在 $\mathrm{span}(\mathbf 1)$ 上为零，其余方向不变。对 Broyden 更新，也要投影 $s_k$ 与 $y_k$，否则零空间分量会累积。

**（4）正则化/ALM。**
用 $J+\epsilon I$ 或 ALM 的 $kQ_\rho^{T}Q_\rho$ 给零空间加曲率；乘子仍不唯一，但原始解稳定。适合与其他方法组合，不适合单独使用（会改变步长方向）。

### 秩亏与近线性相关

除了精确的规范零空间，$J$ 还可能因为以下原因接近奇异：

- 多个目标投影 $A_i$ 近线性相关；
- 某些 $A_i$ 在给定态上的响应很弱（$Q_\rho T_\mu$ 很小）；
- 金属/近简并处 $T_\rho$ 的本征值接近 1，$(I-T_\rho)^{-1}$ 放大噪声；
- 约束数大于独立约束数。

处理方式：

- **SVD 截断**：$J=U\Sigma V^{T}$，只保留 $\sigma_i>\tau\sigma_{\max}$ 的方向；用伪逆求最小范数步。
- **Tikhonov**：$(J^{T}J+\alpha I)\delta\mu=-J^{T}r$；$\alpha$ 按残差/步长自适应。
- **QR 最小二乘**：对非方/秩亏问题，用 QR 或 $LDL^{T}$ 求解最小二乘，而不是正规方程（避免条件数平方）。
- **约束预处理**：用块对角/块三角预条件改善 KKT 系统的条件数；约束预条件子可以保持约束可行。

### 数值检查

1. 构造 $\mu+c\mathbf 1$，验证 $H$ 只差常数平移、$q$ 不变；检查 $J$ 的最小奇异值接近 0。
2. 对 $J$ 做 SVD，记录条件数与零空间维数；若零空间维数大于 1，说明还有额外的线性相关约束。
3. 在投影前后分别做一次 Newton 步，比较步长与残差；若投影后步长显著更合理，说明零空间确实在消耗步数。
4. 对 Broyden 历史做投影，检查 $J$ 的零空间是否被污染。

---

## 黑箱、无导数与随机化：备用工具箱

### 什么时候退化为黑箱

如果暂时没有线性化 SCF/响应求解器，只能把 $r(\mu)=q_{\rm full}(\mu)-t$ 当作黑箱，那么可用的算法是：

- **Broyden** [31,32]：低秩更新 $J$，每步一次 $q$ 求值；实现简单，适合 $n_c$ 小。
- **JFNK** [33]：用有限差分近似 $Jv$，GMRES 解 Newton 方程；每次 $Jv$ 需要一次 $q$ 求值（或一次线性化 SCF）；比 Broyden 更精确，但更贵。
- **Levenberg–Marquardt**：最小化 $\|r\|^2$，对秩亏/非方/非单调稳健；每步需要 $J$ 或其 $Jv$。
- **无导数信赖域（DFO）**：COBYLA [34]、基于插值的 DFO [35]；每步用模型拟合 $q(\mu)$，适合 $n_c\le10$ 且求值极贵的情况。

### DFO 的适用边界

DFO 的优点是“不需要导数”，缺点是：

- 每次迭代需要 $O(n_c)$ 或 $O(n_c^2)$ 次求值来建模型；
- 对 $n_c>20$ 通常不划算；
- 对非光滑/多解问题，模型拟合会被跳变污染。

因此，DFO 只适合作为“没有响应求解器时的临时方案”，或者作为“最后 1–2 个约束的抛光”。一旦能算 $Jv$，就应该用 LM/Newton。

### 随机化与低秩

当 $n_c$ 很大（例如逐原子磁矩约束，$n_c\sim10^2$–$10^3$），显式 $J$ 的 $n_c^2$ 存储与 $n_c$ 次响应求解都不划算。此时可以用：

- **随机化草图**：用随机矩阵 $\Omega$ 探测 $J$ 的低秩结构，$J\approx Q Q^{T}J$；用 $r\ll n_c$ 次响应求解近似 $J$ [36]。
- **低秩 + 对角**：$J\approx J_{\rm diag}+U V^{T}$；对角部分来自局域响应，低秩部分来自长程耦合。
- **矩阵自由 GMRES**：不显式形成 $J$，每次 $Jv$ 一次响应求解；Krylov 迭代数由 $J$ 的谱决定。
- **随机化子空间**：对 $\mu$ 空间做随机投影，减少 trial 方向；适合高维但有效秩低的约束。

随机化的代价是概率性保证：需要 oversampling 与幂迭代；对生产计算，要设置足够的采样数并做残差校验。

### 联合 Anderson/DIIS：低风险的第一步

把 $x=(\rho,\mu)$、$G(x)=(T(\rho,\mu),q(\rho))$ 放在一起做 AA/DIIS [10,37]：

$$
x_{k+1}=\sum_i\alpha_iG(x_{k-m+i}),
\qquad
\sum_i\alpha_i=1.
$$

优点：不需要 Jacobian、不需要响应求解、实现最简单；缺点：

- 残差尺度不一致：$\rho$ 的残差维度大，$q$ 的残差维度小；直接做最小二乘会被 $\rho$ 主导，约束被忽略。必须加权：
  $$
  \|r\|_W^2=\|r_\rho\|^2/\sigma_\rho^2+\|r_q\|^2/\sigma_q^2.
  $$
- 固定密度 $q$ 不能进入历史；必须用全空间 $q$。
- AA 对非单调/折叠没有全局保证；需要 safeguard/trust region。

因此，联合 AA 适合作为 Phase 1 的“低风险起点”，不适合作为最终求解器。

### 块 Gauss–Seidel 与 ADMM

DeltaSpin 的内循环本质是块 Gauss–Seidel：

$$
\mu^{k+1}=\mathcal S(\rho^k),
\qquad
\rho^{k+1}=T(\rho^k,\mu^{k+1}).
$$

它的收敛性由复合映射的谱半径决定；当 $T_\rho$ 接近 1 时，谱半径接近 1，收敛慢。改进方式：

- 用 Anderson/Broyden 加速块 Gauss–Seidel（联合或分块）；
- 用 ADMM/分裂把约束块与密度块解耦，加对偶变量；对非凸问题有收敛性条件 [38]；
- 用非线性预条件（ASPIN）替代简单交替。

ADMM 的优势是模块化（每块用专用求解器），劣势是罚参数调参与非凸收敛保证弱。对 allocation，ADMM 更适合“约束块 + 密度块”的分解，而不是替代 Schur 补。


---

## 与之前两篇调研的衔接：保留什么、降级什么、新增什么

前两篇调研分别覆盖了“工程收敛”和“一般优化算法”。本篇的视角更窄，但更贴近 allocation 的结构。三者的关系可以概括为：

- 第一篇回答“SCF 怎么收敛”；
- 第二篇回答“数学/计算机界有哪些优化算法”；
- 本篇回答“在 allocation 约束这个具体结构上，哪些算法值得用、怎么组合”。

### 保留

| 方法 | 在本问题中的新角色 |
|---|---|
| Anderson/DIIS | 内层 SCF；联合 $(\rho,\mu)$ 的低风险外层；线性化 SCF 的响应求解器 |
| Broyden | 外层 $J$ 的低秩更新；减少响应求解次数 |
| JFNK/Newton–Krylov | 计算 $Jv$、解响应方程、全耦合 KKT |
| 信赖域/LM | 外层全局化；处理秩亏与非单调 |
| Deflation/continuation | 磁支/多解；Fe$_2$ 必需 |
| 多保真/space mapping | 固定密度模型的正确定位；预测—校正 |
| 隐式微分/DEQ | 可微 DFT 的伴随；精确 Schur |
| 随机化/低秩 | 大 $n_c$ 时的 $J$ 压缩 |

### 降级

| 方法 | 为什么降级 |
|---|---|
| 流形优化 | 波函数规范/占据-空带混合的几何语言有用，但不是约束外层的主求解器；本问题的约束是密度泛函，不是正交约束 |
| SDP/1-RDM | 可用于约束子问题/强关联基准，但规模与成本不适合生产级 allocation |
| 学习型初猜/优化器 | 可降低内层迭代，但对多解/强关联不可靠；只能作为可选加速 |
| 纯 DIIS 无 safeguard | 对非光滑/非单调外层没有保证；必须配 trust region/LM |
| 固定密度子空间作为求解器 | 只能作为预条件器/预测器；不能作为 Jacobian/收敛判据 |

### 新增

| 方法 | 解决的问题 |
|---|---|
| 缩减空间 Newton/GN/LM | F1+F2：小约束 + 精确响应 |
| 全耦合 KKT/鞍点预条件 | F3+F6：零空间 + 强耦合 |
| SQP/缩减 SQP | 约束优化的全局化与收敛理论 |
| ALM/罚函数 | 秩亏/零空间的正则化 |
| 非线性预条件/ASPIN | 内层 SCF 瓶颈；固定密度模型的正确角色 |
| MPEC/平衡约束 | 多解、非唯一内层、超梯度 |
| 零空间/约束预条件 | 单位分解导致的 LICQ 失效 |
| DFO | 没有响应求解器时的备用 |

### 一个关键的观念转变

第一篇的视角是“**让不动点迭代更快**”；第二篇的视角是“**用优化算法替代/加速不动点**”；本篇的视角是“**把约束平衡问题拆成状态响应 + 约束根 + 分支跟踪，三层各用各的算法**”。

![推荐的三层求解器与反馈回路](figures/fig2_solver_loop.png)

这个转变的直接后果是：不再追求“一个更好的混合器”，而是追求“一个正确的求解器栈”。混合器仍然重要，但它只负责最内层；外层的正确性由 Schur 补、全空间残差与分支跟踪保证。

---

## 推荐路线图：分阶段、带决策门

### Phase 0：诊断先于实现

在写任何新算法之前，先把上传文档第 9 节的五项检查 [1] 跑通，并补充三项算法级诊断：

| 诊断 | 目的 | 通过标准 |
|---|---|---|
| D1 酉旋转不变量 | 验证 $A_i$ 代数与占据子空间 | $q_i$ 变化 $<10^{-12}$ |
| D2 $J_0$ vs $J$ 有限差分 | 量化缺失响应 | 记录 $\|J-J_0\|/\|J\|$ |
| D3 stage-2 残差 | 验证全空间验收 | 每步记录 $q_{\rm full}-t$ |
| D4 规范零空间 | 验证 $J\mathbf 1=0$ | 最小奇异值 $<10^{-10}$ |
| D5 分支连续性 | 验证多根/折叠 | 扫描 $t$ 记录根与能量 |
| D6 响应求解成本 | 评估 $(I-T_\rho)^{-1}$ 的 Krylov 迭代数 | 记录每次 $Jv$ 的迭代数/时间 |
| D7 $J$ 的条件数 | 评估秩亏/尺度 | 记录 $\sigma_{\max}/\sigma_{\min}$ |
| D8 trust 比值一致性 | 验证预测/实际同模型 | 比值落在 $[0.1,10]$ 之外的比例 |

D2 和 D6 是决定“是否值得上缩减空间方法”的关键：如果 $\|J-J_0\|/\|J\|$ 很小（强屏蔽/绝缘体），固定密度预测可能已经够好；如果很大（金属/磁支），必须补响应。

### Phase 1：低风险起点（联合 AA + 全空间验收）

目标：不改动内层 SCF，只把外层从“固定密度判据”改成“全空间判据”，并加规范投影。

- 变量 $x=(\rho,\mu)$，残差加权：$\|r\|_W^2=\|r_\rho\|^2/\sigma_\rho^2+\|r_q\|^2/\sigma_q^2$；
- 用 AA/DIIS 或 Broyden 更新 $x$；
- 每步做 stage-2 全空间更新，记录 $q_{\rm full}$；
- 对 $\mu$ 做零空间投影；
- 加简单的 trust region/safeguard：残差上升时回滚、减小步长。

预期：迭代数可能仍高于 DeltaSpin，但会消除“子空间收敛、全空间不收敛”的假象。Phase 1 的价值是建立**可信的 baseline**，而不是追求速度。

### Phase 2：核心（缩减空间 Gauss–Newton/LM）

目标：把 $J$ 算对，把外层迭代数降下来。

- 对每个约束方向，计算 $b=T_\mu e_i$（固定 $H$ 的线性响应）；
- 解 $(I-T_\rho)y=b$（GMRES/AA，复用 SCF 预条件）；
- 组装 $J$，做规范投影 $J\leftarrow PJP$；
- 用 LM/trust region 求 $\delta\mu$；
- 每 $K$ 步用精确响应刷新 $J$，其余步用 Broyden 更新；
- 用 $q_{\rm full}$ 做唯一验收。

预期：如果 $n_c$ 小、$J$ 条件数可控，外层迭代数可降低 3–10 倍；每次迭代的成本增加 $n_c$ 次响应求解。净收益取决于 $n_c$、Krylov 迭代数与 baseline 外层迭代数的乘积。

### Phase 3：多保真与分支（生产化）

- 用 $J_0/J_1$ 作为初猜/预条件，减少精确响应次数；
- 用 output space mapping 校正 $q_{\rm sub}$，提高预测质量；
- 用 trust-region 模型管理保证“低保真只提议、高保真验收”；
- 对 Fe$_2$ 等磁支体系，加入 continuation（目标 $t$ 或外场 $\lambda$）；
- 对多解，加入 deflation 或分支切换。

### Phase 4：备选（全耦合/ASPIN）

只有当 Phase 2/3 仍受内层 SCF 瓶颈限制时才启动：

- 全耦合 KKT + 块三角预条件（$\widehat A$ 用 SCF 预条件，$\widehat S$ 用 $J$）；
- 或 ASPIN：把固定 $H$ 子问题包装成非线性预条件器，外层 Newton；
- 对 $n_c$ 很大，用矩阵自由 GMRES + 随机化/低秩。

### 决策门

| 门 | 条件 | 决策 |
|---|---|---|
| G1 | $\|J-J_0\|/\|J\|<0.1$ | 固定密度预测可能够用；Phase 2 可延后 |
| G2 | 每次 $Jv$ 的 Krylov 迭代 $>50$ | 响应太贵；优先 monolithic/ASPIN 或 Broyden |
| G3 | $n_c>50$ | 显式 $J$ 不划算；转矩阵自由 + 随机化 |
| G4 | $\det J$ 变号/多根 | 加 continuation/deflation |
| G5 | Phase 2 两轮仍无法接近 DeltaSpin | 按上传文档的结论，把 allocation 限制为只读诊断 |

### 伪代码

```text
输入: rho_0, mu_0, target t, tolerance tol
for k = 0, 1, 2, ...:
    # 1. 内层/全空间状态
    rho_k = SCF(rho_{k-1}, mu_k)          # 可用 AA/Broyden
    q_full = Q(rho_k)
    r = q_full - t
    if ||r||_W < tol: break

    # 2. 响应与 Jacobian
    if need_exact_J(k):
        for i in 1..n_c:
            b_i = T_mu(e_i)                # 固定 H 的线性响应
            y_i = solve(I - T_rho, b_i)    # GMRES/AA + SCF 预条件
            J[:, i] = Q_rho(y_i)
    else:
        J = broyden_update(J, dmu_prev, dr_prev)
    J = P @ J @ P                          # 规范投影

    # 3. 外层步
    dmu = LM_or_Newton(J, r, lambda)
    dmu = clip_to_trust_region(dmu, Delta)
    mu_trial = mu_k + dmu

    # 4. 全空间 trial
    rho_trial = SCF(rho_k, mu_trial)
    q_trial = Q(rho_trial)
    r_trial = q_trial - t

    # 5. 接受/拒绝（全空间 merit）
    ratio = (||r||^2 - ||r_trial||^2) / predicted_reduction(J, dmu, r)
    if ratio > eta:
        accept; mu_k = mu_trial; rho_k = rho_trial
        if ratio > eta_good: Delta *= 2; lambda *= 0.5
    else:
        reject; Delta *= 0.5; lambda *= 2

    # 6. 分支保护
    if det(J) changed sign or ||dmu|| stagnates:
        switch_to_continuation()
```

注意：第 2 步的 `solve(I - T_rho, b_i)` 与第 4 步的 `SCF` 可以共享线性化/预条件，但**不能共享残差历史**：一个是线性响应，一个是非线性 SCF。

---

## 验证与基准计划

### 单元测试（数学正确性）

1. **$J$ 的有限差分校验**：对一个小体系，用全空间差分逐列验证 $Jv=Q_\rho(I-T_\rho)^{-1}T_\mu v$；误差应随步长 $h$ 按 $O(h)$ 或 $O(h^2)$ 收敛。
2. **$J_0$ 与 $J$ 的差异**：在同一参考态，比较 $J_0$ 与 $J$；用一维标量模型验证 $J_0=\beta$、$J=\beta/(1-\alpha)$，并检查 $\alpha\to1$ 时的放大。
3. **规范零空间**：验证 $J\mathbf 1=0$；检查投影前后最小奇异值；验证 $\mu+c\mathbf 1$ 不改变 $q$。
4. **截断阶数**：比较 $J_0,J_1,J_2,J$ 的谱与预测步长；验证 $J_K\to J$ 的收敛阶。
5. **伴随一致性**：对可微版本，比较隐式微分梯度与有限差分梯度；误差应随 SCF 阈值收紧而下降。

### 算法 A/B（性能）

在相同几何、相同约束、相同 `constraint_thr` 下比较：

| 算法 | 变量 | 预期 |
|---|---|---|
| Baseline（当前 fixed-density subspace） | $\mu$ | 复现 82 次迭代/失败模式 |
| Native DeltaSpin | $\mu$ | 复现 13 次迭代 |
| Phase 1：联合 AA + 全空间验收 | $(\rho,\mu)$ | 迭代数下降，但可能仍高于 DeltaSpin |
| Phase 2：缩减 Newton/LM | $\mu$ | 目标 ≤2× DeltaSpin，且全空间残差达标 |
| Phase 2b：Broyden + 精确刷新 | $\mu$ | 减少响应求解，迭代数略增 |
| Phase 3：TRMM + output space mapping | $\mu$ | 预测更准，trial 拒绝率下降 |
| Phase 4：monolithic/ASPIN | $(\rho,\mu)$ | 内层瓶颈时更优 |

指标：全空间残差、外层迭代数、内层 SCF 迭代数、$Jv$ 次数、wall time、显存、失败率。

### 消融实验

- 有/无规范投影；
- $J_0$ vs $J_1$ vs 精确 $J$；
- Newton vs LM vs Broyden；
- 有/无 trust region；
- 有/无 output space mapping；
- 有/无 continuation；
- 不同 $n_c$（1, 2, 5, 10, 50）。

### 合成测试

- **一维标量模型**：$q(\mu)=\beta\mu/(1-\alpha)$，验证 $J_0/J=1-\alpha$ 与步长放大。
- **二维折叠模型**：$q_1=\mu_1^2+\mu_2$、$q_2=\mu_1+\mu_2^2$，制造多根与折叠，验证 LM/continuation。
- **规范模型**：$q_i=\mu_i-\mu_{i+1}$，$\sum q_i=0$，验证零空间投影。

### 真实体系

- **Fe$_2$-bcc**：磁支翻转；比较 fixed-density、DeltaSpin、Phase 2、Phase 3。
- **LiCl**：`mu=0` 回归；验证零空间与平凡约束。
- **NiO/Fe$_3$O$_4$**（若可用）：多解 + DFT+U；验证 deflation/continuation 的必要性。

### 验收标准

- 全空间残差 $\|q_{\rm full}-t\|_\infty<10^{-4}$（与 `constraint_thr` 一致）；
- 不出现“子空间收敛、全空间不收敛”的步；
- 外层迭代数 ≤2× DeltaSpin（Phase 2 目标），或 wall time 更短；
- 多初猜/多分支下结果可复现，分支标签明确；
- 梯度与有限差分一致（可微版本）。

---

## 开放问题

1. **$Jv$ 的真实成本**：$(I-T_\rho)^{-1}$ 的 Krylov 迭代数由 $T_\rho$ 的谱决定；在金属/磁支附近可能很大。能否用低秩/预条件把每次 $Jv$ 降到接近一次 SCF？
2. **非光滑 $q(\mu)$**：占据跳变、磁支翻转使 $J$ 不存在或不连续；LM/continuation 能处理，但缺少严格的收敛理论。
3. **规范零空间与拟牛顿**：Broyden 更新在零空间上的漂移如何严格投影？投影后的割线条件是否仍然成立？
4. **Merit function 的选择**：全空间残差 $\|q-t\|^2$ 与能量+约束的增广拉格朗日，哪个更适合 trust region？在能量噪声大时，残差 merit 是否足够？
5. **预测—校正的收敛证明**：当低保真模型有偏时，TRMM 的“fully linear”条件能否在 $q_{\rm sub}$ 上验证？output space mapping 能否提供可证明的误差界？
6. **多解与可微性**：在相变/分支点，梯度不存在；如何定义物理上有意义、数值上稳定的“广义梯度”？
7. **大规模 $n_c$**：逐原子约束使 $n_c$ 达到 $10^2$–$10^3$；随机化/低秩能否保持 $J$ 的有效秩？如何选采样数？
8. **与内层 SCF 的协同**：能否让内层 SCF 的 AA 历史直接为响应求解提供 Krylov 子空间，从而“一次迭代、两用”？

---

## 附录 A 关键公式 {-}

**A.1 耦合残差与 Schur 补**

$$
F(\rho,\mu)=
\begin{pmatrix}\rho-T(\rho,\mu)\\ q(\rho)-t\end{pmatrix},
\qquad
J=Q_\rho(I-T_\rho)^{-1}T_\mu.
$$

**A.2 固定密度近似**

$$
J_0=Q_\rho T_\mu,
\qquad
J-J_0=Q_\rho(I-T_\rho)^{-1}T_\rho T_\mu.
$$

**A.3 Neumann 级数与截断**

$$
J=Q_\rho\left(\sum_{k=0}^{\infty}T_\rho^k\right)T_\mu,
\qquad
J_K=Q_\rho\left(\sum_{k=0}^{K}T_\rho^k\right)T_\mu.
$$

**A.4 Levenberg–Marquardt**

$$
\left(J^{T}J+\lambda\,\mathrm{diag}(J^{T}J)\right)\delta\mu=-J^{T}r.
$$

**A.5 KKT 系统与块预条件**

$$
\mathcal K=
\begin{pmatrix}
I-T_\rho & -T_\mu\\
Q_\rho & 0
\end{pmatrix},
\qquad
\mathcal P=
\begin{pmatrix}
\widehat A & -T_\mu\\
0 & \widehat S
\end{pmatrix},
\quad
\widehat S\approx-J.
$$

**A.6 增广拉格朗日**

$$
\mathcal L_k=E-\mu^{T}(q-t)+\frac{k}{2}\|q-t\|^2,
\qquad
\mu\leftarrow\mu-k(q-t).
$$

**A.7 信赖域模型管理比值**

$$
\varrho_k=
\frac{f(x_k)-f(x_k+s_k)}
{m_k(x_k)-m_k(x_k+s_k)}.
$$

**A.8 输出空间映射**

$$
q_{\rm corr}(\mu)=q_{\rm sub}(\mu)
+\left[q_{\rm full}(\mu_k)-q_{\rm sub}(\mu_k)\right].
$$

**A.9 伪弧长延拓**

$$
\begin{pmatrix}
r(\mu,\lambda)\\
\dot\mu^{T}(\mu-\mu_{\rm prev})
+\dot\lambda(\lambda-\lambda_{\rm prev})-\Delta s
\end{pmatrix}=0.
$$

**A.10 零空间投影**

$$
P=I-\frac{\mathbf 1\mathbf 1^{T}}{\mathbf 1^{T}\mathbf 1},
\qquad
J\leftarrow PJP,
\qquad
\delta\mu\leftarrow P\delta\mu.
$$

**A.11 隐式微分**

$$
\frac{d\rho^\star}{d\mu}=(I-T_\rho)^{-1}T_\mu,
\qquad
\frac{d\mathcal L}{d\mu}
=\frac{\partial\mathcal L}{\partial\mu}
+\frac{\partial\mathcal L}{\partial\rho}(I-T_\rho)^{-1}T_\mu.
$$

---

## 附录 B 术语表 {-}

| 术语 | 含义 |
|---|---|
| F1–F6 | 本文归纳的六个结构特征（小约束、响应、零空间、折叠、黑箱、强耦合） |
| Schur 补 | 消去状态变量后得到的约束方向有效 Jacobian |
| $J_0$ | 固定密度/冻结 H 的近似 Jacobian |
| $J_K$ | Neumann 级数截断到 $K$ 阶的 Jacobian |
| 缩减空间 | 消去 $\rho$，只在 $\mu$ 空间求解 |
| 全耦合/monolithic | 同时求解 $\rho$ 与 $\mu$ 的 KKT 系统 |
| 块预条件 | 利用 KKT 块结构构造的预条件 |
| ASPIN | additive Schwarz preconditioned inexact Newton |
| 非线性消除 | 先精确解一组变量，再对剩余变量做 Newton |
| TRMM | trust-region model management，多保真模型管理 |
| output space mapping | 用高保真值校正低保真输出的偏置 |
| MPEC | mathematical program with equilibrium constraints |
| 伪弧长 | 把参数与变量都当作曲线变量的延拓方法 |
| deflation | 通过极点排除已知解、寻找新根 |
| LICQ | 线性无关约束规范；单位分解使其失效 |
| LM | Levenberg–Marquardt |


---

# 参考文献 {-}

1. 本项目内部文档, "Allocation 约束的不动点数学分析," 2026-10-28, 分支 `feat/unified-local-moment-allocation`.
2. X. Liu, X. Wang, Z. Wen, and Y.-X. Yuan, "On the convergence of the self-consistent field iteration in Kohn–Sham density functional theory," *SIAM J. Matrix Anal. Appl.* **35**, 546 (2014).
3. Z. Bai and D. Lu, "Variational characterization of monotone nonlinear eigenvector problems and geometry of self-consistent field iteration," *SIAM J. Matrix Anal. Appl.* **45** (2024).
4. S. Güttel and F. Tisseur, "The nonlinear eigenvalue problem," *Acta Numerica* **26**, 1 (2017).
5. Z.-Q. Luo, J.-S. Pang, and D. Ralph, *Mathematical Programs with Equilibrium Constraints* (Cambridge University Press, 1996).
6. B. Colson, P. Marcotte, and G. Savard, "An overview of bilevel optimization," *Ann. Oper. Res.* **153**, 235 (2007).
7. G. Biros and O. Ghattas, "Parallel Lagrange–Newton–Krylov–Schur methods for PDE-constrained optimization. Part I: The Krylov–Schur solver," *SIAM J. Sci. Comput.* **27**, 687 (2005).
8. R. S. Dembo, S. C. Eisenstat, and T. Steihaug, "Inexact Newton methods," *SIAM J. Numer. Anal.* **19**, 400 (1982).
9. S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an inexact Newton method," *SIAM J. Sci. Comput.* **17**, 16 (1996).
10. H. F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations," *SIAM J. Numer. Anal.* **49**, 1715 (2011).
11. H. Fang and Y. Saad, "Two classes of multisecant methods for nonlinear acceleration," *Numer. Linear Algebra Appl.* **16**, 197 (2009).
12. J. J. Moré, "The Levenberg–Marquardt algorithm: Implementation and theory," *Lecture Notes in Mathematics* **630**, 105 (1978).
13. A. R. Conn, N. I. M. Gould, and P. L. Toint, *Trust-Region Methods* (SIAM, 2000).
14. M. Benzi, G. H. Golub, and J. Liesen, "Numerical solution of saddle point problems," *Acta Numerica* **14**, 1 (2005).
15. J. Hicken and J. Alonso, "Comparison of reduced- and full-space algorithms for PDE-constrained optimization," *AIAA* 2013-1043 (2013).
16. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. (Springer, 2006).
17. L. T. Biegler, J. Nocedal, and C. Schmid, "A reduced Hessian method for large-scale constrained optimization," *SIAM J. Optim.* **5**, 314 (1995).
18. D. P. Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods* (Academic Press, 1982).
19. E. G. Birgin and J. M. Martínez, *Practical Augmented Lagrangian Methods for Constrained Optimization* (SIAM, 2014).
20. R. Fletcher and S. Leyffer, "Nonlinear programming without a penalty function," *Math. Program.* **91**, 239 (2002).
21. X.-C. Cai and D. E. Keyes, "Nonlinearly preconditioned inexact Newton algorithms," *SIAM J. Sci. Comput.* **24**, 183 (2002).
22. M. G. Fernández-Godino, C. Park, N. H. Kim, and R. T. Haftka, "Review of multi-fidelity models," *arXiv*:1609.07196 (2016).
23. N. M. Alexandrov, J. E. Dennis Jr., R. M. Lewis, and V. Torczon, "A trust-region framework for managing the use of approximation models in optimization," *Struct. Optim.* **15**, 16 (1998).
24. J. W. Bandler, Q. S. Cheng, S. A. Dakroury, *et al.*, "Space mapping: The state of the art," *IEEE Trans. Microwave Theory Tech.* **52**, 337 (2004).
25. D. E. Bryson and M. P. Rumpfkeil, "Multifidelity quasi-Newton method for design optimization," *AIAA J.* **56** (2018).
26. A. March and K. Willcox, "Provably convergent multifidelity optimization algorithm not requiring high-fidelity derivatives," *AIAA J.* **50** (2012).
27. S. Bai, J. Z. Kolter, and V. Koltun, "Deep equilibrium models," *Adv. Neural Inf. Process. Syst.* **32** (2019).
28. B. Amos and J. Z. Kolter, "OptNet: Differentiable optimization as a layer in neural networks," *Proc. ICML* (2017).
29. H. B. Keller, "Numerical solution of bifurcation and nonlinear eigenvalue problems," in *Applications of Bifurcation Theory* (Academic Press, 1977), p. 359.
30. P. E. Farrell, Á. Birkisson, and S. W. Funke, "Deflation techniques for finding distinct solutions of nonlinear partial differential equations," *SIAM J. Sci. Comput.* **37**, A2026 (2015).
31. D. D. Johnson, "Modified Broyden's method for accelerating convergence in self-consistent calculations," *Phys. Rev. B* **38**, 12807 (1988).
32. V. Eyert, "A comparative study on methods for convergence acceleration of iterative vector sequences," *J. Comput. Phys.* **124**, 271 (1996).
33. D. A. Knoll and D. E. Keyes, "Jacobian-free Newton–Krylov methods: A survey of approaches and applications," *J. Comput. Phys.* **193**, 357 (2004).
34. M. J. D. Powell, "A direct search optimization method that models the objective and constraint functions by linear interpolation," in *Advances in Optimization and Numerical Analysis* (Kluwer, 1994), p. 51.
35. C. Audet and W. Hare, *Derivative-Free and Blackbox Optimization* (Springer, 2017).
36. N. Halko, P.-G. Martinsson, and J. A. Tropp, "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions," *SIAM Rev.* **53**, 217 (2011).
37. J. Zhang, B. O'Donoghue, and S. Boyd, "Globally convergent type-I Anderson acceleration for nonsmooth fixed-point iterations," *SIAM J. Optim.* **30** (2020).
38. Y. Su, Y. Hu, X. Liu, and G. Hu, "A damped subspace splitting algorithm for constrained density functional theory," *arXiv*:2608.05682 (2026).
