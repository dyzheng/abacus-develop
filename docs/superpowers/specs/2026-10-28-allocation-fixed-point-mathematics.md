# Allocation 约束的不动点数学分析

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 范围：PW allocation（\(A_i=Q_i+Q_c w_iQ_c\)）的数学结构、固定密度近似、
  DeltaSpin 内循环的关系，以及相邻的优化问题求解范式。
- 关联文档：
  `docs/superpowers/specs/2026-10-21-unified-local-moment-allocation-design.md`、
  `docs/superpowers/specs/2026-10-27-allocation-critical-review.md`、
  `docs/superpowers/specs/2026-10-28-allocation-subspace-warm-start.md`、
  `docs/superpowers/specs/2026-10-28-allocation-exact-subspace.md`。

## 0. 结论摘要

**低层代数没有问题，问题出在“把哪个残差当作不动点残差”。** 对固定的
几何，\(A_i\) 是只依赖目标轨道和空间权重的固定算符，
\(\sum_i A_i=I\) 成立；在给定的密度和波函数上，
\(\langle\psi_m|A_i|\psi_n\rangle\) 与全空间读数可以在机器精度内一致。

真正的约束问题是耦合系统

\[
F(\rho,\mu)=
\begin{pmatrix}
\rho-T(\rho,\mu)\\
q(\rho)-t
\end{pmatrix}=0,
\]

其中 \(T\) 是一次电子 SCF 映射，\(\rho=T(\rho,\mu)\) 是电子自洽条件，
\(q(\rho)=t\) 是局域量约束。消去 \(\rho\) 后，约束方向的有效 Jacobian 是
Schur 补

\[
J=Q_\rho (I-T_\rho)^{-1}T_\mu, \qquad Q_\rho\equiv \partial q/\partial\rho.
\]

固定输入密度 \(\rho_0\)、只求解固定 Hamiltonian 的约束子问题时，得到的是
近似 Jacobian

\[
J_0=Q_\rho T_\mu,
\]

它缺少密度响应项
\(Q_\rho[(I-T_\rho)^{-1}-I]T_\mu\)。因此固定密度模型可以作为初值、预测器
或预条件器，但不能单独作为外层收敛判据。

**只在占据子空间内做酉旋转不会改变 \(Q\)。** 若占据数相同，旋转后的占据
投影算符不变，因而 \(\rho\) 和 \(q_i=\operatorname{Tr}(\rho A_i)\) 都不变。
要改变 \(Q\)，必须改变占据子空间本身（占据态与空态混合，或改变占据数），
这必然改变输出密度。所谓“固定密度只调波函数系数”若指真正固定输出密度，
则不存在可用的约束控制方向；若指固定输入 Hamiltonian \(H[\rho_0]\)，则
输出密度会改变，这已经是 DeltaSpin 的固定势/冻结 H 内循环，而不是严格的
固定密度旋转。

**与这个问题最接近的求解背景**是：约束 Kohn-Sham/constrained DFT、
PDE 约束优化或最优控制、bilevel optimization、昂贵黑箱非线性方程
（Broyden/Newton-Krylov/JFNK）、不动点加速（Anderson/DIIS/Pulay）、
KKT/SQP/增广拉格朗日、Stiefel 流形上的波函数优化、以及
homotopy/continuation 与 multi-fidelity/space mapping。DeltaSpin 的内循环
可以理解为其中一种**块 Gauss-Seidel + 多保真预测—全空间校正**方案。

## 1. 符号、约束与能量口径

为避免不同实现中的符号漂移，本文件采用

\[
\mathcal L(\rho,\mu)
=E_{\rm KS}[\rho]-\sum_i\mu_i\bigl(q_i(\rho)-t_i\bigr),
\]

并约定 \(q_i\) 是实际被约束的观测量（电荷或自旋通道）。于是

\[
\delta\mathcal L/\delta\rho=0
\quad\Longleftrightarrow\quad
\frac{\delta E_{\rm KS}}{\delta\rho}-\sum_i\mu_i\frac{\delta q_i}{\delta\rho}=0.
\]

对应的约束 Hamiltonian 写成

\[
H[\rho;\mu]=H_{\rm KS}[\rho]+\sum_i\mu_i A_i.
\]

这里 \(A_i=\delta q_i/\delta\rho\) 在几何固定时是固定算符。若代码采用
\(E+\lambda(q-t)\) 的形式，则代码中的 \(\lambda\) 与本文 \(\mu\) 相差一个
整体符号；比较能量导数时应以能量对目标的有限差分为准，不能只看符号。

约束 Kohn-Sham 方程是

\[
H[\rho;\mu]\psi_n=\epsilon_n S\psi_n,
\qquad
\rho=\sum_n f_n|\psi_n\rangle\langle\psi_n|,
\qquad
q_i=\operatorname{Tr}(\rho A_i).
\]

\(f_n\) 是占据数（含 k 点权重和自旋通道），\(S\) 是重叠算符；在 PW
实现中通常 \(S=I\)，在 LCAO 中 \(S\neq I\)。本文件保留 \(S\) 只是为了
说明广义本征问题，下面的响应公式不依赖具体基组。

对于 allocation 算符

\[
A_i=Q_i+Q_c w_iQ_c,
\qquad Q_c=I-\sum_jQ_j,
\]

有

\[
\sum_iA_i=\sum_iQ_i+Q_c\left(\sum_iw_i\right)Q_c
=\sum_iQ_i+Q_c=I.
\]

这条单位分解恒等式是“求和一致性”的来源，但它同时带来了下一节的约束
冗余/规范自由度。

## 2. 真正的未知量是 \((\rho,\mu)\)，不是单独的 \(\mu\)

### 2.1 嵌套映射

把一次电子迭代（或在固定 \(\mu\) 下收敛后的 SCF 映射）记作

\[
\rho'=T(\rho,\mu),
\]

它表示：用输入密度 \(\rho\) 构造 \(H[\rho;\mu]\)，求解本征问题，更新
占据数，再由新的波函数构造输出密度 \(\rho'\)。给定 \(\mu\) 时，电子
自洽解 \(\rho^*(\mu)\) 满足

\[
\rho^*(\mu)=T(\rho^*(\mu),\mu).
\]

外层约束要求

\[
r(\mu)\equiv q(\rho^*(\mu))-t=0.
\]

因此这是两层嵌套问题：

- 内层：固定 \(\mu\)，求电子密度不动点 \(\rho=T(\rho,\mu)\)；
- 外层：在电子不动点流形上求约束根 \(q(\rho)=t\)。

把它误写成只依赖 \(\mu\) 的普通标量根问题，虽然形式上是
\(r(\mu)=0\)，但 \(r\) 的每一次求值都隐含一次内层 SCF。外层算法的
收敛速度由内层映射的响应决定，而不是由 \(\mu\) 的一维步长单独决定。

### 2.2 耦合残差与 Schur 补

为了同时描述密度和乘子，定义

\[
F(\rho,\mu)=
\begin{pmatrix}
\rho-T(\rho,\mu)\\
q(\rho)-t
\end{pmatrix}.
\]

在解附近对 \((\rho,\mu)\) 线性化，记
\(T_\rho=\partial T/\partial\rho\)、
\(T_\mu=\partial T/\partial\mu\)、
\(Q_\rho=\partial q/\partial\rho\)，
\(f_\rho=\rho-T(\rho,\mu)\)、
\(f_q=q(\rho)-t\)，得到

\[
\begin{pmatrix}
I-T_\rho & -T_\mu\\
Q_\rho & 0
\end{pmatrix}
\begin{pmatrix}
\delta\rho\\ \delta\mu
\end{pmatrix}
=-
\begin{pmatrix}
f_\rho\\
f_q
\end{pmatrix}.
\]

这是块鞍点/KKT 结构。对密度块做 Schur 消元：

\[
\delta\rho=(I-T_\rho)^{-1}\bigl(-f_\rho+T_\mu\delta\mu\bigr),
\]

代回约束块得到

\[
J\,\delta\mu=-f_q+
Q_\rho(I-T_\rho)^{-1}f_\rho,
\]

其中

\[
J=Q_\rho(I-T_\rho)^{-1}T_\mu.
\]

若已经在内层电子不动点上（\(f_\rho=0\)），Newton 步就是

\[
\delta\mu=-J^{-1}\bigl(q-t\bigr).
\]

\(J\) 的物理含义很清楚：先由约束乘子扰动产生一波“裸”密度响应
\(T_\mu\delta\mu\)，再经过电子自洽反馈
\((I-T_\rho)^{-1}\) 放大/修正，最后由观测算符 \(Q_\rho\) 读出约束变化。

### 2.3 固定输入密度得到的是 \(J_0\)，不是 \(J\)

设当前输入密度是 \(\rho_0\)，只构造
\(H_0=H[\rho_0]\)，在固定 \(H_0\) 下求约束子问题

\[
\rho_1=T(\rho_0,\mu).
\]

在这个“固定密度/冻结 Hamiltonian”模型里，约束读数是

\[
q_0(\mu)=q\bigl(T(\rho_0,\mu)\bigr),
\]

其 Jacobian 是

\[
J_0=Q_\rho T_\mu.
\]

与精确 Jacobian 的差为

\[
J-J_0
=Q_\rho\left[(I-T_\rho)^{-1}-I\right]T_\mu
=Q_\rho(I-T_\rho)^{-1}T_\rho T_\mu.
\]

如果 \(T_\rho\) 的谱半径很小（强屏蔽、绝缘体、电子响应快），这个差可以
很小；如果 \(T_\rho\) 接近单位算符（金属、近不稳定点、近简并、磁支附近），
\((I-T_\rho)^{-1}\) 会放大误差，固定密度模型给出的步长方向甚至可能错误。

一个一维标量模型足以说明量级。令 \(Q_\rho=1\)、\(T_\mu=\beta\)、
\(T_\rho=\alpha\)，则

\[
J=\frac{\beta}{1-\alpha},
\qquad J_0=\beta.
\]

若用 \(J_0\) 代替 \(J\) 解同一个残差，冻结密度步长与精确 Newton 步长的
比值是 \(1/(1-\alpha)\)：当 \(0<\alpha<1\) 时冻结步长过大，当
\(\alpha\to1\) 时它会发散并给出错误的约束灵敏度。
这不是某个实现常数的问题，而是省略了密度反馈这一整个线性响应通道。

## 3. 当前 exact subspace 到底算了什么

当前实验路径在当前能带子空间 \(C=[\psi_1,\ldots,\psi_{n_{\rm band}}]\)
中构造

\[
H_{\rm sub}=C^\dagger H_0C,\qquad
S_{\rm sub}=C^\dagger SC,\qquad
A_{i,\rm sub}=C^\dagger A_iC,
\]

然后对

\[
\left(H_{\rm sub}+\sum_i\mu_iA_{i,\rm sub}\right)u
=\epsilon S_{\rm sub}u
\]

做广义本征分解，并令 \(C\leftarrow Cu\)。这是在固定 \(H_0\) 下的
Rayleigh-Ritz 子空间近似，不是完整的电子自洽响应。它有两个近似：

1. **子空间截断**：只允许 \(C\) 张成的有限维空间内的混合；
2. **密度冻结**：Hamiltonian 使用输入密度 \(\rho_0\)，不包含输出密度
   变化对 \(H_{\rm KS}\) 的反馈。

第一项可以通过扩大子空间、加入空带或做全空间 Davidson 修正来改善；第二项
不能靠“多迭代几次固定密度子问题”消除，因为每次都仍在同一个
\(H[\rho_0]\) 上工作。固定密度内层迭代次数增加，只会更精确地求解
\(q_0(\mu)=t\)，不会自动得到 \(q(\rho^*(\mu))=t\)。

这解释了一个容易误判的现象：子空间代数可以精确到 \(10^{-14}\)，但由它
给出的 \(\mu\) 仍可能在密度更新后产生远大于约束阈值的偏差。前者验证的是
“在给定态上 \(A_{i,\rm sub}\) 与 \(A_i\) 一致”；后者检验的是“投影模型
是否等于自洽约束映射”。两者不是同一个命题。

已有 Fe2 数据正是一个直接例子：固定密度预测已经接近目标时，
stage-2 全空间求解并更新密度后，偏磁通道仍移动约 \(0.145\,\mu_B\)，
远大于 `constraint_thr=1e-4`。该数值只能证明“密度响应项不可忽略”，
不能反过来证明 \(A_i\) 子空间代数错误。

## 4. 为什么“固定密度 + 仅调波函数系数”不能改变约束

### 4.1 占据子空间内的酉旋转是规范变换

设占据带为 \(n=1,\ldots,n_{\rm occ}\)，且这些带的占据数相同（例如
零温绝缘体每个自旋通道的占据带都为 1）。取一个酉矩阵 \(U\)，令

\[
\widetilde\psi_n=\sum_{m=1}^{n_{\rm occ}}\psi_mU_{mn}.
\]

则占据投影算符满足

\[
\widetilde P
=\sum_{n=1}^{n_{\rm occ}}|\widetilde\psi_n\rangle f_n\langle\widetilde\psi_n|
=\sum_{m=1}^{n_{\rm occ}}|\psi_m\rangle f_m\langle\psi_m|
=P.
\]

因此 \(\widetilde\rho=\rho\)，从而对任何固定算符 \(A_i\)，

\[
\widetilde q_i
=\operatorname{Tr}(\widetilde\rho A_i)
=\operatorname{Tr}(\rho A_i)=q_i.
\]

这不是数值精度问题，而是严格恒等式。仅在占据带内部做酉混合、相位旋转或
重新正交化，不能改变任何只依赖密度（或占据投影）的约束读数。

如果占据数在所选子空间内不完全相同，结论要弱化为：只在同一占据数的简并
子块内旋转才保持 \(\rho\)。跨越不同占据数的混合会改变密度。

### 4.2 一阶本征矢响应也说明必须离开占据子空间

固定 \(S\)，对 \(H\) 加微扰 \(\delta H\)。非简并一阶微扰给出（在
\(c_m^\dagger S c_n=\delta_{mn}\) 的归一化下）

\[
\delta c_n
=\sum_{m\ne n}
\frac{c_m^\dagger\delta H c_n}{\epsilon_n-\epsilon_m}c_m .
\]

若 \(n\) 是占据带而 \(m\) 只取占据带，则 \(\delta c_n\) 仍是占据子空间内
的旋转；它不改变投影 \(P\)。只有当求和包含空带（或改变占据数）时，
\(\delta P\) 才可能非零，进而 \(\delta q_i\) 才可能非零。近简并时
\(\epsilon_n-\epsilon_m\) 很小，响应会变大，这也是带交叉附近外层迭代
困难的原因之一。

### 4.3 “固定密度”一词的三种不同含义

必须区分以下三个对象：

1. **输入密度** \(\rho_{\rm in}\)：用于构造 \(H[\rho_{\rm in}]\)；
2. **输出密度** \(\rho_{\rm out}\)：由当前 \(\psi\) 和占据数重新构造；
3. **自洽密度** \(\rho^*\)：满足 \(\rho^*=T(\rho^*,\mu)\) 的密度。

DeltaSpin/当前子空间路径所说的“固定密度内循环”，实际是固定
\(\rho_{\rm in}\)（更准确地说，固定 \(H_0=H[\rho_{\rm in}]\)）并对
波函数做约束子空间求解。此时 \(\rho_{\rm out}\) 一般不等于
\(\rho_{\rm in}\)，所以约束读数可以变化；变化来自占据态与空态混合，
不是来自占据子空间内的酉旋转。

如果把“固定密度”严格解释为 \(\rho_{\rm out}=\rho_{\rm in}\)，则
波函数只剩占据子空间内的规范自由度，\(q_i\) 严格不变，根本不存在
“用系数调整把 \(q_i\) 拉到目标”的控制方向。若强行改变系数而不改变
密度，只能改变规范，不能改变可观测量。

因此可以尝试的是 **固定 Hamiltonian/冻结密度势的约束子问题**，而不是
**固定输出密度的酉旋转**。前者是合法的预测器，后者在数学上不可实现。

## 5. DeltaSpin 内循环的准确数学定位

DeltaSpin 的 PW 流程可以先抽象为：

1. 用当前密度 \(\rho^k\) 做一次全空间求解，得到
   \(H^k=H[\rho^k]\)；
2. 缓存子空间矩阵 \(H_{\rm sub}^k\)、\(S_{\rm sub}^k\) 和投影矩阵；
3. 在冻结的 \(H^k\) 上反复改变 \(\mu\)，用子空间对角化得到近似的
   \(q_{\rm sub}(\mu)\)，用 BFGS/CG 求
   \(q_{\rm sub}(\mu)-t\approx0\)；
4. 做一次 stage-2 全空间更新（或至少用全空间波函数重新计算约束量），
   得到新的 \(\rho^{k+1}\) 和全空间 \(q^{k+1}\)；
5. 用全空间残差决定是否接受、是否继续外层 SCF。

写成映射就是

\[
\mu^{k+1}=\mathcal S(\rho^k),
\qquad
\rho^{k+1}=T(\rho^k,\mu^{k+1}),
\]

其中 \(\mathcal S(\rho^k)\) 是冻结 \(H[\rho^k]\) 下的约束根求解。这是一次
**非线性块 Gauss-Seidel 迭代**：先固定密度块求乘子，再用乘子更新密度块。
它并不声称 \(\mathcal S(\rho^k)\) 已经解决了完整 Schur 系统；它只是提供
一个内层预测，真正的外层收敛由 stage-2 后的全空间读数把关。

这与当前 exact-subspace 失败路径的关键差别是：

- DeltaSpin 内层的 \(q_{\rm sub}\) 是**提议**，最终接受/拒绝使用全空间
  \(q\)；
- 不能让固定密度 \(q_0\) 同时充当内层目标、外层 Jacobian 和收敛判据；
- 当前的 trust region 如果把固定密度预测下降量和全空间实际下降量混在
  一起，就会得到不一致的预测/实际比值，导致半径不断收缩或过冲。

当前 exact-subspace 路径也在最后调用一次 stage-2，因此问题并不是“完全
没有全空间更新”。更准确的差别是：它在一组固定密度 trial 之后才做 stage-2，
而 trial 期间已经把固定密度 \(q_0\) 送进了外层 mu/trust 逻辑；DeltaSpin
则把内层 \(q_{\rm sub}\) 限定为 BFGS 的候选生成器，stage-2 后的全空间
读数才承担外层验收。前者会让同一个量既当模型又当裁判，后者不会。
已有 Fe2 A/B 中 allocation 为 82 次电子迭代、native DeltaSpin 为 13 次；
两者约束算符不同，不能把全部差异归因于这一处结构，但“内层预测值是否被
冒充为外层残差”是首先必须排除的数学原因。

所以“为什么不能采用 DeltaSpin 的固定密度内循环”有两层答案：

- 若真的固定输出密度，只做占据子空间酉旋转，则 \(Q\) 不变，不能成立；
- 若固定输入 Hamiltonian、允许波函数在占据+空带子空间内响应，则可以，
  但必须把它当作预测器，并在每个 trial 后做全空间更新和全空间 \(Q\)
  验收。后者正是 DeltaSpin 内循环的数学本质。

## 6. 约束结构带来的额外困难

### 6.1 单位分解导致规范零方向

因为 \(\sum_iA_i=I\)，对所有约束同时做

\[
\mu_i\longrightarrow\mu_i+c
\]

会给 Hamiltonian 加上 \(cI\)。在固定粒子数和固定自旋通道的电子问题中，
整体常数平移不改变本征矢和密度，因此约束读数不变，但乘子改变了。只要
约束集包含完整的单位分解（例如所有目标分配算符都参与），约束 Jacobian
就存在至少一个零方向：

\[
J\,\mathbf 1=0.
\]

这意味着 \(\mu\) 不是唯一的；必须固定一个参考乘子，或在每一步投影掉
公共分量。若目标值之和也等于总量，约束方程本身还会有一条冗余，独立约束
数最多是 \(n_c-1\)。不处理这个零空间时，线性求解器、Broyden 更新和
trust region 都可能在无物理意义的方向上消耗步数。

### 6.2 约束可能秩亏

多个目标投影可能近线性相关，或者某些 \(A_i\) 在给定态上的响应很弱，
导致 \(J\) 的条件数很大。此时应使用截断 SVD/最小二乘、Tikhonov 正则化
或 ALM 的惩罚项 \(k(q-t)\) 改善条件数，而不是直接把病态矩阵求逆。

### 6.3 \(q(\mu)\) 可能非单调并出现折叠

即使每个 \(A_i\) 固定，\(\mu\) 改变会移动本征值、改变占据数和费米面；
在金属、近简并和磁支附近，\(q(\mu)\) 可以有平台、拐点、跳变甚至多个根。
Fe2 上观察到的磁支翻转就是这种结构。此时单一 secant/线性插值不具备全局
收敛保证，必须结合：

- trust region 或 line search；
- 目标值的 homotopy/continuation；
- 分支标签和上一几何的延续；
- 必要时用 pseudo-arclength 跟踪折叠曲线，而不是只追 \(\mu\) 的一维根。

### 6.4 \(I-T_\rho\) 的条件数决定外层难度

精确 Schur 补含 \((I-T_\rho)^{-1}\)。如果电子密度响应接近临界，最小
奇异值很小，精确约束 Jacobian 会很大或变号，固定密度近似会失真，Newton
步也会很大。这可解释为什么“增加内层次数”有时反而更差：内层越精确地
解一个错误的冻结模型，外层越容易得到一个与真实密度响应不匹配的 \(\mu\)。

## 7. 与哪些优化问题最接近

### 7.1 约束优化、KKT、SQP 与增广拉格朗日

原始问题是

\[
\min_\rho E_{\rm KS}[\rho]
\quad\text{s.t.}\quad q_i(\rho)=t_i.
\]

拉格朗日函数 \(\mathcal L=E-\mu^\top(q-t)\) 的 KKT 条件同时包含
电子驻点条件和约束。对 KKT 系统做 Newton 消元就得到第 2 节的 Schur 补。
这与 **SQP（序列二次规划）** 的结构一致：约束线性化、Lagrangian
Hessian 或拟 Hessian、信赖域/线搜索。**增广拉格朗日/ALM** 把
\(q-t\) 加入罚项，等价于给约束 Jacobian 加上 \(kI\) 型正则化，通常能
改善近奇异和分支附近的稳定性。

### 7.2 PDE 约束优化与最优控制

把 Kohn-Sham 方程看作状态方程，把 \(\mu\) 看作控制量，把 \(q\) 看作
观测/目标，问题是

\[
\min_{\rho,\mu}E(\rho)
\quad\text{s.t.}\quad \text{KS}(\rho,\mu)=0,\quad q(\rho)=t.
\]

这正是 **PDE-constrained optimization / optimal control / parameter
identification**。常用方法包括：

- 全耦合（monolithic）Newton；
- 消除状态后的 reduced-space Newton；
- 伴随（adjoint）方法计算梯度；
- 矩阵自由的 Newton-Krylov。

其核心经验与本问题完全一致：不要用一次冻结状态的预测替代状态响应，
除非它只是预条件器。

### 7.3 Bilevel optimization 与嵌套不动点

外层变量 \(\mu\)、内层变量 \(\rho\) 的结构可以写成

\[
\min_\mu \Phi(\mu)\quad\text{s.t.}\quad
\rho=T(\rho,\mu),\ \ q(\rho)=t.
\]

这属于 **bilevel optimization / equilibrium-constrained optimization**
（平衡约束或带平衡约束的数学规划）。内层 SCF 是下层平衡条件，外层约束
是上层问题；它不是普通的“内层无约束最小化”bilevel，但隐式微分和
Schur 补的结构相同。常见做法是：

- 对内层方程做隐式微分，得到超梯度/Schur 补；
- 用 one-step/truncated unrolling 近似内层响应；
- 用 penalty/ALM 替代严格下层约束。

固定密度预测相当于对内层响应做零阶截断；它能降低计算量，但必须明确它是
有偏的。

### 7.4 昂贵黑箱非线性方程：Broyden、Newton-Krylov、JFNK

把外层写成

\[
r(\mu)=q(\rho^*(\mu))-t=0.
\]

每次求值都要一次或多次 SCF，因此 \(r\) 是昂贵黑箱。适合的工具是：

- **Broyden**：用低秩更新近似 \(J\)，不显式求密度响应；
- **Newton-Krylov/JFNK**：只实现 \(Jv\) 的矩阵自由作用，用 GMRES 解
  Newton 方程；
- **Gauss-Newton/Levenberg-Marquardt**：最小化
  \(\|r\|^2\)，对非方、秩亏或非单调问题更稳健；
- **trust-region**：限制 \(\mu\) 步长，处理大残差和分支跳变。

对 \(Jv\)，可以用一次线性化 SCF 近似

\[
Jv\approx Q_\rho(I-T_\rho)^{-1}T_\mu v,
\]

而不形成 \(T_\rho\) 和 \(T_\mu\) 的全部矩阵。约束数通常远小于密度维度，
这是低秩/矩阵自由方法有吸引力的原因。

### 7.5 不动点加速：Anderson、DIIS、Pulay

电子 SCF 本身就是不动点迭代，常用 Anderson/DIIS/Pulay 混合。约束外层也
可以纳入同一向量：

\[
x=(\rho,\mu),\qquad
G_\eta(x)=
\begin{pmatrix}
T(\rho,\mu)\\
\mu-\eta\,(q(\rho)-t)
\end{pmatrix}.
\]

它的不动点 \(x=G_\eta(x)\) 同时给出 \(\rho=T(\rho,\mu)\) 和
\(q(\rho)=t\)；\(\eta\) 是阻尼步长。不能把 \(F=(T-\rho,q-t)\) 直接当成
Anderson 的 \(G\)：Anderson/DIIS 加速的是 \(x=G(x)\)，而 \(F=0\) 是根残差，
两者只有在显式构造 \(x-\eta F\) 后才等价。对 \(x\) 做 Anderson/DIIS 可以
作为低风险的第一步；它不需要完整 Jacobian，但仍应使用全空间 \(q\) 构造
残差，不能把固定密度读数混入同一历史。

### 7.6 非线性本征值问题与约束 DFT

固定 \(\mu\) 的方程

\[
\left(H_{\rm KS}[\rho]+\sum_i\mu_iA_i\right)\psi
=\epsilon S\psi
\]

是参数化非线性本征值问题（NEPv）：矩阵依赖密度，密度又依赖本征矢。
constrained DFT、\(\Delta\)SCF 和 DeltaSpin 都属于这一族。常用策略是
自洽迭代、子空间本征分解、线性响应和 Newton 修正。这里的 \(A_i\) 是
约束算符，不是普通的外部场；因此要区分约束算符的代数性质与约束映射的
非线性响应。

### 7.7 Stiefel/Grassmann 流形与波函数规范

波函数系数在重正交化下具有规范自由度；占据子空间可以用 **Stiefel 流形**
或 **Grassmann 流形**描述。占据子空间内部的酉旋转是切空间方向，不改变
占据投影，因此不影响只依赖 \(\rho\) 的约束。流形优化的语言有助于：

- 明确哪些方向是规范（应投影掉）；
- 分离“占据—空带混合”与“占据子空间内部旋转”；
- 设计保持正交性的更新和预条件器。

但本问题的约束主要是密度/投影泛函，不能只靠流形上的无约束优化解决；
仍需处理约束乘子和自洽密度响应。

### 7.8 多保真、space mapping 与子空间模型

固定密度 \(A_i\) 子空间是一个**低保真模型**，全空间自洽 \(q\) 是
**高保真模型**。这对应 **multi-fidelity optimization / space mapping**：
低保真模型负责廉价地产生候选 \(\mu\)，高保真模型负责评估和校正。关键是
保持两级模型的角色分离：

- 低保真模型可以决定 trial 方向；
- 高保真模型决定接受/拒绝、trust 半径和最终收敛；
- 两级残差不能混用同一个历史或同一个收敛标志。

这与 DeltaSpin 的“内层子空间 + stage-2 全空间”结构非常接近，也是当前
allocation 需要补上的部分。

### 7.9 连续延拓、分支跟踪与伪弧长

当目标 \(t\) 改变或几何改变导致 \(q(\mu)\) 出现折叠时，普通根跟踪会跳支。
**homotopy/continuation** 把目标从容易解的值逐步移到目标值；
**pseudo-arclength continuation** 把 \(\mu\) 和目标都当作曲线参数，可以
跟踪折叠和转向。对 Fe2 的磁支问题，这比单纯缩小 trust 半径更符合问题
结构，但代价是需要分支状态和更多试算。

### 7.10 耦合物理/区域分解中的块迭代

把密度块和乘子块看作两个耦合物理场，外层迭代就是 **block
Gauss-Seidel** 或 **block Jacobi**；整体 Newton 是 monolithic 方法。
区域分解/ Schwarz 方法中的收敛分析同样关注块间耦合算子的谱半径。
DeltaSpin 的内循环是“先解乘子块，再更新密度块”的块 Gauss-Seidel；
当前 allocation 若把固定密度子问题当作完整解，则相当于把块间耦合丢掉。

## 8. 由数学结构直接推出的算法原则

### 8.1 用全空间残差做唯一验收标准

外层必须使用

\[
r_{\rm full}=q_{\rm full}(\mu)-t
\]

作为收敛、trust 半径和 Jacobian 更新的依据。固定密度读数可以用于：

- 产生初值；
- 预测下一步；
- 作为预条件器/方向；
- 估计局部响应；

但不能作为“已满足约束”的证明。一个实用的预测—校正结构是：

1. 在当前 \(\rho^k\) 上构造冻结 \(H^k\)，用子空间约束求解得到
   \(\mu_{\rm trial}\)；
2. 用 \(\mu_{\rm trial}\) 做一次全空间 stage-2，得到 \(\rho^{k+1}\)；
3. 由 \(\rho^{k+1}\) 计算 \(q_{\rm full}\) 和 \(r_{\rm full}\)；
4. 用 \(r_{\rm full}\) 更新 Broyden/Newton 模型和 trust 半径；
5. 只在 \(\|r_{\rm full}\|<\tau\) 时停止。

这比“固定密度内层收敛后直接宣布约束收敛”多一次全空间求解，但这是
正确性问题，不是可选的精度修饰。

### 8.2 使用 Schur 补或它的低秩近似

若约束数 \(n_c\) 很小，可以近似构造

\[
J\approx Q_\rho(I-\widehat T_\rho)^{-1}\widehat T_\mu,
\]

其中 \(\widehat T_\rho\) 可以来自：

- 少量线性响应/有限差分方向；
- Broyden 的拟 Newton 更新；
- 对角或块对角响应模型；
- ALM 的 \(kI\) 正则化。

若 \(n_c\) 较大，则用矩阵自由的 \(Jv\) 和 GMRES，而不是显式形成
\(n_c\times n_c\) 的完整响应矩阵（除非 \(n_c\) 确实很小）。

### 8.3 处理规范零空间

在每次线性求解前做投影

\[
\mu\leftarrow\mu-\frac{\mathbf 1^\top\mu}{\mathbf 1^\top\mathbf 1}\mathbf 1
\]

（或固定一个参考约束的乘子），并在 Broyden/Newton 的 Jacobian 上投影掉
公共方向。若所有约束的 \(A_i\) 不构成单位分解，则不应盲目套用这个投影，
需要先检查 \(\sum_iA_i\) 是否真的为 \(I\)。

### 8.4 对分支和非单调性使用 continuation

对 Fe2 这类有磁支跳变的体系，建议把目标值或外场作为 continuation 参数，
保存上一分支的 \(\mu\)、\(\rho\) 和约束历史；不要让一个无记忆的 secant
在分支边界自由跳转。trust region 只能限制步长，不能替代分支跟踪。

### 8.5 预存只改变成本，不改变数学模型

缓存 \(A_i\)、\(H_{\rm sub}\)、\(S_{\rm sub}\)、目标乘积或
\(\langle\psi_m|A_i|\psi_n\rangle\) 可以显著降低成本，但它们仍然只给出
冻结模型。它们不能把 \(J_0\) 变成 \(J\)。性能优化和收敛判据必须分开
评价：

- 计算成本：FFT、hPsi、投影、归约、子空间对角化；
- 数学正确性：是否包含密度响应、是否使用全空间 \(q\)、是否正确处理
  零空间和分支。

## 9. 可验证的数学预言与后续测试计划

以下检查不依赖新的物理近似，适合作为后续实现的验收：

1. **酉旋转不变量**：在同一占据子空间内随机生成酉矩阵，旋转前后比较
   \(q_i=\operatorname{Tr}(\rho A_i)\)，应在机器精度内相等
   （目标 \(<10^{-12}\)）。若不等，说明旋转跨越了不同占据数或改变了
   占据子空间。
2. **固定密度与全空间 Jacobian 的差异**：在同一参考态做有限差分，比较
   \(J_0\) 与 \(J\)；用一维/二维玩具模型验证
   \(J_0=\beta\)、\(J=\beta/(1-\alpha)\) 的关系。
3. **stage-2 残差检查**：任何冻结密度预测之后，必须记录全空间
   \(q_{\rm full}-t\)；若该残差大于阈值，不能把该步标记为约束收敛。
4. **规范零空间检查**：若 \(\sum_iA_i=I\)，构造 \(\mu+c\mathbf1\)，
   验证 Hamiltonian 只差常数平移，并检查 Jacobian 的零奇异值。
5. **分支连续性检查**：在 Fe2 的目标附近扫描 \(t\)，记录每个根的
   \(\mu\)、总能和磁支标签；若出现多条根，不能用单次 secant 的结果
   作为全局结论。

## 10. 最终判断

1. **Allocation 的算符代数可以继续保留。** \(A_i\) 的单位分解、正性、
   低秩结构和目标/互补分解没有数学矛盾；求和一致性是真实的优点。
2. **固定密度 exact subspace 不应作为生产约束求解器。** 它精确求解了一个
   不包含密度反馈的替代问题；在 Fe2 上，固定密度与全空间读数的差异已经
   直接超过约束阈值。保留它作为诊断/预条件器是合理的，作为外层收敛判据
   不合理。
3. **DeltaSpin 内循环不是反例，而是提示。** 它的内层同样是冻结
   Hamiltonian 的近似，但它通过 stage-2 全空间更新和最终全空间读数把
   近似限制在“候选生成”层面；若 allocation 复制这一结构，并显式处理
   Schur 补/规范零空间/分支，仍有优化空间。若只复制固定密度子空间而
   不复制全空间校正，则会重复当前的失败模式。
4. **迭代次数高的根源不只是步长。** 主要来自：缺失密度响应导致的错误
   Jacobian、trust 模型混用两种观测、单位分解的零方向、以及 Fe2 的
   非单调磁支。单纯调 `step_max`、`inner_thr` 或增加固定密度子空间
   迭代次数，不能从根本上解决。
5. **工程决策应分层。**
   - 若目标只是 Fe 的局域磁矩约束：当前直接使用 native DeltaSpin；
   - 若目标是保留 allocation 的目标/互补可加性与电荷诊断：继续做
     “冻结 H 预测 + 全空间校正 + 全空间残差验收”的混合求解；
   - 若两轮真实 inner-loop 实现后仍无法接近 DeltaSpin 的迭代量级，
     应把 allocation 限制为只读诊断，停止其生产约束路径。

一句话概括：**\(A_i\) 子空间是好的预条件器，不是好的不动点方程。**
约束的数学核心是密度响应和 Schur 补；任何跳过它的外层迭代都必须诚实地
用全空间残差校正，否则就会在“子空间读数已经收敛”和“真实自洽解尚未收敛”
之间反复摇摆。

## 11. 本轮验证状态

- **Test plan**：见第 9 节的五条可验证预言；本轮只建立数学框架，不新增
  数值扫描。
- **Test setup**：引用已有的 Fe2-bcc（4 MPI、单线程 OpenMP/BLAS/FFT、
  `constraint_thr=1e-4`）和 LiCl `mu=0` 回归数据；未改变输入。
- **Results**：本轮为纯文档轮，没有新的物理测试结果；已有
  \(1.20\times10^{-14}\) 子空间一致性、Fe2 固定密度与全空间读数约
  \(0.145\,\mu_B\) 的差异，以及 82 对 13 次电子迭代，仅作为背景证据引用。
- **Analysis**：这些证据分别支持“低层代数正确”“密度响应不可忽略”和
  “块 Gauss-Seidel/全空间验收是性能与收敛的关键差异”，不能被解释为同一
  命题。
- **Next steps**：先实现第 8.1 节的预测—校正结构并用第 9 节的五项检查
  验收，再决定 allocation 是否进入生产路径。

## 12. 对《Allocation 约束优化算法调研》的再评估

`docs/superpowers/specs/Allocation约束优化算法调研.md` 的总体判断与本文件
一致：真正的对象是平衡约束/缩减空间问题，固定密度子问题只能作为预测器、
预条件器或低保真模型；外层必须用全空间 \(q\) 验收，并处理规范零空间、
非单调分支和响应成本。下面只列出需要补充或修正的地方。

### 12.1 接受的部分

- 把问题定位为 **equilibrium-constrained optimization** 比泛称 bilevel
  更准确；它与 MPEC 的连续、无互补约束特例相近，但不应直接套用标准
  MPEC 的互补约束理论。PDE 约束优化的缩减 Newton 工具箱确实最贴近。
- 用 \((I-T_\rho)y=T_\mu v\) 计算 \(Jv\)、用全空间残差做唯一验收、对
  \(\mu\) 做规范投影，这三条是正确的主线。
- 把固定密度模型看作多保真模型/非线性预条件器的方向是合理的；它的价值
  是降低候选生成成本，而不是替代真实状态响应。
- Phase 0 的 D2（\(J_0\) vs \(J\)）、D6（响应求解成本）、D7（条件数）、
  D8（trust 比值一致性）应优先于继续增加子空间迭代次数。

### 12.2 需要修正或限定的部分

**1. 联合 Anderson 的映射不能直接写成 \((T-\rho,q-t)\)。**

若定义

\[
x=(\rho,\mu),\qquad
F(x)=\begin{pmatrix}\rho-T(\rho,\mu)\\q(\rho)-t\end{pmatrix},
\]

则 \(F=0\) 是根问题，不是 Anderson 的标准不动点形式。调研文档中的
\(G(x)=(T(\rho,\mu),q(\rho))\) 更不对：它的不动点要求
\(\mu=q(\rho)\)，而不是 \(q(\rho)=t\)。正确的联合不动点映射应写成

\[
G_\eta(x)=
\begin{pmatrix}
T(\rho,\mu)\\
\mu-\eta\,(q(\rho)-t)
\end{pmatrix},
\qquad
x=G_\eta(x)\Longleftrightarrow F(x)=0.
\]

Anderson/DIIS 只能加速这个 \(G_\eta\)（或等价的 \(x-\eta F\)）；
不能把根残差历史直接当作不动点历史。调研文档在正文里已有“历史不混用”
的提醒，但公式需要按上式修正。

**2. Neumann 级数只在局部、谱半径小于 1 时成立。**

\[
J=Q_\rho\left(\sum_{k=0}^{\infty}T_\rho^k\right)T_\mu
\]

是对线性化映射 \(T_\rho\) 的形式展开，不是对非线性 SCF 历史的展开。
它要求 \(T_\rho\) 在关注点附近可微且谱半径 \(<1\)。若 SCF 依靠混合器、
Anderson 或 Broyden 才收敛，原始单步映射的谱半径可能大于 1；此时截断
\(J_K\) 只能作为预条件器，不能声称逼近 \(J\)。在占据跳变、磁支翻转或
近简并处，\(q(\mu)\) 可能不可微，\(J\) 本身不存在，Neumann 展开也不成立。
同理，非线性 SCF 的 Anderson/Broyden 历史不能自动当作线性响应方程的
Krylov 子空间；它可以提供初值或预条件，但线性化算子的历史必须单独建立。

**3. 响应成本公式需要统一量纲。**

记一次 Hamiltonian/线性化算子应用的成本为 \(C_H\)，一次完整 SCF 的
迭代数为 \(N_{\rm scf}\)，响应 Krylov 迭代数为 \(\kappa\)，则

\[
C_{\rm SCF}\approx N_{\rm scf}C_H,
\qquad
C_{\rm resp}\approx\kappa C_H,
\qquad
C_J\approx n_c\kappa C_H
=\frac{n_c\kappa}{N_{\rm scf}}C_{\rm SCF}.
\]

调研文档中“一次线性化 SCF 成本约 \(\kappa C_{\rm SCF}\)”与后文
“\(n_c=10,\kappa=10\)、SCF=30 次 Hamiltonian 构建，所以 \(J\) 约 3 次
SCF”使用了不同的 \(C_{\rm SCF}\) 口径。按统一口径，若
\(n_c=10,\kappa=10,N_{\rm scf}=30\)，一次完整 \(J\) 约为
\(100C_H=(100/30)C_{\rm SCF}\approx3.3C_{\rm SCF}\)，不是
\(100C_{\rm SCF}\)。若每个响应方向还要重复构建/正交化，成本还要乘上
相应常数。这个区别直接决定 Phase 2 是否值得做。

**4. “固定 H 子问题 = ASPIN 非线性预条件器”需要额外的良定义条件。**

ASPIN/非线性消除要求子问题给出一个（至少在分支内）良定义、足够光滑的
映射 \(E(\mu)\) 或其残差，并能定义预条件后的方程。当前固定 H 子问题可能
有多个根、依赖初值、在磁支附近不连续，因此它更像**分支内的非线性
平滑器/变量消除候选**，不能仅凭类比就继承 ASPIN 的收敛保证。若要用它，
至少需要：

- 固定分支标签和初值选择；
- 保证子问题可解且解对 \(\mu\) 连续（或显式检测跳变）；
- 定义清楚的预条件残差，而不是把 \(q_{\rm sub}\) 直接喂给外层；
- 用全空间 \(q\) 检查预条件后的残差。

**5. SQP 的 QP 不能漏掉状态方程。**

调研文档中的 SQP 展示式只写了约束
\(Q_\rho\delta\rho=-(q-t)\)，但若变量同时包含 \((\delta\rho,\delta\mu)\)，
还必须线性化状态方程

\[
(I-T_\rho)\delta\rho-T_\mu\delta\mu=-f_\rho.
\]

否则 QP 允许任意改变密度而不保持电子平衡。实际做法应是：
先消去 \(\delta\rho\) 得到缩减 QP，或把状态方程作为附加等式约束；
不能只保留约束行而省略电子状态行。

**6. output space mapping 只校正零阶偏置，不会自动修正 Jacobian。**

\[
q_{\rm corr}(\mu)=q_{\rm sub}(\mu)
+\left[q_{\rm full}(\mu_k)-q_{\rm sub}(\mu_k)\right]
\]

在当前点匹配高保真值，但其导数仍是 \(J_{\rm sub}\)。因此下一步的预测
方向仍可能错误；要一阶一致，必须计算或用 Broyden 更新
\(J_{\rm full}\)。调研文档对此有提醒，但“多保真拟牛顿”的收敛性依赖于
全空间残差的割线历史和 trust-region 模型管理，不能把零阶 bias 当作
完整修正。调研中的
\(J_{\rm corr}=J_{\rm sub}+[J_{\rm full}-J_{\rm sub}]\) 在代数上只是
\(J_{\rm full}\)，它本身不是可执行的一阶校正；必须给出 \(J_{\rm full}\)
的实际响应计算或其 Broyden 近似。

**7. 规范投影是有条件的。**

\(J\mathbf 1=0\) 只在所有参与求和的 \(A_i\) 构成单位分解、且没有额外
显式 \(\mu\) 依赖时成立。若只约束一个原子、一个片段或一个目标壳层，
\(\sum_iA_i\neq I\)，不应盲目做 \(P=I-\mathbf 1\mathbf 1^T/n_c\)。
即使满足零空间条件，也要区分“公共乘子规范”与“目标值求和冗余”：
前者总存在，后者只在 \(\sum_it_i\) 恰好等于总可观测量的情况下出现。

**8. 性能数字是目标，不是已证结果。**

调研文档中的“3–10 倍迭代下降”和“≤2× DeltaSpin”是合理的 benchmark
目标，不是由现有 Fe2 数据推出的结论。现有 82 vs 13 次还受不同约束算符、
不同磁支和不同内层实现影响；在完成统一约束、统一全空间残差和统一分支
选择之前，不能把差异全部归因于 Schur 方法或固定密度近似。

**9. 块三角预条件中的 Schur 符号要统一。**

按本文件的 KKT 约定

\[
\mathcal K=
\begin{pmatrix}
I-T_\rho & -T_\mu\\
Q_\rho & 0
\end{pmatrix},
\]

块消元得到

\[
S=-Q_\rho(I-T_\rho)^{-1}(-T_\mu)
=Q_\rho(I-T_\rho)^{-1}T_\mu=J.
\]

因此块三角预条件中的 \(\widehat S\) 应近似 \(J\)，而不是 \(-J\)。
调研文档里的 \(\widehat S\approx-J\) 与前面的 \(J\) 定义不一致；若实现
使用相反的 KKT 行符号，必须同时翻转右端项和更新方向，不能只改一个块。

### 12.3 修正后的优先级

1. **先做可信 baseline**：全空间 \(q\) 验收 + 规范投影 + Broyden/LM
   保护，不把固定密度 \(q\) 放进 trust/Jacobian 历史（调研 Phase 1）。
2. **再测响应是否值得**：用 D2 比较 \(J_0\) 与 \(J\)，用 D6 记录
   \(Jv\) 的 Krylov 成本，用 D7 检查条件数；只有误差大且响应成本可接受时
   才进入显式缩减 Newton。
3. **固定 H 子问题的角色**：初值、预测器、分支内非线性平滑器或
   预条件器；不作为外层残差，不继承 ASPIN 的收敛结论。
4. **Fe2 的分支处理优先于性能**：先用 continuation/分支标签保证同一
   磁支，再比较迭代数和 wall time；否则“更快”可能只是跳到了错误分支。

### 12.4 最终再评估

调研文档把方向从“改进固定密度子空间”提升到“响应层 + 缩减外层 +
多保真 + 分支层”，这个总体结论应保留。需要降级的是其中的强断言：
Neumann 截断不是全局展开，固定 H 子问题不是自动合法的 ASPIN 预条件器，
SQP 不能省略状态方程，响应成本必须按统一量纲重算，性能改善仍是待验证
目标。最稳妥的下一步仍是：**全空间残差验收和规范投影先落地，再根据
\(J_0\) 与 \(J\) 的实测差距决定是否支付响应求解成本。**
