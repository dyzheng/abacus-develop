# 实空间格点权重实现 DFT+U 的可行性再分析（公式级）

> 问题：是否通过实空间权重划分重新实现 DFT+U 会更好？
> 方法：从 Dudarev 形式出发做算符级推广，推导能量/势/力的完整表达式，分析物理含义变化、实现路径与验证路线。
> 结论先行：**数学上良定义且有深刻协同（与约束框架共享观测量、PW≡LCAO 口径一致、内置线性响应定 U 机器），但它改变的是 U 的相关子空间定义（模糊盆地 ≠ d 壳层）——是研究项目，不是工程替换；且它并不解决 II-1b 的"约束控不住 d 矩"问题（那是观测量选择问题）。**

---

## 1. 出发点：Dudarev 形式的算符结构

旋转不变 Dudarev DFT+U：

$$
E_U=\frac{U_{\rm eff}}{2}\sum_{\sigma}\Big[\mathrm{Tr}\,\hat n^\sigma_P-\mathrm{Tr}\big((\hat n^\sigma_P)^2\big)\Big],\qquad \hat n^\sigma_P=\hat P\hat n^\sigma \hat P
$$

其中 $\hat P=\sum_m|\phi_m\rangle\langle\phi_m|$ 是**幂等投影子**（$\hat P^2=\hat P$，截断 NAO/PAW β/Wannier 均可），把相关子空间钉在 d 壳。能量两项的角色：$\mathrm{Tr}\,\hat n_P$ 计数子空间占据；$\mathrm{Tr}(\hat n_P^2)$ 测子空间的**非幂等度**（分数占据），惩罚它驱动整数占据。

势（对密度矩阵变分）：

$$
\hat V_U^\sigma=U_{\rm eff}\Big(\tfrac12\hat P-\hat P\hat n^\sigma\hat P\Big)
$$

**三个结构要素**：①幂等投影子定义相关子空间；②占据矩阵的壳层内结构（Dudarev 已丢弃多极项，只留标量）；③双计数扣除（FLL/AMF，与投影子无关的独立难题）。

## 2. 实空间推广：把投影子换成权重算符

权重算符 $\hat w$（实空间乘子，$w(\mathbf r)\in[0,1]$，逐点 $\sum_I w_I\equiv 1$）是**非幂等**的"模糊投影子"。自然推广（唯一保持两项结构的写法）：

$$
\boxed{\;E_U^{(w)}=\frac{U}{2}\sum_\sigma\Big[\mathrm{Tr}(\hat w\hat n^\sigma)-\mathrm{Tr}\big((\hat w\hat n^\sigma)^2\big)\Big]\;}
$$

### 2.1 幂等极限回归检查（正确性验证）

若 $\hat w=\hat P$ 幂等：$\mathrm{Tr}((\hat P\hat n)^2)=\mathrm{Tr}(\hat P\hat n\hat P\hat n)=\mathrm{Tr}((\hat P\hat n\hat P)^2)=\mathrm{Tr}(\hat n_P^2)$（用 $\hat P^2=\hat P$ 与迹循环）。**精确退回 Dudarev**。所以该形式是 DFT+U 对非幂等窗的自然延拓，不是任意构造。

### 2.2 能带空间展开（实现形式）

KS 基 $\hat n=\sum_i f_i|\psi_i\rangle\langle\psi_i|$，记权重算符的能带矩阵

$$
W_{ij}=\langle\psi_i|\hat w|\psi_j\rangle=\int w(\mathbf r)\,\psi_i^*(\mathbf r)\psi_j(\mathbf r)\,d\mathbf r
$$

则

$$
\mathrm{Tr}(\hat w\hat n)=\sum_i f_i W_{ii}\equiv N_I^{(w)}\quad(=\text{本框架的约束读数 }Q)
$$

$$
\mathrm{Tr}((\hat w\hat n)^2)=\sum_{ij}f_if_jW_{ij}W_{ji}=\sum_{ij}f_if_j|W_{ij}|^2
$$

$$
\Rightarrow\;E_U^{(w)}=\frac{U}{2}\sum_\sigma\Big[\sum_i f_iW_{ii}-\sum_{ij}f_if_j|W_{ij}|^2\Big]
$$

**对角元 $W_{ii}$ 就是逐带加权布居**（框架输出标签"逐带电荷分解/权重投影 DOS"的同一条数据）；非对角元 $W_{ij}$ 是新增量（能带对网格积分）。

### 2.3 势

对 $\hat n$ 变分（注意 $\mathrm{Tr}((\hat w\hat n)^2)=\mathrm{Tr}(\hat w\hat n\hat w\hat n)$）：

$$
\hat V_U^{(w)}=\frac{U}{2}\hat w-\frac{U}{2}\big(\hat w\hat n\hat w+\hat w\hat n\hat w\big)\cdot\tfrac12\cdot 2=\frac U2\,\hat w-U\,\hat w\hat n\hat w
$$

能带矩阵元：

$$
\boxed{\;V^{(w)}_{ij}=U\Big[\tfrac12 W_{ij}-\sum_k f_k W_{ik}W_{kj}\Big]\;}
$$

与 Dudarev 的 $V_{mm'}=U[\frac12\delta_{mm'}-n_{mm'}]$ 逐项同构：**$\hat w$ 扮演相关子空间内的"单位算符"**，$WfW$ 扮演占据矩阵。注入路径上，$\hat w$ 是格点乘子（与约束注入同一通道），能带空间收缩 $WfW$ 是有限秩代数——实现复杂度中等。

### 2.4 力

包络定理下只剩显式项：$\partial\hat w/\partial R_J$（M0/M1 已有导数网格）穿过同一组公式；能带响应项被 SCF 驻点消去（与约束力同构）。**不需要新的 Pulay 链**（权重不进基组）。

## 3. 物理含义变化（这是"好不好"的核心）

### 3.1 模糊窗的幂等读数偏移（关键技术警告）

对单一条完全占据态 $|\psi\rangle$：

$$
\mathrm{Tr}((\hat w\hat n)^2)=W^2=\langle\psi|\hat w|\psi\rangle^2<\langle\psi|\hat w|\psi\rangle=\mathrm{Tr}(\hat w\hat n)
$$

即使该态"物理上完全局域于原子 I"，只要 $w<1$ 处有尾巴，读数就是"分数"。于是 $E_U^{(w)}$ 对**所有**占据态施加一个把 $W\to1$ 的驱动——即"**把态压进盆地内部**"的局域化压力。对 d 态 $W\sim0.8$–$0.9$，该压力会把它们进一步挤进盆地——**相对标准 DFT+U 可能过度局域化**，且定量上 U(w) 的基线与 Dudarev 不同（必须用线性响应重新定标，不能搬 U 值）。

### 3.2 共价区的幂等冲突（适用域边界）

盆地划分在化学上是任意的：共价键电子在两个盆地里**固有地分数占据**（共价键合本身破坏区域幂等性）。$E_U^{(w)}$ 驱动区域幂等化 ⇔ 与键合电子结构作对。在离子型 TM 氧化物（d 壳主导盆地幂亏欠）里该矛盾温和；在共价体系里该模型**方向性错误**。→ 适用域≈标准 DFT+U 的适用域（近原子离子的 TM/稀土化合物），不更宽。

### 3.3 相关子空间语义变化

Dudarev 的相关子空间是 **d 壳**（角动量分辨）；权重盆地是 **Fe-ish 空间区**（含配体尾部）。U(w) 校正的是"该区域的有效关联"，多极结构无从谈起（Dudarev 已丢弃，U(w) 更无从恢复）。它是一个**不同的有效模型**，有效性只能靠"自定标 + 实验/谱学验证"确立。

### 3.4 双计数不受影响

FLL vs AMF 的双计数难题与投影子/权重无关，U(w) 不改善也不恶化。

## 4. 与约束框架的协同（真正的吸引力所在）

1. **观测量统一**：U 的校正量与约束的控制量成为**同一个** $N_I^{(w)}$ 与同一组 $W_{ij}$——§2 的口径错配（约束动盆地矩、U 管 on-site d）在定义层面消失：不再有"廉价尾部通道绕过 U"，因为 U 也作用在同一个加权子空间上。
2. **基组口径一致**：权重网格 PW≡LCAO → U(w) 的投影子依赖问题（U 值只对同投影子可移植的著名痛点）原则上被消除。
3. **内置的线性响应定 U 机器**：Cococcioni–de Gironcoli 的 $U=\chi_0^{-1}-\chi^{-1}$ 需要响应矩阵 $\chi=\partial Q/\partial\mu$——**这正是本框架外环 Jacobian（阶段 B 的 Broyden 对象）**。约束框架本身就是 U(w) 的自洽定标仪。
4. **增量实现路径短**：对角 $W_{ii}$ 已有（输出标签数据）；非对角 $W_{ij}$ 复用网格与能带内积基建；势注入与约束同通道；力无新 Pulay 链。

## 5. 成本与工程量

| 项 | 量级 | 说明 |
|---|---|---|
| $W_{ij}$ 全矩阵 | $O(N_k\cdot N_{\rm bands}^2\cdot N_{\rm grid})$ | bulk TM 氧化物数百带 × k 点——比投影子占据（$O(N_{\rm proj}N_bN_{pw})$）重，但可接受；可分块/截断（$|W_{ij}|$ 随能级差衰减需实测） |
| $V_U^{(w)}$ 应用 | 格点乘 $\hat w$ + 能带秩更新 | 与约束注入同通道 |
| 定标（线性响应） | 复用外环 | μ–Q 扫描即 χ 测量 |
| **总计** | **~1–2 周 PW 原型 + 验证轮** | 研究级投入，非工程修复 |

## 6. 验证路线（若立项）

| 步 | 内容 | 判据 |
|---|---|---|
| V-a | 只读实现 $\mathrm{Tr}((\hat w\hat n)^2)$ 作**诊断量**（不进物理路径），FeO/NiO 上与 on-site 幂等度做趋势对照 | 趋势一致（顺序/相对变化），记录 basin 模糊偏移量 |
| V-b | 标量极限检验：FeO 上加 $E_U^{(w)}$，自定标 U(w)（线性响应），看 Mott/CT 带隙趋势 | 与标准 DFT+U 同趋势；报告偏移与过度局域化迹象（§3.1） |
| V-c | 与标准 DFT+U 的带结构/磁矩逐点对拍 + 双基组 PW≡LCAO 一致性 | 口径一致声明只在同 U(w) 定义下成立 |
| V-d | 文献核查（**必须先做**）："DFT+U real-space charge / Hirshfeld-U / fragment idempotency / Wannier DFT+U"——本仓库无法核实新颖性，疑似与 Wannier-DFT+U 及 Mulliken-charge-U 类工作同族 | 不重复造轮子；若已存在则改为对齐实现 |

## 7. 结论

1. **它解决不了 II-1b**："约束控不住 d 矩"是观测量选择问题（Becke 盆地矩 vs d 矩），U(w) 不改变约束观测量，只改变校正项。
2. **作为研究方向成立且吸引力真实**：公式良定义（幂等极限精确回归 Dudarev）、与约束框架深度协同（观测量统一 + 基组一致 + 内置定标机器）、实现路径短。
3. **但它不是 DFT+U 的替代品**：相关子空间从 d 壳变为模糊盆地（§3.1 过度局域化偏移、§3.2 共价冲突、§3.3 语义变化），有效模型不同，必须独立定标与验证；双计数问题照旧。
4. **定位建议**：登记为**阶段 C/D 研究项**（"DFT+U(w)：与约束一致的实空间关联校正"），先做 V-d 文献核查与 V-a 诊断量（低成本高信息），再决定是否投入原型。当前 II-1 链不受影响，继续走 onsite= 仪器 + 半径敏感性 + 阶段 C on-site 权重权衡的路线。

---

## 附录：关键公式速查

- 能量：$E_U^{(w)}=\frac U2\sum_\sigma[\sum_i f_iW_{ii}-\sum_{ij}f_if_j|W_{ij}|^2]$
- 势：$V^{(w)}_{ij}=U[\frac12W_{ij}-\sum_k f_kW_{ik}W_{kj}]$
- 幂等极限：$\hat w=\hat P$ 时精确退回 Dudarev（§2.1）
- 单态偏移：$E_U^{(w)}$ 对 $W<1$ 的完全占据态仍施加 $W\to1$ 驱动（§3.1）
- 定标：$U=\chi_0^{-1}-\chi^{-1}$，$\chi_{\alpha\beta}=\partial Q_\alpha/\partial\mu_\beta$（外环 Jacobian 即测量器）
