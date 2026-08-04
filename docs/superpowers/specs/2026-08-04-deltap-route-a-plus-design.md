# Route A+ 设计方案：算符期望记账 + 代理目标外循环校准（审阅稿）

> 2026-08-04 · 状态：**待审阅，未实施**。
> 目的：修复"约束激活时力与 FD 不对上"（驻点组② 残差 84.8–419 eV/Å）。
> 依据文档：`2026-08-04-deltap-force-resolution-plan.md` §1 公式、
> `2026-08-03-deltap-force-stationary-group2-fd.md` 判决数据。

---

## 1. 设计思想（一句话）

**把 escon 记账和 SCF 约束变量都从 γ（Wilson 观测量）换成 Γ（哈密顿量里约束算符的
期望值），恢复 dspin 恒等式结构；γ 的物理目标通过一个廉价的外循环校准代理目标
t_Γ 来达成。**

三个组件必须**同时**改，缺一个恒等式都不闭合：

| 组件 | 现状 | Route A+ |
|------|------|----------|
| ① escon 记账 | −Σλ_I·γ_I（Wilson） | −Σλ_I·Γ_I（算符期望） |
| ② SCF 约束变量 | 残差 = γ − t_γ | 残差 = Γ − t_Γ |
| ③ 物理目标达成 | 直接约束 γ | 外循环：调 t_Γ 使 γ → t_γ（secant） |

---

## 2. 公式推导（完整）

### 2.1 Γ 的定义（逐原子）

H_c = H_HR + H_HK 已按 λ_I 逐原子分解，Γ_I 定义为"每单位 λ_I 的算符能量"：

```
E_c = Tr[ρ·H_c] = Σ_I λ_I·Γ_I

Γ_I = Γ_I^HR + Γ_I^HK
Γ_I^HR = τ_α(I)·⟨P̂_I⟩，  ⟨P̂_I⟩ = Σ_k w_k Σ_n f_n Σ_{lm∈I} |D_{I,lm,n}(k)|²
Γ_I^HK = −0.5·Im[ Σ_j Σ_p f_p · w_{I,p}^{(j)} · T_pp^{(j)} ]
        （compute_hk_force 的 E_HK 累加在 λ 求和前按原子拆分的同一对象）
```

注：⟨P̂_I⟩ 的 per-k SMO 权重形式与实空间 Tr[ρ·P̂_I] 应数值相等（实现后一次性
核对，见 §5-T0）。**PW 侧**：Γ_I = ⟨P̂_I^onsite⟩（becp 期望，dspin PW 的 Mi
累加机制现成）。

### 2.2 E' 恒等式（dspin 结构的恢复）

```
escon_new = −Σ_I λ_I·Γ_I
E' = E_tot + escon_new = E_tot − Tr[ρ·H_c] ≡ E_KS(ψ*)    （恒等式，无近似）
```

E' 的物理含义变为"约束态的 KS 能量"——这正是 relax 应该最小化的对象。
（原记账 E'_old = E_KS(ψ*) + Σλ_I(Γ_I − γ_I) 多一个 O(λ) 的失配项。）

### 2.3 力公式（恒等式恢复后的 Pulay 代表）

```
F_exact(E') = F_std + λ_I·(δΓ_I/δψ)(dψ/dR)          （严格推导，上轮评审）
```

约束激活于 Γ（dΓ/dR = 0）时：
```
Tr[Ô·∂ρ/∂R] = −Tr[ρ·∂Ô/∂R]  ⇒  λ·响应项 = Pulay(λÔ)
∴ F_exact = F_std + Pulay(λÔ) = F_std + A1 + A2 + B   （= 当前已实现的力）
```

残余误差 = 约束激活近似的精度，O(λ)：
```
|ΔF| ≈ λ·dΓ/dR|path ≈ 0.002 Ry × O(1 rad/Å) ≈ 0.01–0.05 eV/Å   （估算，待实测）
```

### 2.4 驻点路径的 λ-leakage（Route A+ 后的预言）

```
∂E'/∂λ_I = Γ_I − Γ_I − λ_I·∂Γ_I/∂λ_I = −λ_I·∂Γ_I/∂λ_I    （O(λ)，对比现状 O(1)=224 eV/Ry）
leak_A+ = λ·∂Γ/∂λ·dλ*/dR ≈ 0.002 Ry × (0.3–1 rad/Ry) × dλ*/dR
```

且 dλ*/dR 本身改善：约束变量 Γ 是平滑的（无分支阶梯），λ*(R) 轨迹病态消失。
**可证伪预言**：驻点组② 残差从 84.8 eV/Å 降至 **≤0.02 eV/Å 量级**（分支 A 几何）。
若实测 ≫ 此值 → 推导有误或有未识别项，停下重审。

### 2.5 外循环校准（物理目标 γ = t_γ 的达成）

```
t_Γ^(k+1) = t_Γ^(k) + κ·(t_γ − γ_meas^(k))
κ = secant 估计：κ = Δt_Γ/Δγ（用最近两次迭代的差分；首轮 κ=1，因 Γ 与 γ 对 λ
    的响应同号且量级相近——实现后用数据核实，见 §5-T4）
```

- relax 中每离子步一次（或每 N 步）；单点计算收敛后一次；
- 收敛判据：|γ_meas − t_γ| < tol_γ（默认 1e-2 rad，物理上足够）；
- γ_meas 必须用**跨离子步连续**的分支（branch.dat 机制已有；分支连续性修复
  （Phase 0.3）仍是前置，但其作用域从"SCF 内部"缩到"外循环读数"）。

### 2.6 E-field 等效公式的语义更新（文档项）

λ 现在共轭于 Γ 而非 γ。E=−πλ/(2a) 的推导基于 λ–γ 共轭对，Route A+ 后需重推导
或改报"λ 与实测 γ 的对应关系"。**列入交付物，不许遗漏**（F1 备忘录联动）。

---

## 3. 代码修改清单

### 3.1 module_deltap（Γ 计算，主要新增）

| 位置 | 改动 |
|------|------|
| `deltap.h` | 新增 `compute_operator_observable()` 声明；成员 `gamma_op_`（per-atom Γ） |
| `deltap_wannier.cpp` | `compute_gamma_scf` 流程内顺带累加 Γ_I^HR = τ_I·Σ_kwΣ_nf w_In（w_In 已算）；`compute_hk_correction`/`compute_hk_force` 的 E_HK 累加按原子拆分出 Γ_I^HK |
| 核对 | T0：⟨P̂_I⟩ 的 per-k 形式 vs 实空间 Tr[DMR·pre_hr]（hhrdbg 机制）数值一致 |

### 3.2 状态机（约束变量与记账切换）

| 位置 | 改动 |
|------|------|
| `deltap_scf.h` | `DeltapParams` 加 `observable_mode`（gamma/operator，A/B 对照用，默认 operator）；`DeltapState` 加 `gamma_op`、代理目标 `t_proxy` |
| `deltap_scf.cpp` | 残差/has_target 逻辑按 observable_mode 选 γ 或 Γ；`compute_dp_escon` 换为 −ΣλΓ（operator 模式）；`reset_ionic_step` 挂钩外循环 secant 更新 t_proxy |
| `deltap_common.h` | `compute_dp_escon` 语义不变（换输入向量即可），单测补 operator 模式用例 |

### 3.3 输入/输出

| 位置 | 改动 |
|------|------|
| INPUT | 新增 `deltap_observable`（默认 operator；gamma = 旧行为，供 A/B 与回归对照）；target 文件语义**保持物理 γ**（用户界面不变），程序内部转 t_Γ |
| `[DeltaP P3]` 打印 | 加 Γ_I 与 escon_new；`deltap_results.dat` 加列（解析方同步通知） |
| E-field 打印 | 标注新语义或暂时下挂 WARNING（§2.6） |

### 3.4 外循环位置

最小实现放 `DeltapScfSolver::reset_ionic_step`（relax 每离子步触发）+
`iter_finish` 收敛后单次（单点）。secant 状态（上次 t_Γ、γ）存 DeltapState。

### 3.5 不需要动的

A1/A2/B 力实现（恒等式恢复后它们自动成为正确的力代表）；分支选择/gauge/
Wilson 核心；PW 算符链（Γ^onsite 走 dspin 的 becp 机制）。

---

## 4. 附带收益（设计层面）

1. **分支阶梯与 SCF 解耦**：λ 更新的残差改用平滑的 Γ 后，target-aware 分支选择的
   阶梯不再进入 SCF 收敛路径——驻点协议非唯一（disp_plus 双驻点、E' 差 1.77 eV）
   这类病态在力问题上消失。分支连续性（Phase 0.3）仍需要，但降级为
   "外循环读数平滑"而非"力正确性前提"。
2. **E' 物理含义改善**：E' ≡ E_KS(ψ*) 是约束态真实能量，不同构型/不同 λ 间
   可直接比较（原点敏感性中的记账失配部分消失）。
3. **A/B 对照内建**：`deltap_observable=gamma` 保留旧路径，回归与复现可随时切换。

## 5. 测试计划（每步带可证伪预言）

| # | 测试 | 方法 | 预言（偏离即停） |
|---|------|------|------------------|
| T0 | Γ 实现的两种形式一致 | per-k SMO 权重 vs 实空间 Tr[DMR·pre_hr]，h2o1 base | 相差 <1e-10 |
| T1 | E' 恒等式 | operator 模式跑 h2o1，比较 E' 与"λ=0 同密度 KS 能量"构造 | E'−E_KS(ψ*) < 1e-8 eV |
| T2 | ∂E'/∂λ 重测 | 重复驻点轮的 λ 扫描（base 几何） | 斜率从 224 → O(λ)≈0.3–1 eV/Ry·λ 量级 |
| T3 | **驻点组② 复判（判决实验）** | h2o1 O1-z 三几何驻点（约束 Γ）FD | 残差 84.8 → **≤0.02 eV/Å** |
| T4 | 外循环收敛 | t_γ 设为 0.9×γ_natural，secant 迭代 | ≤5 步内 |γ−t_γ|<1e-2；κ 估计稳定 |
| T5 | 组① 冻结 λ FD | 对照 | 残差 0.615 → ~0.05 eV/Å 量级（λ·dΓ/dR） |
| T6 | 回归 | `deltap_observable=gamma` 全锚点逐字节不变；operator 模式 anchors 重建 | gamma 模式零回归 |
| T7 | MPI | D_I 修复（Phase 0.1）后 4-rank Γ/γ 跨 rank 一致 | 逐原子 ±0.01 rad |

**前置依赖**：Phase 0.1（D_I MPI 修复）必须在 T7 前；Phase 0.3（分支连续性）
只需在外循环投产（T4 之后）前完成，T0–T3 不依赖。

## 6. 风险与开放问题

| 风险 | 评估 | 缓解 |
|------|------|------|
| Γ(λ) 响应太弱/非单调，secant 发散 | 低（Γ 与 γ 对 λ 同族响应） | κ 限幅；T4 先验证映射单调性 |
| ⟨P̂⟩ 两种形式不等（SMO 权重 vs 实空间收缩的口径差） | 中 | T0 一次性核对；不一致则以实空间 Tr[DMR·pre_hr] 为准 |
| E' 语义变化影响 P 系列判据（F1/F2 链） | 中 | §2.6 文档联动；F2（γ→μ 换算）不受影响，F1 重推导 |
| λ≠0 锚点再次重建 | 确定发生 | 接受（operator 模式为新基准；gamma 模式锚点冻结保留） |
| H_HK 关闭时 Γ_I^HK=0 的退化行为 | 明确 | 自动退化（Γ=τ⟨P̂⟩），T3 在两种 H_HK 开关下各跑一次 |

## 7. 交付物清单

1. 代码：§3 全部（含 INPUT 开关、打印、单测）；
2. 文档：dev-guide v3 章节（三种记账推导 + dspin 恒等式 + Route A+ 结构）、
   §2.6 E-field 语义更新、实验 dated 文档（T0–T7）；
3. 锚点：operator 模式新锚点（ecutwfc=100 设置沿用）。

---

## 本轮记录

- 设计审阅稿，无代码改动。公式经 `2026-08-03-deltap-force-stationary-group2-fd.md`
  数据自洽校验（∂E'/∂λ=224 eV/Ry 用 Γ_op−γ 失配解释；Route A+ 后预言 O(λ)）。
