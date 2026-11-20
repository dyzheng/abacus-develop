# 2026-11-18 saw 场、fixed-mu 与周期性：为什么 bulk 不能直接用 saw

## 0. 结论摘要

1. **fixed-mu 只固定 μ，不改变势的空间形状**。约束势始终是
   `v_con(r) = μ w(r)`；周期性由 `w(r)` 决定，不由“fixed”还是“动态 μ”决定。
2. **dipole-saw 的 `w(r)` 本身是周期、连续的**：
   `w = -Lz * saw(z)`，`saw` 是周期函数，所以 fixed-mu + saw **满足周期边界**。
   它的问题不是“不周期”，而是：它是**零平均场的锯齿势**（ramp + 反向 return），
   只有把 **return/decay 区放在真空** 时，材料才等效看到一个均匀偏置场。
3. **dipole-molecular 的 `w = -z` 才是非周期的**：在 `z=0/Lz` 处有跳变；
   fixed-mu + molecular 不满足周期势，跳变必须落在真空里。
4. **fixed-mu + charge 权重**是周期的，但它是局域原子/片段势，**不是均匀电场**。
5. 所以对 **bulk**（密度充满整个周期盒）：
   - saw 的 return 区必然穿过电子云 → 不是干净的均匀场；
   - molecular 的跳变必然穿过电子云 → 周期势被破坏；
   - 正确做法是 **slab + 真空**、**finite-field/Berry phase**，或把 fixed-mu 用于
     **局域电荷转移**（charge 约束）而不是均匀场。

用户的两个直觉基本正确，但需要精确化：

- “电荷分布要局域” → **更准确地说：saw 的 return/decay 区必须有可忽略的电子密度**。
  局域分子/团簇/slab 自然满足；bulk 液体/固体不满足。
- “周期边界不能穿过电子云” → 对 **molecular 权重**（跳变在边界）成立；
  对 **saw 权重**，saw 本身周期连续，**边界不是跳变**，真正不能穿过电子云的是
  **decay window**。

---

## 1. 约束势与 fixed-mu 的定义

`ConstraintInjectPW::inject` 注入的是

```
v_con(r) = Σ_alpha μ_alpha w_alpha(r)
```

对单个约束：

```
v_con(r) = μ * w(r)
```

`fixed_mu` 模式（`ConstraintLoop::init`）只做一件事：把 `μ` 设为输入值并跳过
外循环更新；它**不改变** `w(r)`。因此：

| 方案 | μ | w(r) | 势的周期性 |
|---|---|---|---|
| fixed-mu + dipole-saw | 常数 | `-Lz * saw(z)` | **周期**（saw 周期连续） |
| fixed-mu + dipole-molecular | 常数 | `-z` | **非周期**（边界跳变） |
| fixed-mu + charge (MBIS/Becke) | 常数 | 周期原子权重 | **周期**（局域势） |
| 动态 μ + dipole-saw | 外循环更新 | 同上 | 同上 |

所以“fixed-mu 不满足周期性”这句话只对 **molecular 权重**成立；对 saw 权重，
fixed-mu 与动态 μ 的周期性完全一样。

---

## 2. saw 权重是如何给出 saw 形式的

代码：`WeightGrid::build_dipole()`（saw 分支）与
`DipoleObserver::saw()` / `Efield::saw_function()`：

```
fac = 1 - pos_dec
x = fractional z in [0,1)

x <= pos_max:              saw = x - pos_max + 0.5*fac
pos_max < x <= pos_max+pos_dec:
                           saw = 0.5*fac - fac*(x-pos_max)/pos_dec
x > pos_max+pos_dec:       saw = x - pos_max - 1 + 0.5*fac
```

约束权重：

```
w(z) = -Lz * saw(z)
v_con(z) = μ w(z) = -μ Lz * saw(z)
```

### 2.1 saw 是周期、连续函数

在 `x=pos_max`、`x=pos_max+pos_dec` 两侧取值相等；且

```
saw(0) = saw(1) = -pos_max + 0.5*fac
```

所以 `saw(x+1) = saw(x)`，`w` 和 `v_con` 都是周期的。**周期边界条件本身没有问题**；
导数在 `pos_max` 和 `pos_max+pos_dec` 处有折点（kink），势连续但场不连续。

### 2.2 平均场为零

对分数坐标 `x`：

- ramp 区（长度 `1-pos_dec`）：`dsaw/dx = +1`
- decay 区（长度 `pos_dec`）：`dsaw/dx = -(1-pos_dec)/pos_dec`

一个周期内的总变化：

```
(1-pos_dec)*1 + pos_dec*(-(1-pos_dec)/pos_dec) = 0
```

即 **saw 势的平均斜率为零**，平均电场为零。它不是一个“处处均匀的场”，而是
“大部分区域是均匀 ramp，另有一小段反向的 return”。这正是 saw（锯齿）模型：
用一段反向 return 把势拉回周期值。

### 2.3 为什么 slab 可以、bulk 不行

- **slab/分子/团簇 + 真空**：把 decay window 放在真空中，电子密度在
  return 区近似为 0。材料内的电子只感受到 ramp 区的均匀场，return 区对能量/
  受力的贡献可忽略。这就是 `efield`/dipole-saw 的适用场景。
- **bulk**：密度充满整个周期盒，没有真空区放 decay window。return 区会穿过
  电子云，电子同时感受到 ramp 的正向场和 return 的反向场；平均场为零，
  这不是“加了一个均匀偏置”，而是加了一个周期锯齿势，结果依赖 return 区
  切过哪些分子/原子。

这也解释了为什么之前 32 H₂O 必须做成 **slab**：不是忘了 bulk，而是 saw 模型
在 bulk 里没有物理上干净的 return 区域。

---

## 3. molecular 权重 `w=-z`：真正的不周期

代码：`WeightGrid::build_dipole()` 的 molecular 分支：

```
w(z) = -z
v_con(z) = -μ z
```

`w(z+Lz) = -z - Lz != w(z)`，`v_con` 在 `z=0/Lz` 处有跳变。fixed-mu 只是把
这个势的幅度固定，**不能消除跳变**。

- 对有限分子/团簇放在大盒子里：跳变落在真空，密度在跳变处为 0，
  电子不感受跳变，分子内部近似均匀场。这是 molecular frame 的用法。
- 对 bulk：跳变穿过水/固体，周期势被破坏，结果依赖跳变面位置，不能作为
  均匀场。

另外，dipole 本身在带电体系中依赖 Cartesian 原点；`delta` 语义可以消掉参考
偶极，但**势的跳变位置**仍然存在，不能靠 delta 消掉。

---

## 4. fixed-mu + charge 约束：周期，但不是场

如果“fixed-mu 方案”指的是对 **charge/MBIS 权重**固定 μ，那么：

- `w_alpha(r)` 是周期原子权重（`Σ w=1` 或片段和），势是周期的；
- 但它是**局域势**：`μ w_alpha(r)` 只在原子/片段附近非零，不会产生宏观均匀
  电场；
- 适合做“局域电荷转移/偏置某个片段”，不适合“整个 bulk 加均匀电场”。

所以 fixed-mu 能否用于 bulk，取决于你想加的是 **局域偏置** 还是 **均匀场**：

| 目标 | 可用方案 | 是否适合 bulk |
|---|---|---|
| 均匀电场 | dipole-saw + 真空 return | 只适合 slab |
| 均匀电场 | finite-field/Berry phase | 适合 bulk（需实现） |
| 均匀电场 | dipole-molecular + 真空跳变 | 只适合有限体系/大盒 |
| 局域电荷偏置 | fixed-mu + charge 约束 | 适合 bulk（周期、局域） |

---

## 5. 对用户两个判断的精确化

### 5.1 “电荷分布一定要局域而不能离散”

更准确地说：

> **saw 的 return/decay 区必须有可忽略的电子密度。**

- 局域分子/团簇/slab：自然有真空区，满足。
- bulk 液体/固体/金属：密度充满整个周期盒，没有真空区放 return；
  即使总电荷局域在某个溶质上，只要溶剂/背景电子密度覆盖 return 区，
  仍会有 artifact。
- 所以“局域”是充分条件，不是必要条件；关键是 **return 区的密度**。

### 5.2 “周期边界不能穿过电子云”

分两种权重：

- **molecular `w=-z`**：跳变就在周期边界上，所以 **边界必须落在真空**；
  边界穿过电子云会直接破坏周期势。
- **saw `w=-Lz*saw`**：势本身周期连续，**边界不是跳变**；真正不能穿过电子云的
  是 **decay window（return 区）**。边界可以穿过电子云，只要 decay 区在真空里；
  但因为 bulk 没有真空区，实际还是不能用。

---

## 6. 代码与已有验证

- `source/source_estate/module_constraint/weight_grid.cpp`：`build_dipole()`
  - saw 分支：`cw_[0][ir] = -lz * DipoleObserver::saw(...)`
  - molecular 分支：`cw_[0][ir] = -grid_position(ir)[dipole_axis_]`
- `source/source_estate/module_constraint/constraint_dipole.cpp`：
  `DipoleObserver::saw()`（与 `Efield::saw_function()` 同形）。
- `source/source_estate/module_constraint/constraint_loop.cpp`：
  `fixed_mu_` / `fixed_mu_value` 只设置 `mu_`，不改变 `w`。
- `source/source_estate/module_pot/efield.cpp`：`Efield::saw_function()` 与
  saw 权重同形；`Potential` 以 `fac*saw` 注入。
- 已有数值验证（`2026-11-18-fixed-mu-vs-efield-and-charged-constraint.md`）：
  固定 μ dipole-saw 与匹配幅度的 `efield` 总能量差 **9.9×10⁻⁸ eV**，
  说明二者是同一个 saw 势；fixed-mu 不会改变 bulk 的 saw return 问题。
- 已有 32 H₂O 结果（slab，12×12×20 Å，水层 z≈6–11 Å）：
  - 无偏置/有偏置各 ~14m45s（96 atoms，LCAO，OMP=20）；
  - 偏置下上下层净电荷和变化约 **±0.021 e**，总 z 偶极增加 **0.138 e·Å**；
  - 这是“有真空 return 区”的合法 saw 场结果。

---

## 7. 测试结果与解释

### 7.1 fixed-mu dipole-saw 与 efield 等价（LiCl 15 Å）

设置：LiCl/PW，15 Å 立方盒，saw 参数 `pos_max=0.9`、`pos_dec=0.1`；
固定 μ dipole-saw（`mu=-0.01`）vs 外电场（`efield_amp=+0.005`，
立方胞换算 `efield_amp=-mu/2`）。

| 量 | 固定 μ dipole-saw | efield | 差 |
|---|---:|---:|---:|
| `FINAL_ETOT` | −642.7326304217 eV | −642.7326305203 eV | **9.9×10⁻⁸ eV** |
| Li `F_z` | 0.12552 eV/Å | 0.12362 eV/Å | 0.0019 |
| Cl `F_z` | −0.12552 eV/Å | −0.13026 eV/Å | 0.0047 |

结论：fixed-mu dipole-saw 与 efield 是**同一个 saw 势**；fixed-mu 只固定
振幅，不改变周期性/return 区问题。力差 0.002–0.005 eV/Å（场致力量级的
2–4%）来自两套力实现与 SCF 阈值。

### 7.2 32 H₂O slab：无偏置 vs 有偏置

几何：12×12×20 Å 盒子，水层 z≈6.1–11.0 Å，上下真空；32 个 H₂O，
96 atoms，256 e。LCAO，gamma-only，PBE，140 bands，`ecutwfc/ecutrho=40/160`，
OMP=20，np=1。

电场：saw，`efield_dir=2`（z），`efield_amp=0.002`，
`efield_pos_max=0.9`，`efield_pos_dec=0.1`；打印的势幅
`0.13606 Ry`（≈1.85 eV over the ramp）。saw 的 decay window 在
z=18–20 Å，落在水层上方的真空区。

MBIS 电荷：对 96 个原子各一个 charge 约束，`target_mode=delta`、
`target=0`，即读取 reference MBIS 布居；μ=0，不改变 SCF。
`constraint_mbis_defer_reference=1`，reference 只构建一次。
（这里场由 `efield` 施加；它与 fixed-mu dipole-saw 等价，见 7.1。
之所以不直接把 dipole 固定 μ 和 MBIS 电荷约束同时放进一个
`WeightGrid`，是因为一个 `WeightGrid` 只能有一种权重类型；MBIS
电荷读取需要用 MBIS 权重。）

| 量 | 无偏置 | 有偏置 |
|---|---:|---:|
| wall time（OMP=20, np=1） | **14m46s** | **14m44s** |
| ABACUS 内部 total | 885 s | 883 s |
| `FINAL_ETOT` | −14923.0172726457 eV | −14923.0373257749 eV |
| MBIS 总电荷 ΣQ | 256.000000 | 256.000001 |

分项时间（两次几乎相同，以无偏置为例）：

| 项 | 时间 | 占比 |
|---|---:|---:|
| `Gint cal_gint_vl` | 217 s | 24.5% |
| `HSolverLCAO solve` | 118 s | 13.3% |
| `DiagoElpa elpa_solve` | 52 s | 5.9% |
| `HamiltLCAO updateHk` / `Veff contributeHR` | 32 s | 3.7% |
| `dm2rho` / `cal_gint_rho` | 32 s | 3.6% |

其余为初始化、电荷混合与**一次** MBIS reference 构建；reference 相没有
逐 SCF 重建。

### 7.3 电荷分布差别

“每分子净电荷” = `Q_O + Q_H1 + Q_H2 − 8`（e）；总电荷始终为 256。

| 量 | 无偏置 | 有偏置 |
|---|---:|---:|
| 每分子净电荷 std | 0.01994 e | 0.01986 e |
| O 平均电荷 | 6.9375 e | 6.9375 e |
| H 平均电荷 | 0.5312 e | 0.5312 e |
| 逐原子平均 |ΔQ| | — | **0.00250 e** |
| 逐原子最大 |ΔQ| | — | **0.00710 e**（O） |
| O 平均/最大 |ΔQ| | — | 0.00345 / 0.00710 e |
| H 平均/最大 |ΔQ| | — | 0.00202 / 0.00453 e |
| 每分子净电荷变化 RMS | — | **0.00138 e** |
| 每分子净电荷变化最大 | — | **0.00216 e** |
| 净电荷变化 >0.01 e 的分子数 | — | **0** |

按层（下层 z≈7 Å / 上层 z≈10 Å）：

| 层 | 无偏置净电荷和 | 有偏置净电荷和 | 变化 |
|---|---:|---:|---:|
| 下层 | −0.01284 e | +0.00848 e | **+0.02132 e** |
| 上层 | +0.01284 e | −0.00848 e | **−0.02132 e** |

由 MBIS 净电荷算出的总 z 方向偶极（Σ(Z−Q)z，总电荷为零，原点无关）：

```
无偏置  0.14098 e*Angstrom
有偏置  0.27943 e*Angstrom
变化   +0.13845 e*Angstrom
```

### 7.4 偶极响应来自哪里的电荷重分布

前面的净偶极只是总数；这里把它严格分解成 **分子间/层间电荷转移** 与
**分子内 O–H 极化** 两部分。定义：

```
δ_i = Z_i - Q_i                 (净正电荷；正 = 电子亏缺)
q_m = Σ_{i in m} δ_i = 8 - (Q_O + Q_H1 + Q_H2)   (分子净正电荷)
p_z = Σ_i δ_i z_i
    = Σ_m q_m z_{O_m}                     (intermolecular / layer transfer)
    + Σ_m Σ_{i in m} δ_i (z_i - z_{O_m})  (intramolecular O–H polarization)
```

> 注意 §7.3 用的是电子超出量 `Q_sum-8`；本节用净正电荷
> `q_m = 8-Q_sum = -(Q_sum-8)`，符号相反，物理信息相同。

**总偶极分解**

| 分量 | 无偏置 (e·Å) | 有偏置 (e·Å) | Δ (e·Å) | 占总 Δ 比例 |
|---|---:|---:|---:|---:|
| `p_z` 总 | 0.14098 | 0.27943 | **+0.13845** | 100% |
| `p_inter`（分子间/层间） | −0.03285 | +0.03194 | **+0.06479** | **46.8%** |
| `p_intra`（O–H 分子内） | 0.17383 | 0.24749 | **+0.07365** | **53.2%** |

两项之和 `0.06479 + 0.07365 = 0.13845`，严格闭合。

**层间电荷转移对 `p_inter` 的贡献**

按层平均（下层 z≈7 Å，上层 z≈10 Å；`q_m` 为净正电荷）：

| 层 | 无偏置 `Σq_m` | 有偏置 `Σq_m` | Δ |
|---|---:|---:|---:|
| 下层 | +0.01284 e | −0.00848 e | **−0.02132 e**（电子富集） |
| 上层 | −0.01284 e | +0.00848 e | **+0.02132 e**（电子亏缺） |

即约 **0.0213 e** 从上层转移到下层，形成沿 +z 的层间偶极，
贡献 `p_inter` 的 +0.06479 e·Å。

**分子内 O–H 极化对 `p_intra` 的贡献**

每个分子的 `d_m = Σ δ_i (z_i - z_O)`（只有 H 项非零，因为 O 是参考点）：

| 量 | 无偏置 | 有偏置 | Δ |
|---|---:|---:|---:|
| 每分子 `d_m` 平均 | +0.00543 | +0.00773 | **+0.00230** |
| 每分子 |d_m| 平均 | 0.22279 | 0.22316 | +0.00037 |
| 每分子 `d_m` 最大变化 | — | — | **0.00506 e·Å** |

32 个分子的 `d_m` 变化同向、幅度小，合计 +0.07365 e·Å：场使 O–H 键的
电荷分布发生一致的微小极化。O 与 H 的电荷变化总和分别为
`ΣΔQ_O=+1.04×10⁻⁴ e`、`ΣΔQ_H=−1.04×10⁻⁴ e`，总电荷守恒；O 的电荷变化
主要进入分子净电荷 `q_m`（即 `p_inter`），H 相对 O 的变化进入 `p_intra`。

**电子云质心**

总偶极还可以写成电子云质心的移动（总电荷 256，离子位置固定）：

```
z_e = Σ Q_i z_i / 256
无偏置 8.503200 Å -> 有偏置 8.502659 Å
Δz_e = -0.000541 Å
Δp_z = -256 * Δz_e = +0.13845 e·Å
```

即电子云整体向 **−z** 移动约 **5.4×10⁻⁴ Å/电子**，产生 +z 的总偶极。
这与层间电荷转移（下层电子富集）和 O–H 极化方向一致。

**结论**

- 该 saw 场下的偶极响应不是“刚性整体平移”，而是 **≈47% 分子间/层间
  电荷转移 + ≈53% 分子内 O–H 极化** 的叠加；
- 每分子净电荷变化最大仅 0.00216 e，没有分子电离；层间 0.0213 e 的
  转移是长程极化的主体；
- 总电荷严格守恒，电子云质心移动 −5.4×10⁻⁴ Å，给出 +0.13845 e·Å 偶极。

### 7.5 结果说明：这组数据为什么支持本文的结论

1. **电荷守恒、分区完整**：两次 ΣQ=256，说明 slab + 真空里的 MBIS
   分区没有漏电荷；saw 的 decay 在真空，没有把 return 区的电荷算进来。
2. **响应很小但方向清楚**：场幅 1.85 eV 下，逐原子平均变化仅
   0.0025 e，最大 0.0071 e（O）；每分子净电荷变化最大 0.0022 e，
   没有整分子电离。总偶极增加 **0.13845 e·Å**，严格分解为
   **≈47% 层间电荷转移**（+0.06479 e·Å）与 **≈53% O–H 分子内极化**
   （+0.07365 e·Å），电子云质心移动 −5.4×10⁻⁴ Å（§7.4）。
   这与液态水的强屏蔽/介电响应一致。
3. **为什么这是 saw 的“合法”用法**：水层 z≈6–11 Å，decay 在 z=18–20 Å
   的真空；电子只感受 ramp 的均匀场，return 区密度≈0。若把同一个 saw
   直接放进 **bulk** 水盒（密度覆盖 z=18–20 Å），return 会切过水分子，
   上下层电荷转移/偶极会混入反向场的 artifact，不能解释为均匀偏置。
4. **fixed-mu 不改变结论**：7.1 已证明 fixed-mu dipole-saw 与 efield 是
   同一势；因此“bulk 不适合 saw”是 saw 的 return 几何问题，不是
   fixed-mu vs efield 的实现问题。fixed-mu 只固定 μ，不会把 saw 变成
   周期 bulk 的均匀场。
5. **关于 MBIS 电荷约束**：本算例的 charge 约束 μ=0，只用于读取 MBIS
   布居，不施加偏置。若把 fixed-mu 用在 **charge** 约束上（μ≠0），它是
   周期、局域的势，可以用于 bulk 的局域电荷偏置，但那不是均匀电场。

### 7.6 适用边界

- 该 slab 结果的物理解释限于“带真空 return 区的 saw/gate 场”；
  不能外推为三维 bulk 的均匀电场。
- 真正的 bulk 均匀场需要 finite-field/Berry phase，或带真空/电极的
  slab 超胞；当前 `efield`/dipole-saw 的实现不是这个模型。
- 若只关心 bulk 中的局域电荷转移，可用 fixed-mu + charge/MBIS 约束；
  但要明确它是局域势，不是宏观均匀场。

## 8. 建议

1. **要做均匀电场**：
   - 有限体系/薄膜：slab + 真空，saw 的 decay 放在真空（当前 `efield` /
     dipole-saw 的正确用法）；
   - 真正 bulk：实现 finite-field/Berry phase，或使用带电极/真空的
     slab 超胞，不要直接把 saw 塞进密度充满的周期盒。
2. **要在 bulk 里用 fixed-mu**：
   - 用 **charge/MBIS 约束**做局域电荷偏置（周期、合法），不要声称是均匀场；
   - 或把固定 μ 的 dipole 约束仅用于有限体系/大盒（跳变在真空）。
3. **判断是否可用 saw 的实用检查**：
   - 画出沿场方向的平均密度 `ρ(z)`；
   - 若 decay window 内的 `ρ` 不是数值零，结果就含有 return 区 artifact，
     不能解释为均匀偏置场。
