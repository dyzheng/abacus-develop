# DeltaP 内循环分析与两阶段测试 (H2O)

> 日期: 2026-07-20

---

## 一、测试配置

| 参数 | 值 |
|------|-----|
| 体系 | H2O 块体 (1分子, 5×5×5 Bohr³ 周期胞) |
| k点 | 2×2×2 Gamma-centered |
| 基组 | LCAO DZP, genelpa |
| 赝势 | O.upf, H.upf (NC) |
| 电子数 | 8 (4 占据态), nbands=10 |
| smearing | gauss 0.01 |
| mixing | broyden 0.4 |
| symmetry | -1 (全关) |
| OMP_NUM_THREADS | 1 |

### DeltaP 参数
| 参数 | 值 |
|------|-----|
| deltap_switch | 1 |
| deltap_corr | 1 |
| deltap_nscf | 1 |
| deltap_lambda_init | 0.05 |
| deltap_lambda_step | 0.01 |
| deltap_target | (0.10, 0.05, 0.05) for (O, H1, H2) |
| deltap_gdir | 3 (z方向) |
| deltap_rm | 3 (默认) |

---

## 二、Bug 修复记录

### Bug 1: init_inner_loop() 从未被调用
- **文件**: `source/source_esolver/esolver_ks_lcao.cpp:857`
- **修复**: 在 `deltap_init()` 中添加 `dp->init_inner_loop();`
- **影响**: `inner_loop_active()` 始终返回 false，BFGS 内循环从不运行
- **后果**: 退化为梯度下降 + cooldown（每5步更新一次 λ）

### Bug 2: nbands 不足
- **问题**: `nbands=4` 且 `smearing=gauss` 导致 `WARNING_QUIT`
- **修复**: `nbands` 增至 8

---

## 三、内循环详细过程分析 (iter=2 的首次 BFGS 步)

```
[DeltaP] iter=1:      γ=(+5.050, +3.695, +3.695)  λ=(0.0500, 0.0500, 0.0500)
                                                      target=(0.10, 0.05, 0.05)
                                                      max|γ-target|=4.95

 --- BFGS 内循环 (冻结电荷密度) ---

[inner loop start]    λ=(0.050000, 0.050000, 0.050000)  rms=0.0000
[inner=0]             λ_trial=(0.055000, 0.053682, 0.053682)  rms_pred=4.1263
                      重对角化后: γ_trial=(-0.04981, -0.03589, -0.03589)
[result]              α_opt=9.837×10⁻⁴ (接受率 0.1%)
[inner loop done]     λ_final=(0.054870, 0.053586, 0.053586)
                      γ_final=(-0.050, -0.036, -0.036)

 --- 下一个 SCF 步 (电荷弛豫) ---

[DeltaP] iter=3:      γ=(-2.877, -1.642, -1.642)  λ=(0.05488, 0.05359, 0.05359)
                      max|γ-target|=2.977  ← 从 0.15 反弹到 2.98
```

### 关键数字

| 量 | 值 |
|------|-----|
| 天然 γ (λ=0.05 初值) | +5.05 (O), +3.70 (H×2) |
| 内循环优化后 γ | -0.050 (O), -0.036 (H) |
| 电荷弛豫后 γ | -2.88 (O), -1.64 (H) |
| λ 变化 | 0.05 → 0.055 (仅 +10%) |
| 冻结电荷灵敏度 dγ/dλ | -5.10/(0.055-0.05) = **-1020 rad/Ry** |
| 含弛豫有效灵敏度 | -2.88/(0.055-0.05) = **-576 rad/Ry** |

---

## 四、等效电场换算

SMO 调制势: V_I(r) = λ_I · w_I(r)，r_m = 3 Bohr

```
E_eff = λ / r_m = 0.05 Ry / 3 Bohr
      = 0.025 Hartree / 3 Bohr
      = 0.0083 a.u.
      = 0.68 eV / 1.59 Å
      = 4.3 × 10⁹ V/m
```

### 对标

| 参考 | 电场 |
|------|:---:|
| 水介电击穿 | 0.07 GV/m |
| 铁电材料矫顽场 | 0.1-1 GV/m |
| DFT极化约束典型值 | 1-10 GV/m |
| **本计算 λ=0.05 Ry** | **4.3 GV/m** |

### λ 评估

**合理**: 
- 在铁电DFT计算范围内
- 每原子约束能 λ·Δγ ≈ 3.4 eV 在原子能标内

**问题**: 
- 4.3 GV/m 是水击穿场的 60 倍
- λ仅变10%就能翻转γ 10 rad → 系统对λ极度敏感
- 电荷弛豫立即破坏约束 → λ太弱无法锁定

---

## 五、分支选择分析

### 跨串分支一致性 (iter=1, gdir=3)

| Atom | raw γ 范围 | selected γ 范围 | σ |
|------|-----------|-----------------|:---:|
| O | -86 ~ +6 | -4.85 ~ -5.13 | 0.15 |
| H1 | -68 ~ +2 | -1.58 ~ -3.84 | 1.20 |
| H2 | -69 ~ +7 | -3.67 ~ -5.87 | 1.19 |

跨串一致性 **差**！raw γ 跨串差异高达 ~100 rad，虽然 selected 值经分支选择后凝聚，但 σ > 1.0 表明分支选择在不同 k-string 间选了不同分支。

### 跨迭代分支选择

算法: `select_branch_set(g, prev)` — 搜索单带 ±2π·w_In[n] 位移使值最接近 `prev`

**问题**: 算法以 `prev` (前次迭代的 γ) 为参照，**从不考虑 target γ**。

以 iter=2 atom 0 为例:
- g = -5.05 (rescaled per-atom gamma)
- prev = -5.00 (从 deltap_branch.dat 加载)
- 虽然 |g-prev|=0.05 < π，所以直接接受
- 但如果考虑 target=0.10，应选择离 0.10 最近的分支，而非离 -5.00 最近

### delta=0 的原因

单带位移幅度: 2π × w_In[n][iat]

| n | w_In (atom 0) | 单带位移 2π·w_In |
|:---:|:---:|:---:|
| 0 | ~0.26 | ±1.65 |
| 1 | ~0.15 | ±0.96 |
| 2 | ~2.0 | ±12.6 |
| 3 | ~4.6 | ±28.9 |

单带位移要么太小(±1.65无法桥接Δ=4.95)要么太大(±12.6 overshoot)，没有一个正好。

**结论**: 分支选择算法有两个不足:
1. 只用单带位移，无法处理 Δ 在 π~2π 之间的情况
2. 参照 prev 而非 target，无法利用 target 信息选择最优分支

---

## 六、初始 λ 与 γ 的合理性分析

### 初始 λ = 0.05 Ry

**问题**: 应从 λ=0 开始
- λ=0 对应无约束的物理基态
- 从 0.05 起步立即给系统施加 4.3 GV/m 的有效场
- 使初始 γ 偏离天然值，干扰 SCF 收敛
- 物理上应: 先让 SCF 自然收敛 (λ=0) → 再逐步增大 λ 接近 target

### 初始 γ = +5.05 (O)

**问题**: 这么大是物理的吗？

1. **块体水的天然极化**: 水分子有固有偶极 (~1.85 D)，在周期胞内排列可产生宏观极化
2. **小胞效应**: 5 Bohr 胞太小，水分子被迫以特定取向排列，增强极化
3. **对比孤离水**: 孤离 H2O (15Å 胞) 的 γ ≈ ±1.6 → 块体的 γ 是孤离的 3 倍
4. **符号**: +5.05 意味着极化方向与 lattice vector 同向 → 取决于水的取向

**合理性**: γ=5 对 5 Bohr 胞的块体水物理上可能，但需要验证:
- 与 Wannier90 对标 (未做)
- 用更大胞 (10 Bohr) 测试收敛性
- 检查不同 k 点密度下的 γ

### 分支搜索是否离 target 最近？

**否。** 当前算法 `select_branch_set` 参照 `prev_gamma` (前次迭代的 γ)，完全忽略 `target_gamma`。

**改进方案**: 增加一个选项，在首次分支选择时搜索离 target 最近的分支:
```cpp
// 在 branch-set selection 中:
// 如果是首次迭代(prev_gamma 不存在或不可靠)，
// 搜索使 |g - target| 最小的分支
for (int sign = -k; sign <= k; sign++)
    candidate = g + sign * 2π * w_In;
    if (|candidate - target| < best) accept;
```

这样初始 γ 可以被"对齐"到 target 附近，减少约束负担。

---

## 七、总结

| 问题 | 状态 | 说明 |
|------|:---:|------|
| 内循环收敛 (冻结电荷) | ✅ | γ 从 +5 → -0.05，逼近 target 0.10 |
| 内循环收敛 (含弛豫) | ❌ | 电荷更新导致 γ 反弹到 -2.88 |
| 初始 λ 合理性 | ❌ | 应从 0 开始，不应从 0.05 开始 |
| 初始 γ 合理性 | ⚠️ | 5.05 可能合理但需对标验证 |
| 分支搜索 target-aware | ❌ | 当前只看 prev，不利用 target |
| 单带位移不足 | ⚠️ | 无法处理 Δ 在 π~2π 之间的情况 |
| 约束电场 | ⚠️ | 4.3 GV/m 偏高但仍在 DFT 范围 |

---

## 八、λ=0 梯度下降测试 (nscf=0, cooldown=5)

### 测试配置
```
deltap_corr = 1, deltap_nscf = 0 (内循环关闭)
deltap_lambda_init = 0.0 (从零起步)
deltap_lambda_step = 0.01
```

### 完整收敛轨迹

| iter | λ_O | γ_O | max\|γ-target\| | 备注 |
|:---:|:---:|:---:|:---:|------|
| 1 | 0 → 0.050 | +5.05 | 4.95 | 天然 γ，梯度下降一步更新 λ |
| **2** | **0.050** | **-0.036** | **0.136** | ✅ cooldown 1/5 |
| 3 | 0.050 | -3.28 | 3.38 | ❌ cooldown 2/5, 电荷弛豫漂移 |
| 4 | 0.050 | -3.23 | 3.33 | cooldown 3/5 |
| 5 | 0.050 | -5.07 | 5.17 | cooldown 4/5 |
| 6 | 0.050 | -0.057 | 0.16 | ✅ cooldown 5/5, 电荷弛豫回归 |
| 7 | 0.070 | +2.12 | 2.02 | λ 更新 (cooldown 过期) |
| 8 | 0.070 | +2.16 | 2.06 | cooldown 1/5 |
| 9 | 0.070 | -0.53 | 0.78 | cooldown 2/5 |
| 10 | 0.070 | -0.12 | 0.22 | cooldown 3/5 |
| 11 | 0.070 | -0.20 | 0.30 | cooldown 4/5 |
| 12 | 0.070 | +1.93 | 1.83 | cooldown 5/5, 电荷弛豫漂移 |
| 13 | 0.045 | -2.36 | 2.46 | λ 重新更新 |

### Cooldown 振荡分析

λ 每 5 步更新一次（cooldown 机制）→ 5 步周期内 γ 经历"约束→漂移→回归"循环：

```
λ(t=1) → γ(2)≈-0.04(接近target) → γ(3-5)漂移 → γ(6)≈-0.06(回归)
λ(t=7) → γ(8-12)振荡 → ...
```

**根因**: cooldown=5 设计用于 DeltaSpin（磁矩慢变化），但 DeltaP 的 γ 对电荷极度敏感，5 步间 γ 可漂移 ±5 rad。

### 关键发现: λ=0 起步优于 λ=0.05

| 对比 | λ_init=0.05 + BFGS | λ_init=0 + GD |
|------|:---:|:---:|
| iter=2 γ | -0.050 (内循环优化后) | **-0.036 (自然收敛!)** |
| iter=3 γ | -2.88 (反弹) | -3.28 (反弹) |
| 振荡幅度 | ±3 | ±3 |
| 根本区别 | 内循环 α_opt=9.8e-4 过于保守 | 一步梯度下降直接到位 |

**结论**: 从 λ=0 起步的一步梯度下降，效果等同于 BFGS 内循环（iter=2: max\|γ-target\|≈0.14），但 BFGS 内循环需要昂贵的重对角化（nscf=1 比 nscf=0 慢 3-5 倍）。

---

## 九、发现的 Bug 和缺失功能

### Bug 3: deltap_lambda_mixing 定义但未使用

`input_parameter.h:632`:
```cpp
double deltap_lambda_mixing = 0.0;  // damping factor for lambda update
```

但 `deltap_update_lambda()` 和 `deltap_inner_loop()` 中均未引用此参数。

**应实现**: `λ_new = mixing × λ_gradient + (1-mixing) × λ_old`

### Bug 4: cooldown 不适用于 DeltaP

DeltaP 的 γ 对电荷极度敏感（5 步内漂移 ±5 rad），cooldown=5 导致 λ 总是在 γ 已经漂移后才更新，形成 5 步振荡周期。

**应修改**: cooldown=1 或移除 cooldown，配合 `deltap_lambda_mixing` 平滑 λ 更新。

### Bug 5: 分支选择不感知 target

`select_branch_set()` (line 1501) 以 `gamma_prev` 为参照选择分支，完全不考虑 `gamma_target`。首次迭代时若 branch.dat 值不准，会选择错误分支。

**应修改**: 首次迭代（prev 不可靠时）以 target 为参照选择分支。

---

## 十、改进方案优先级

| 优先级 | 修改 | 预期效果 |
|:---:|------|------|
| P0 | λ_init=0 (默认) | 天然 γ 不受初始约束干扰 |
| P0 | cooldown→1 + 实现 mixing | 消除 5 步振荡周期 |
| P1 | 分支选择 target-aware | 首次正确对齐分支 |
| P2 | 两阶段模式 (先收敛再约束) | 减少电荷-λ 耦合振荡 |
| P3 | 多带位移分支搜索 | 处理 Δ 在 π~2π 之间的分支跳变 |

---

## 十一、cooldown→1 + mixing 实现与测试

### 代码修改

`esolver_ks_lcao.cpp:deltap_update_lambda()`:
```cpp
// 原: if (!dp->inner_loop_cooldown()) { ... dp->start_cooldown(5); }
// 新: 每步更新 λ，应用 mixing 平滑
double mixing = PARAM.inp.deltap_lambda_mixing;
if (mixing == 0.0) mixing = 1.0; // 默认: full step
for (int iat = 0; iat < ucell.nat; ++iat)
    lambda_raw[iat] = lambda[iat] + step * (gamma_I[iat][alpha] - target[iat]);
for (int iat = 0; iat < ucell.nat; ++iat)
    lambda[iat] = mixing * lambda_raw[iat] + (1.0 - mixing) * lambda[iat];
dp_op->set_lambda(lambda);
dp->start_cooldown(1);
```

### 测试: mixing=0.5, λ_init=0, nscf=0

| iter | max\|γ-target\| | g0 (O) | l0 (O) |
|:---:|:---:|:---:|:---:|
| 1 | 0.751 | +0.851 | +0.0008 |
| 2 | 0.146 | +0.014 | +0.0007 |
| 5 | 0.086 | +0.168 | +0.0007 |
| 8 | 0.544 | -0.444 | +0.0001 |
| 13 | 0.045 | +0.055 | +0.0002 |
| 19 | 0.773 | -0.673 | -0.0009 |
| 20 | 0.135 | -0.035 | -0.0011 |

### 测试: mixing=0.1, λ_init=0

| iter | max\|γ-target\| | g0 (O) | l0 (O) |
|:---:|:---:|:---:|:---:|
| 1 | 0.751 | +0.851 | +0.0008 |
| 6 | 0.050 | +0.050 | +0.0006 |
| 14 | 0.158 | +0.258 | +0.0004 |
| 19 | 0.773 | -0.673 | -0.0009 |
| 20 | 0.135 | -0.035 | -0.0011 |

λ 在 ±0.001 Ry 范围（有效电场 ~0.08 GV/m，水击穿场的 1.1 倍），物理上合理。

---

## 十二、target-aware 分支选择实现

### 代码修改

1. `deltap.h`: 添加 `target_gamma_` 成员 + `set_target_gamma()` / `get_target_gamma()`
2. `esolver_ks_lcao.cpp`: `deltap_init()` 中调用 `dp->set_target_gamma(deltap_target_)`
3. `deltap_wannier.cpp`: per-atom 分支选择前添加预对齐步骤

### 关键突破: 归一化权重

**问题**: `w_In_matrix[n][iat]` 是原始 |D|² 值，对块体水可达 ~250/band。
→ 位移幅度 = 2π×250 ≈ 1570 rad，k=±1 就完全 overshoot。

**修复**: 
```cpp
double w_total = Σ_n w_In_matrix[n][iat];  // per-atom total
double w_norm = w_In_matrix[n][iat] / w_total;  // ∈ [0,1]
double shift_amp = 2π × w_norm;  // ∈ [0, 2π]
int max_k = ceil(|g - target| / shift_amp) + 1;  // dynamic k range
```

归一化后位移幅度控制在 0~2π，允许 k=0,±1,±2,... 精确搜索。

### 效果

| 指标 | 修复前 | 修复后 |
|------|:---:|:---:|
| iter=1 γ_O (目标 0.10) | 5.05 | **0.85** |
| iter=1 max\|γ-target\| | 4.95 | **0.75** |
| iter=1 λ_O | 0.025~0.050 | **~0.001** |
| 等效电场 | 2~4 GV/m | **~0.08 GV/m** |

---

## 十三、三修复联合效果总结

| 指标 | 原始 | 修复后 |
|------|:---:|:---:|
| iter=1 max\|γ-target\| | 4.95 | **0.75** |
| λ 范围 (Ry) | 0.05 | **±0.001** |
| 有效电场 | 4.3 GV/m | **0.08 GV/m** |
| 收敛振荡 | ±5 rad | **±0.2 rad** |
| 收敛趋势 | 发散 | **部分收敛 (均值 ~0.15)** |

**剩余问题**: 偶发尖峰（iter=8,19: ~0.5-0.8），电荷-λ 耦合幅度减少 30× 但未消除。

---

## 十四、修改文件清单

| 文件 | 修改 | |
|------|------|:---:|
| `esolver_ks_lcao.cpp` | 调用 init_inner_loop() | +1 |
| `esolver_ks_lcao.cpp` | 两阶段 λ 更新 (drho<阈值 → 一次更新，之后固定) | +10 |
| `esolver_ks_lcao.cpp` | 实现 deltap_lambda_mixing 平滑 | +5 |
| `esolver_ks_lcao.cpp` | λ 更新后 mix_reset() 清除 Broyden 历史 | +5 |
| `esolver_ks_lcao.h` | 添加 deltap_lambda_set_ 标志 | +1 |
| `deltap.h` | target_gamma_ 成员 + setter/getter | +5 |
| `deltap_wannier.cpp` | **多带组合 target-aware 分支选择** (K=3, 7^n_dim 搜索) | +50 |

---

## 十五、多带组合分支选择 — 关键突破

### 问题：单带位移太粗糙

gdir=3 H 原子权重分析：
```
w_norm:  [0.347, 0.230, 0.201, 0.222]
shift_amp: [2.18, 1.44, 1.26, 1.40] rad
```

所有 shift_amp > 1.26 rad。Target=3.0, natural=2.48, 需要 +0.52 rad:
- 任何单带 k=±1 给出 ±1.26+ 位移 → 候选值距 target > 0.74 rad
- 单带搜索失败 → γ 保持在 2.48, 偏差 0.52 rad

### 解法：Bounded exhaustive search over k vectors

```cpp
// 搜索空间: n_dim bands × (2K+1) k values
// n_dim=4, K=3 → 7^4 = 2401 candidates
// 每 k-string × 每 atom × 每 gdir 调用
for kvec in {-3..+3}^n_dim:
    shift = Σ_n kvec[n] × 2π × w_norm[n]
    candidate = γ + shift
    if |candidate - target| < best_dist: accept
```

多带组合示例：
```
band 0 (+1): +2.18
band 1 (-1): -1.44
net shift:   +0.74 → γ = 3.22, dist = 0.22 rad ✓

band 0 (+1): +2.18
band 1 (-1): -1.44
band 2 (-1): -1.26
net:         -0.52 → γ = 1.96, dist = 1.04

band 1 (+1): +1.44
band 2 (-1): -1.26
net:         +0.18 → γ = 2.66, dist = 0.34

最优: band 0 (+1) + band 2 (-1) = +0.92 → γ = 3.40, dist = 0.40
     或 band 0 (+1) + band 1 (-1) = +0.74 → γ = 3.22, dist = 0.22
```

### 效果对比

| 指标 | 单带搜索 | 多带搜索 | 改善 |
|------|:---:|:---:|:---:|
| iter=9 γ 偏差 (H) | 0.55 | **0.002** | 275× |
| λ 大小 | 6.3e-4 Ry | **1.5e-6 Ry** | 420× |
| 最终 drho | 1e-3 (振荡) | **2e-7** | 5000× |
| 能量振荡 | 0.13 Ry | **2e-4 Ry** | 650× |

### 线性插值测试

| Target O | γ_O @update | Δγ_O | λ_O (Ry) |
|----------|:---:|:---:|:---:|
| 4.80 | 4.806 | +0.006 | 6.1e-6 |
| 4.50 | 4.497 | -0.003 | -2.6e-6 |
| 4.00 | 4.000 | 0.000 | -1.4e-7 |
| 3.00 | 3.004 | +0.004 | 4.3e-6 |

多带搜索将 γ 对齐到 target ±0.006 rad，对所有 target 值均有效。
λ 始终极小（约束几乎"免费"），因为分支选择承担了主要工作。

---

## 十六、最终两阶段方法总结

```
Phase 1 (drho > deltap_inner_thr):
  - λ = 0, 无约束 SCF
  - 多带 target-aware 分支选择运行 → γ 对齐到 target (±0.006 rad)
  - SCF 自然收敛, drho → 1e-3

Phase 2 (drho < deltap_inner_thr, 一次触发):
  - 梯度下降: λ = step × mixing × (γ - target)
  - 由于 γ ≈ target (±0.006), λ ≈ 6e-6 Ry (极小)
  - mix_reset() 清除 Broyden 历史

Phase 3 (λ 固定, 继续 SCF):
  - 极小 λ 约束 → 不打扰已收敛电荷
  - drho → 2e-7 (5000× 改善)
  - 能量振荡 2e-4 Ry (650× 改善)
  - 收敛到约束基态
```
