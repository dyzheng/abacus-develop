# DeltaP 当前状态总结与 TODO 列表

> 综合 `risk-review-rebuttal.md` 和 `risk-review-evaluation.md` 的共识产出  
> 日期: 2026-07-13

---

## 一、当前状态

### 1.1 功能验证状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Stage 1 (Wannier90) | ⚠️ 部分完成 | 水分子 O/H Pz≈1.8 定性合理；BTO per-atom Pz 物理正确但 W90 流程未跑 |
| Stage 2 (λ→γ 响应) | ✅ 通过 | `dγ/dλ ≈ 0.1-0.3` per unit λ，HK 修正产生系统性可测量响应 |
| Stage 3 (内循环约束收敛) | ❌ 阻滞 | `λ=0` SCF 正常收敛 (15 轮)，`|λ|≥0.01` gamma 振荡不收敛 |
| Branch 一致性 (B16) | ❌ 阻滞 | 冻结电荷 (相同 ψ) → 3 次运行 gamma 不同 |

### 1.2 已实施的代码修复

| 修复 | 文件 | 效果 |
|------|------|------|
| k-index 非对称映射 (B14) | `deltap_wannier.cpp` | `symmetry=-1` 启用全网格 k 索引 |
| SVD U·V† 列主序 (P1) | `deltap_wannier.cpp` | 修正 polar factor 为正确的 U·V† |
| zgeev 特征值排序 (P4) | `deltap_wannier.cpp:675` | j=0 处按 arg 排序确定初始顺序 |
| Newton-Schulz 替换 SVD | `deltap_wannier.cpp:557-632` | 消除 zgesvd 相位歧义 |
| HR 延迟 (iter=1 不施加) | `deltap_lcao.cpp:43-46` | 消除 iter=1 Hamiltonian 突变 |
| HK 缓存 (λ 不变不重算) | `esolver_ks_lcao.cpp:845-862` | 消除 ψ-HK 自洽循环 |
| λ mixing + cooldown | `esolver_ks_lcao.cpp:620-681` | 内循环 λ 稳定性 |
| Branch 永久 save/load | `deltap_wannier.cpp:298,1124` | 跨运行分支追踪，首次运行锁定 |

### 1.3 已确认的问题

- **B16 (P0 阻滞)**：冻结电荷 (相同 ψ) 下 gamma 跨运行不同 → Wilson loop 本征值计算存在非确定性，来源待定位（见 §2.1）
- **C2 (P1 Critical)**：zeta rescaling 用 `arg(det W)` 作为参照值错误，当 unwrapped Σγ 包含附加 2πk 时 scale 错误缩放 per-atom γ
- **H4 (P1 High)**：`phase_corrections_` 计算后从未应用到 `gauge_phase_`
- **H3 (P1 High)**：内联分支选择用均匀间距 `2π·scale` 而非 per-band `2π·w^I_n·scale`
- **C1 (P3 High)**：`compute_S_dk` 缺少位置修正 `-i·dk·⟨r⟩`，O(dk) 量级精度损失

### 1.4 已确认不成立的风险点

| 风险 | 原评级 | 降级理由 |
|------|--------|---------|
| H5 (fmod 丢失累积相位) | High → Low | fmod 是合理的 2π 约化；匹配歧义由贪心算法引起，应由 L1 匈牙利算法解决 |
| C3 (psi-lambda 不一致) | Critical → Medium | 与电荷 mixing 同质近似；γ 对 ψ 鲁棒 (实测 Δλ=0.1 改变 γ 仅 0.01 rad) |

### 1.5 待验证项

| 项目 | 验证方法 |
|------|---------|
| H1: SMO 重叠矩阵非对称 | 检查运行输出中 `max_asym` 是否非零 |
| H2: D_I conj 约定 | 搜索 ABACUS Bloch 相位约定 (`kvec_d_R`) |

### 1.6 保留的技术分歧

C1 位置修正阶数：评估方认为 O(dk)（提供 overlap 领先虚部），rebuttal 认为 O(dk²)。**但双方对结论一致**：降为 High，当前 dk≈0.33 的 3×3×3 mesh 上误差 ~0.03 rad，可感知但不阻滞。

---

## 二、TODO

### P0: 定位 B16 非确定性根因 (唯一阻滞)

#### 2.1 运行时诊断 [诊断]

在 `compute_wannier_polarization` 末尾添加诊断输出，多次运行比较：

```cpp
// 输出 (1) Wilson loop 矩阵 Frobenius 范数 → 定位非确定性是否在矩阵构建阶段
// 输出 (2) 逐带 gamma_unwrapped → 确认跨运行差异出现在哪个带
// 输出 (3) 逐 eval → 确认是对角化阶段还是匹配阶段
```

**验证**: 3 次冻结电荷运行 (相同 ψ)，比较诊断输出差异所在位置。

**可能根因与对应修复方向**:

| 根因 | 诊断表现 | 修复 |
|------|---------|------|
| zgeev 特征值返回顺序不保证 | evals 跨运行顺序不同 | 固定排序后再匹配 |
| 编译器浮点重排序 (`-O3`) | W 矩阵 norm 相同但 evals 微量不同 | 降低对应函数优化等级 |
| NS 迭代浮点非结合 | W norm 微量不同 | 增加 NS 收敛精度 |
| 贪心匹配对微量差异敏感 | gamma_unwrapped 跨带跳跃 | L1 匈牙利算法全局匹配 |

#### 2.2 实施确定性保障 [代码]

- [ ] 在 `compute_eigenvalues_unitary` 中对 zgeev 输出按 arg 排序
- [ ] 实现匈牙利算法替换贪心匹配 (L1, 见 §2.8)
- [ ] 在 NS 迭代后添加 `err < tol` 收敛检查，未收敛 fallback SVD
- [ ] 冻结电荷 3 次运行验证 gamma 完全一致

**验证**: 3 次冻结电荷运行 `max|Δγ| < 1e-10`

---

### P1: 修复已确认的数学/逻辑错误

#### 2.3 C2: Zeta rescaling 改用 unwrapped sum [代码]

**位置**: `deltap_wannier.cpp:1000-1016`  
**修改**: `scale = arg(det W) / Σγ_raw` → `scale = Σγ_unwrapped / Σγ_raw`

```cpp
// Before (line 1015):
double scale = gamma_correct / gamma_raw_sum;
// After:
double gamma_unw_sum = 0;
for (int n = 0; n < n_dim; ++n) gamma_unw_sum += gamma_unwrapped[n];
double scale = gamma_unw_sum / gamma_raw_sum;
```

**验证**: 检查 unwrapped sum ≈ arg(det W) mod 2π 一致性

#### 2.4 H4: 应用 phase_corrections_ [代码]

**位置**: `deltap_gauge.cpp:134-148`  
**修改**: 在 anchor 切换后将 `phase_corrections_` 回溯应用到已计算的 `gauge_phase_`:

```cpp
// After line 129 (anchor switch detected):
for (int jj = 0; jj < j; ++jj)
    gauge_phase_[jj][n] *= std::polar(1.0, -delta_phi);
```

**验证**: 检查 anchor 切换前后的 gauge phase 连续性

#### 2.5 H3: 启用 select_branch_set [代码]

**位置**: `deltap_wannier.cpp:1023-1052`  
**修改**: 用 `select_branch_set` 替换内联分支选择，使用 per-band `w^I_n` 权重

**验证**: 粗 mesh (2×2×2) 上比较修复前后 per-atom γ

---

### P2: 验证后修复

#### 2.6 H1: 验证 SMO 重叠矩阵对称性 [验证 → 修复]

- [ ] 运行水分子 BTO 算例，检查 `max_asym` 输出
- 若 `max_asym > 1e-15`：显式对称化 `smo_overlap_`
- 若 `max_asym ≈ 0`：关闭此风险点

#### 2.7 H2: 验证 D_I 共轭约定 [验证 → 修复]

- [ ] 搜索 `kvec_d_R` / `exp_ikR` 确认 ABACUS 的 Bloch 相位约定
- 若正号：去掉 `conj(s_val)` → 改用 `s_val` 直接相乘
- 若负号：确认当前 `conj` 正确，关闭风险点

---

### P3: 精度与维护性

#### 2.8 L1: 匈牙利算法全局匹配 [代码]

**位置**: `deltap_wannier.cpp:682-718`, `deltap_wannier.cpp:749-777`  
**修改**: 替换两处贪心最近邻匹配为匈牙利算法 (Munkres)，保证全局最优

**验证**: 简并点测试 (2 个本征值 arg 差 < 1e-6) → 跨运行匹配一致

#### 2.9 C1: 使用 compute_S_dk_link [代码]

**位置**: `deltap_wannier.cpp:1171-1175`  
**修改**: `compute_hk_correction` 改调 `compute_S_dk_link` (含位置修正)

**验证**: 比较修复前后 dγ/dλ (预期增量 ~0.03 rad)

#### 2.10 M1-M6, L2-L4: 维护性 [代码]

| 项目 | 操作 |
|------|------|
| M1 | 重命名 BFGS → FletcherReevesCG |
| M2 | 文档化 lambda mixing 状态不一致 (不修，见 §1.4 C3 降级) |
| M3 | 文档化伪梯度限制 (不修，当前无拉不动的问题) |
| M4 | NS 迭代添加收敛检查 + fallback SVD |
| M5 | 删除或修正错误诊断公式 |
| M6 | 删除 `select_branch_set` 死代码 (已在 H3 中启用) |
| L2 | 步长自适应因子 1.5 → 1.0 |
| L3 | 修正 `sum_I w_In = 1` 注释 |
| L4 | 文档化 w_eff 为预条件器 |

---

## 三、验证回归测试

完成 P0-P1 后运行：

```bash
# 冒烟测试：λ=0 (基线)
deltap_lambda_init=0 deltap_lambda_step=0.0 → SCF 15轮收敛, γ=−0.085

# 临界测试：λ=0.05, HR延迟+HK缓存
deltap_lambda_init=0.05 deltap_lambda_step=0.0 → gamma收敛, 无振荡

# 确定性测试：λ=0.05, 冻结电荷, 3次运行
max|Δγ| < 1e-10

# 功能测试：内循环, 约束目标
deltap_nscf=5 deltap_lambda_step=0.1 → 内循环激活, maxdev递减
```

---

## 四、关键文件清单

| 文件 | 涉及修复 |
|------|---------|
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | C2, H3, C1, L1, M4, B16 诊断 |
| `source/source_lcao/module_deltap/deltap_gauge.cpp` | H4 |
| `source/source_lcao/module_deltap/deltap_overlap.cpp` | H1 (验证后) |
| `source/source_lcao/module_deltap/deltap_berry.cpp` | H2 (验证后) |
| `source/source_esolver/esolver_ks_lcao.cpp` | HK 缓存 (已完成) |
| `source/source_lcao/module_operator_lcao/deltap_lcao.h` | H4 相关 |
| `source/source_lcao/module_optimizer/bfgs.h` | M1 (重命名) |
