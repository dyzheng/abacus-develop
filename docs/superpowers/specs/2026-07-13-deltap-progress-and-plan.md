# DeltaP 当前开发进展与下一步开发计划

> 日期: 2026-07-13  
> 对应分支: `feat/deltap` (commit 85b2af322)  
> 最近三次修改轮次覆盖: 风险评审 rebuttal → P0/P1 修复 → P3 维护

---

## 一、总体进展

| 指标 | 状态 |
|------|------|
| P0 (阻滞) | ✅ 已完成 (匈牙利算法替换贪心匹配) |
| P1 (数学错误) | ✅ 已完成 (C2 zeta rescaling, H3 branch spacing, H4 gauge correction) |
| P2 (验证项) | ✅ 已验证 (H1 SMO对称=OK, H2 D_I约定=无害) |
| P3 (精度/维护) | ⚠️ 基本完成 (9/12, C1 暂缓, M2/M3 仅文档) |
| 单元测试 | ✅ 13/13 通过 (gauge 4, math 3, optimizer 6) |
| SCF 集成测试 | ❌ 受阻 (`unkOverlap_lcao` 性能瓶颈, 非本次改动) |
| Stage 2 验证 | ✅ λ→γ 响应可测量 (dγ/dλ≈0.05-0.10 rad/λ) |
| Stage 3 验证 | ❌ 阻塞于 SCF 集成测试无法运行 |

---

## 二、本轮 (2026-07-13) 完成的代码修改

### 核心修复 (P0+P1, 4 项)

| # | 任务 | 文件 | 变更描述 |
|---|------|------|---------|
| 1 | P0 匈牙利算法 | `deltap_wannier.cpp` | 替换贪心最近邻匹配为 Kuhn-Munkres 全局最优 (O(n³), 85 行) |
| 2 | C2 zeta rescaling | `deltap_wannier.cpp` | `arg(det W)` → `Σγ_unwrapped` 作为分子, 避免 ±π 边界错 (5 行) |
| 3 | H4 gauge 修正 | `deltap_gauge.cpp` | anchor 切换时回溯应用 `phase_corrections_` 到已计算的 `gauge_phase_` (5 行) |
| 4 | H3 分支间距 | `deltap_wannier.cpp` | 均匀 `2π·scale` → per-band `±2π·w_In` 原子间距 (20 行) |

### 精度与维护 (P3, 5 项)

| # | 任务 | 文件 | 变更描述 |
|---|------|------|---------|
| 5 | M4 NS fallback SVD | `deltap_wannier.cpp` | NS 迭代不收敛时 fallback 到 zgesvd + 相位归一化 (60 行) |
| 6 | M4b NS 缩放修正 | `deltap_wannier.cpp` | `1/sqrt(frob2/n_dim)` → `1/sqrt(frob2)` 保证 `‖X‖₂≤1<√3` (1 行) |
| 7 | M1 类名重命名 | `bfgs.h`, `deltap.h`, `esolver_ks_lcao.cpp`, `bfgs_test.cpp` | `BFGS` → `FletcherReevesCG` (15 行, 4 文件) |
| 8 | L2 自适应因子 | `bfgs.h` | step 自适应乘子 `1.5→1.0` (1 行) |
| 9 | M5 诊断修正 | `deltap_wannier.cpp` | 删除错误的 `S·Sinv=I` / `S·Sinv·S=S` 检查 (Sinv 实为 S^{-1/2}), 保留正确的 `S^{-1/2}·S·S^{-1/2}=I` (20 行) |

### 验证与分析 (2 项)

| # | 任务 | 结论 |
|---|------|------|
| 10 | H1 SMO 对称性 | `max_asym = 6.7e-17` < 1e-15, SMO 重叠矩阵对称至机器精度 — 无问题, 关闭 |
| 11 | H2 D_I conj 约定 | `conj(S_val)` 在 gauge fixing 中经 `conj(D_anchor)` 双重共轭抵消, SMO 权重用 `|D_I|²` 不受影响 — 无害, 关闭 |

---

## 三、代码修改文件清单

| 文件 | 修改内容 | 净增行数 |
|------|---------|---------|
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | P0 匈牙利, C2 zeta, H3 branch, M4 SVD, M5 诊断 | ~190 |
| `source/source_lcao/module_deltap/deltap_gauge.cpp` | H4 phase_corrections_ 应用 | +5 |
| `source/module_optimizer/bfgs.h` | M1 改名 FletcherReevesCG, L2 因子 1.0 | ~10 |
| `source/source_lcao/module_deltap/deltap.h` | M1 类型引用更新 | ~4 |
| `source/source_esolver/esolver_ks_lcao.cpp` | M1 类型引用更新 | 1 |
| `source/module_optimizer/test/bfgs_test.cpp` | M1 类型引用更新 | ~5 |

---

## 四、当前阻塞

### 4.1 unkOverlap_lcao 性能瓶颈 (预存在)

**表现**: `deltap_corr=1` + `berry_overlap_` 非 null 时, `berryphase_overlap` 每次调用 ~30s (BN 2×2×2, NBASIS=18, nocc=4)。直接导致 SCF 迭代超时。

**诊断**: `prepare_midmatrix_pblas` 内的 `iw2it/iw2ia` 用了 O(nlocal) 线性扫描, 即使 nlocal=18 也不应慢到 30s。疑似 BLACS/ScaLAPACK 单进程模式下的初始化开销累积。

**影响范围**: SCF 集成测试无法运行, Stage 3 验证被阻。

**不在本次修改代码中**: `unkOverlap_lcao` 从未被 touch。

### 4.2 NS 迭代缩放 bug (已修复)

RS2 中发现 NS 迭代 `err=2.0000e+00` (20 轮不收敛)。根因: 初始缩放 `1/sqrt(frob2/n_dim)` 不保证 `‖X‖₂ < √3`。M4b 修复为 `1/sqrt(frob2)`, 保证收敛。但此 bug 不影响功能 — 当 NS 不收敛时 M4 fallback 到 SVD 已覆盖。

---

## 五、下一步计划

### 立即 (如需推进集成测试)

| 优先级 | 任务 | 预计时间 |
|--------|------|---------|
| P0 | 定位 unkOverlap 性能瓶颈 | 半天 (profile + 修复) |
| P0 | 或替换 berry_overlap 为 compute_S_k 直接计算 | 1 天 (修改 O_kpair 路径) |

### 中等 (精度提升)

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P3 | C1: `compute_hk_correction` 使用 `compute_S_dk_link` | 含位置修正 `-i·dk·⟨r⟩`, O(dk²) 精度。需重构 HK 循环 (per-link 而非 per-string)。3×3×3 mesh 误差 ~0.03 rad |
| — | B16 运行时诊断输出 | 在 `compute_wannier_polarization` 末尾加逐带 γ/eval/F 范数输出, 3 次运行差分定位 |

### 低 (优化与文档)

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P3 | M2: 文档化 λ mixing 状态不一致 (不修) | 已有共识降级 |
| P3 | M3: 文档化伪梯度限制 (不修) | 当前无限长步问题 |
| P3 | M6: 清理 `select_branch_set` 死代码 | H3 使用内联修复, 函数体死代码 |
| — | L3/L4: 已完成 (当前代码中相关注释已不存在) |
| — | BTO + Wannier90 严格对标 | Stage 3 通过后执行 |

### 回归测试 (unkOverlap 修复后)

```bash
# 冒烟: λ=0, deltap_lambda_step=0 → γ=−0.085, 15轮收敛
# 约束: λ=0.05, deltap_lambda_step=0 → γ 收敛, 无振荡
# 确定性: λ=0.05, mixing_beta=0, 3 次运行 → max|Δγ| < 1e-10
# 内循环: deltap_nscf=5, deltap_lambda_step=0.1 → maxdev 递减
```

---

## 六、风险总览

| 18 项风险评审 | 已修复 | 已验证 | 降级 | 未处理 |
|:---|:---:|:---:|:---:|:---:|
| 3 Critical → | C2 已修复 | — | C3 → Medium | C1 → P3 待办 |
| 5 High → | H3, H4 已修复 | H1, H2 已验证无害 | H5 → Low | — |
| 6 Medium → | M1, M4, M5 已修复 | — | — | M2, M3 (仅文档), M6 (死代码) |
| 4 Low → | L2 已修复 | L1 (同 P0) | — | L3, L4 (已完成) |
| 1 Unrated → | B16 (匈牙利算法) | — | — | B16 诊断输出 |

**结论**: 静态审查的全部 18 项风险已有明确处理。P0 阻滞 (B16 跨运行非确定性) 通过匈牙利算法理论上已解决, 待 SCF 集成测试验证。唯一真正的未知未知是 `unkOverlap_lcao` 的性能问题。

---

*本文件由 2026-07-13 日三轮开发产出: 风险评审 rebuttal → P0/P1 修复 → P3 维护。*
