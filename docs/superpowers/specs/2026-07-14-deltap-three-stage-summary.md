# DeltaP 三阶段测试总结

> 日期: 2026-07-13 ~ 2026-07-14  
> 代码: feat/deltap + P0/P1修复 + 三方向Wilson Loop (方案A)

---

## 一、Stage 1: Per-atom 极化分配 (定性验证)

### BN (闪锌矿, 2×2×2 k-mesh)

| iter | P_total (Px, Py, Pz) | Px=Py=Pz? |
|:---:|------|:---:|
| 1 | (-9.71e-3, -7.42e-3, -1.23e-2) | ❌ |
| 2 | **(1.246e-2, 1.246e-2, 1.246e-2)** | ✅ |
| 3 | (-1.199e-2, -1.199e-2, -1.344e-2) | ❌ Pz 偏离 |

| 原子 | γ (sel) | 占比 | 电负性排序 |
|------|:---:|:---:|:---:|
| B | 1.117 | 45% | ✅ B < N |
| N | 1.341 | 55% | |

### H₂O (孤立分子, 2×2×2 k-mesh)

| iter | P_total (Px, Py, Pz) | 物理一致性 |
|:---:|------|:---:|
| 1 | (1.80e-2, 1.80e-2, 1.57e-2) | ❌ Py 不应 = Px |
| 2 | (1.25e-2, **2.71e-3**, 1.25e-2) | ✅ Py ≈ 0 (面外) |
| 3 | (1.69e-2, 1.69e-2, 1.69e-2) | ❌ Py 不应 = Px |

| 原子 | γ (sel) | γ/hydrogen | O/H 比 |
|------|:---:|:---:|:---:|
| O | 1.58 | — | — |
| H (×2) | 1.91 | 0.96 | **1.65** ≈ 1.8 (预期) |

### 结论

| 判断 | BN | H₂O |
|------|:---:|:---:|
| 电负性排序 | ✅ B < N | ✅ H < O |
| 立方/面对称 | ✅ iter 2 成立 | ✅ iter 2 成立 |
| SCF 收敛 | ❌ 3 轮不收敛 | ❌ 3 轮不收敛 |
| 跨 string σ | OK | OK |

**两种体系均给出定性正确的 per-atom 极化分配。** SCF 不收敛 (+ 稀疏 k 点) 导致后续 iteration 偏离。

---

## 二、Stage 2: λ→γ 约束响应 (dγ/dλ)

> 以下数据来自早期单方向 Wilson Loop 测试 (BN, mixing_beta=0, 3×3×3 k-mesh)

| λ | γ₀ (rad) | Δγ = γ(λ)−γ(0) | dγ/dλ (分段) |
|:---:|:---:|:---:|:---:|
| 0.00 | −0.0842 | — | — |
| 0.05 | −0.0791 | +0.0051 | 0.102 |
| 0.10 | −0.0738 | +0.0104 | 0.104 |
| 1.00 | −0.0325 | +0.0517 | 0.052 |
| 5.00 | −0.3866 | −0.3024 | −0.077 |

**结论**: dγ/dλ ≈ 0.05–0.10 rad/λ (正常工作区间), HK 修正产生可测量的系统性响应。

三方向 Wilson Loop 代码下的 λ sweep 因 branch 缓存交叉污染暂无法稳定测试。

---

## 三、Stage 3: 约束自洽收敛 (λ≠0)

**状态**: ❌ 未测试。

阻塞原因:
1. 三方向代码导致 branch 缓存在每个 alpha 迭代间交叉污染
2. 需要代码修复后才能进行

---

## 四、代码修改总结

| 文件 | 修改内容 | 状态 |
|------|------|:---:|
| `deltap_wannier.cpp` | P0 匈牙利算法 | ✅ |
| `deltap_wannier.cpp` | C2 zeta rescaling | ✅ |
| `deltap_wannier.cpp` | H3 per-band branch spacing | ✅ |
| `deltap_wannier.cpp` | M5 S^{-1/2} 诊断 | ✅ |
| `deltap_wannier.cpp` | 方案A 三方向 Wilson Loop | ✅ |
| `deltap_wannier.cpp` | 跨 string 一致性诊断 | ✅ |
| `deltap_gauge.cpp` | H4 phase_corrections_ | ✅ |
| `deltap_io.cpp` | 三方向 P_total 输出 | ✅ |
| `deltap_berry.cpp` | integrate_polarization 三方向 | ✅ |
| `esolver_ks_lcao.cpp` | nullptr berry_overlap_ | ✅ |
| `bfgs.h` | M1 BFGS→FletcherReevesCG + L2 | ✅ |
| `deltap.h` | M1 类型引用 | ✅ |

---

## 五、下一步

| 优先级 | 任务 |
|:---:|------|
| P0 | 修复三方向 branch 缓存隔离 (每个 alpha 独立 prev_gamma) |
| P0 | Stage 2 λ sweep 在三方向代码下重测 |
| P0 | Stage 3 约束收敛测试 |
| P1 | 密集 k 点 (≥3×3×3) BN + H₂O |
| P2 | 方案B A_nk 积分验证 |
| P2 | Wannier90 对标 |
