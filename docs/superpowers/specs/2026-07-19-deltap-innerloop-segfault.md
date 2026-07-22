# DeltaP 内循环 Segfault 修复与诊断

> 日期: 2026-07-19

---

## 一、测试计划

1. 重现内循环 segfault
2. 定位根因
3. 修复并验证
4. 添加诊断输出
5. 测试 SCF+内循环行为

---

## 二、测试设置

- **体系**: BN 2×2×2, lattice 3.615 Bohr
- **参数**: ecutwfc=100, LCAO DZP, genelpa, symmetry=-1, mixing_beta=0 (冻结电荷)
- **输入**: 同 `tests/17_DS_DFTU/66_LCAO_DELTAP_BN`
- **额外**: `deltap_corr=1, deltap_nscf=1→5, scf_nmax=3`

---

## 三、结果

### 3.1 Segfault 定位

```
Signal: Segmentation fault (11)
Failing at address: (nil)
Stack:
  FletcherReevesCG::step()  ← segfault here
  ESolver_KS_LCAO::hamilt2rho_single()
```

### 3.2 根因

`esolver_ks_lcao.cpp:646`:

```cpp
std::vector<double> lam_trial;                         // BUG: 空 vector
bfgs.step(residual, inner, lam_trial, bfgs_converged); // step() 写 lam_trial[i] → NULL deref
```

`FletcherReevesCG::step()` 内部使用 `lambda_out[i] = initial_lambda_[i] + dnu_[i]` (operator[]), 假设 `lambda_out` 已预分配 `n_dim_` 个元素。但 `lam_trial` 声明为空 vector。

### 3.3 修复

```cpp
std::vector<double> lam_trial(ucell.nat);  // FIX: pre-allocate
```

### 3.4 修复后测试

| 配置 | 结果 |
|------|------|
| `deltap_nscf=1, mixing_beta=0` | 通过, 3 iters 无 crash |
| `deltap_nscf=5, mixing_beta=0` | 通过, 3 iters 无 crash |
| `deltap_nscf=5, mixing_beta=0.4` | 通过, 10 iters 无 crash |

### 3.5 诊断输出 (nscf=5, mixing_beta=0.4)

```
[DeltaP] inner loop start: nscf=5 l0=-1.060e-02 l1=-2.347e-02 rms=1.7377e+00
[DeltaP]   inner=0 rms=1.6091e+00 l0=-9.333e-03 l1=-2.194e-02
[DeltaP]   result: g0=1.4460e+00 g1=1.7422e+00 alpha_opt=2.855e-03
[DeltaP]   inner=1 rms=1.6009e+00 l0=-4.402e-03 l1=-1.600e-02
...
[DeltaP] inner loop done: final l0=2.151e-02 l1=1.654e-02
```

### 3.6 收敛行为分析

- **内循环 rms 几乎不减小**: 2.457 → 2.458 (nscf=5, mixing_beta=0)
- **HK 修正极弱**: dγ/dλ≈0.02 rad/λ → 以 λ≈0.01 只能移动 γ 约 0.0002 rad
- **自适应步长快速衰减**: α_opt=2.86e-3 → 8.27e-4
- **分支跳变**: inner=4 处 g1 从 1.97 跳到 4.30 (branch flip)
- **结论**: 内循环不 crash, 但收敛无效——HK 响应太弱

---

## 四、分析

### 4.1 HSolverLCAO 可重入性 — 已排除

HSolverLCAO 只有两个 const 成员, solve() 修改的是参数对象 (psi, pelec, dm, chr)。第二次调用完全安全。DeltaSpin 也每次创建新对象但对象本身可重入。

### 4.2 相位不确定性 — L1 match freeze 保护

`compute_wannier_polarization` 在 iter=1 保存 match 到 `deltap_match.dat`, iter≥2 加载复用。内循环所有 trial λ 使用相同 match, 相位不确定性不影响 gamma 一致性。

### 4.3 H 增量更新 — 已确诊正常

`dp_op->set_lambda()` 置 `dp_hr_done=false`, `updateHk → contributeHR()` 自动计算 dλ×pre_hr 追加到 HR。无需 `refresh()`。

### 4.4 确定性测试 (P1-②)

3 次无缓存全新运行, P_total 完全不重现:

| Run | Px | Py | Pz |
|-----|----|----|-----|
| 1 | -7.34e-03 | -1.44e-04 | -1.44e-04 |
| 2 | 3.51e-03 | 3.55e-02 | -3.79e-03 |
| 3 | -1.44e-04 | 3.51e-03 | -3.79e-03 |

且暴露跨方向累加器共享 (Py≡Pz in Run1, 跨 run 值迁移)。B16 确定性不通过, 与 07-13 列出的 "max|Δγ|<10⁻¹⁰" 判据相差 ~14 个数量级。

---

## 五、下一步

1. **B16 确定性**: 先修累加器共享 (Bug 1), 再测试跨方向污染 (Bug 2), 最后验证无缓存全新运行确定性
2. **HK 响应**: dγ/dλ 太小 (~0.02 rad/λ), 与 07-13 数据 (0.1-0.3) 有 5-15x 差距, 需要独立有限差分标定
3. **内循环有效性**: 即使修好 λ 更新, 小 λ 下 HK 修正不足以驱动 γ 收敛, 需要更大的 λ 步长或更强的响应机制
