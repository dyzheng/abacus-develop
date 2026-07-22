# SCF 收敛修复方案 (2026-07-12)

## 根因诊断

**事实**：BN 的 SCF 在 19 次迭代收敛，ETOT 稳定。但 per-atom gamma 在收敛过程中振荡 ±0.3 rad，甚至在 iter=6 发生符号翻转（+0.59→-0.08）。

**时间线**：
- iter=1-5: gamma 从 +0.29 到 +0.59（正值）
- iter=6: gamma 翻转为 -0.08（drho 仍在下降但 gamma 突变）
- iter=7-19: gamma 在 -0.08 到 +0.28 间振荡，逐步衰减

**gate 问题**：`gate = iter > 1 && (drho < inner_thr)`
- `inner_thr=5e-7`：当 drho 第 1 次降至 5e-7 以下时，gate 激活
- 此时 gamma 可能仍在振荡（drho 小 ≠ gamma 稳定）
- inner loop 以当前振荡态 gamma 为起点调整 λ → 随机方向的修正
- 电荷密度继续收敛 + λ 修正 → 耦合振荡

## 修复方案

### 方案 A：降低 charge mixing + 推迟 gate（最小修改）

```
mixing_beta: 0.4 → 0.1（减少电荷更新步幅）
deltap_inner_thr: 5e-7 → 1e-8（gate 更晚触发，电荷更稳定）
scf_nmax: 30 → 50（允许内循环后有足够迭代再收敛）
```

优点：一行参数修改
缺点：gate 仍基于 drho 而非 gamma 稳定性

### 方案 B：基于 gamma 稳定性的 gate（正确方案）

在 gate 条件中增加 gamma 稳定性检查：
```cpp
bool gate = iter > 1 && (dp_dp->inner_loop_triggered()
    || (this->drho > 0 && this->drho < PARAM.inp.deltap_inner_thr
        && dp_dp->gamma_stable(deltap_target_)));
```

`gamma_stable()` 检查最近 3 次迭代的 gamma 标准差 < 0.01 rad。

优点：只在 gamma 真正收敛后才激活内循环
缺点：需新增函数

### 方案 C：分离 charge + lambda 优化阶段（最优但大改）

```
Phase 1 (iter=1..N): 固定 λ=λ_init，纯 SCF 电荷收敛
  → gate 触发条件：scf 收敛（scf_thr 满足）
Phase 2 (iter=N+1): 固定电荷密度，内循环优化 λ → γ→target
  → 修改 Hamiltonian 但不更新电荷
Phase 3 (iter=N+2..M): 固定 λ，允许电荷重新适应
  → 重复 Phase 1-3 直至收敛
```

优点：物理上正确的分离耦合
缺点：需重构 esolver 迭代循环

## 建议

方案 B（gamma 稳定性 gate）是正确性与工作量之间的平衡。实现步骤：
1. DeltaP 类增加 `gamma_history_` 缓存最近 3 次 gamma 值
2. 每次 `compute_gamma_scf` 后更新
3. gate 增加稳定性条件

方案 A 先做快速验证，确认 mixing_beta=0.1 + 更紧 gate 是否能稳定。
