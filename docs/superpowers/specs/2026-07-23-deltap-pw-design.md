# DeltaP PW 基组实现 — 设计文档

**日期**: 2026-07-23
**参考**: DeltaSpin PW/LCAO 架构分析

---

## 1. DeltaSpin 的 PW/LCAO 分层架构

DeltaSpin 通过**共享核心 + 基组特定算子**实现了 PW 和 LCAO 的统一约束 DFT 框架：

```
                   SpinConstrain<TK> (Singleton, shared)
                     - lambda, target_mag, Mi, constrain
                     - BFGS optimizer (Polak-Ribiere CG)
                     - lambda_loop (shared)
                   /                              \
          LCAO path                                PW path
          /                  \                    /              \
   deltaspin_lcao.cpp   dspin_lcao.cpp    deltaspin_pw.cpp    op_pw_proj.cpp
   (thin ESolver entry) (contributeHR)  (thin ESolver entry)  (non-local operator)
                             |                                       |
          H += λ·|α><α| (NAO HR)              Hψ += |α>·(λ·<α|ψ>) (PW potential)
                             |                                       |
                    cal_mw.cpp (Mi calculation, shared dispatcher)
                    /                              \
            cal_mi_lcao()                    cal_mi_pw()
            Tr[dmR · pre_hr]                Σ w · <ψ|Pσ|ψ>
```

**关键设计原则**：
- **共享层**：约束状态、BFGS 优化器、λ 更新逻辑（`spin_constrain.h/cpp`）
- **基组特定层**：哈密顿修正算子（LCAO: `contributeHR`, PW: `OnsiteProj::act`）、态密度计算（LCAO: `cal_mi_lcao`, PW: `cal_mi_pw`）

---

## 2. DeltaP 现有架构 vs 目标架构

### 2.1 现有架构（仅 LCAO）

```
esolver_ks_lcao.cpp
  ├── deltap_init()           ← 初始化（仅 LCAO，创建 DeltaPOperator<LCAO>）
  ├── deltap_inner_loop()     ← BFGS 内层（共享，但仅 LCAO 路径调用）
  ├── deltap_update_lambda()  ← 两阶段 λ 更新（共享）
  └── iter_finish()           ← γ 计算 + λ 更新 + 输出
      └── dp->compute_wannier_polarization()  ← Wilson loop (LCAO only)

deltap_wannier.cpp             ← 核心 Wilson loop + per-atom 分解（仅 LCAO）
deltap.cpp                     ← berry_connection 路径（仅 LCAO）
deltap_berry.cpp               ← Berry 联络算子（仅 LCAO）
deltap_gauge.cpp               ← 规范固定（仅 LCAO）

deltap_lcao.cpp/.h             ← DeltaPOperator<LCAO>::contributeHR()
                                  (pre_hr + lambda to NAO HR)
```

### 2.2 目标架构（PW + LCAO 统一）

```
esolver_ks_pw.cpp   ← 新增 PW ESolver 中的 DeltaP 调用
  ├── deltap_init_pw()           ← PW 初始化（OnsiteProjector）
  ├── deltap_inner_loop()        ← 复用共享 BFGS
  ├── deltap_update_lambda()     ← 复用共享 λ 更新
  └── iter_finish()              ← γ 计算 + λ 更新 + 输出

deltap_pw.cpp                   ← 新增：PW 基组的 per-atom γ 计算
  ├── compute_gamma_pw()         ← PW 版 Berry 相位计算
  └── accumulate_gamma()         ← 从 becp 累加 per-atom γ

op_pw_proj_deltap.cpp           ← 新增：PW 非局部约束势
  └── OnsiteProj::cal_ps_deltap() ← ps = lambda * becp (类比 DeltaSpin 的 cal_ps_delta_spin)

复用（不改）：
  deltap.h/cpp 约束矩阵逻辑
  deltap_gauge.cpp 规范固定
  esolver_ks_lcao.cpp 中的两阶段/BFGS/约束矩阵/输出格式
```

---

## 3. TODO 分解

### Phase A：哈密顿修正算子（PW 非局部势）

**目标**：在 PW 基组中施加 λ 约束力

**DeltaSpin 的 PW 做法**（`op_pw_proj.cpp`）：
```cpp
// 1. 计算 becp = <alpha|psi>
onsite_p->update_becp(psi_in);
// 2. 施加 Pauli 矩阵: ps = lambda * becp
cal_ps_delta_spin(npol, nbands);
// 3. Hψ += |beta> * ps (非局部势)
add_onsite_proj(hpsi_out);
```

**DeltaP 需要**：
```cpp
// 1. 计算 becp = <alpha|psi>  (same as DeltaSpin, reuse OnsiteProjector)
// 2. ps = lambda[iat] * w_eff[n] * becp (per-band per-atom weight)
// 3. Hψ += |beta> * ps (same add_onsite_proj)
```

**关键差异**：DeltaSpin 的 λ 是 per-atom 3D 向量，作用在自旋空间。DeltaP 的 λ 是 per-atom 标量，作用在每个 k-point 的每个带上（通过 `w_eff[n]` 加权）。

**TODO A.1**：在 `OnsiteProj` 中添加 `cal_ps_deltap()` 方法
- 输入：`lambda[iat]`, `w_In[n][iat]`（per-band per-atom 投影权重）
- 输出：`ps[n][spin] = λ_eff[n] × becp[n][spin]`
- `λ_eff[n] = Σ_iat lambda[iat] × w_In[n][iat]`（与 LCAO 中 `w_eff[n]` 一致）

**TODO A.2**：将 `OnsiteProj` 注册到 PW Hamiltonian operator chain
- 类比现有 `DeltaPOperator<OperatorLCAO>` 的注册方式
- 参数 `deltap_switch=1 && deltap_corr=1` 时激活

**代码量**：~40 行（在现有 `op_pw_proj.cpp` 中扩展）

---

### Phase B：PW 基组 per-atom γ 计算

**目标**：在 PW 基组中计算每原子的 Berry 相位 γ_I

**DeltaSpin 的 PW 做法**（`cal_mi_pw()`）：
```cpp
for each k-point:
    onsite_p->tabulate_atomic(ik);       // 设置原子投影 |alpha>
    onsite_p->overlap_proj_psi();        // 计算 becp = <alpha|psi>
    for each band:
        accumulate_Mi_from_becp();       // 从 becp 提取磁矩
```

**DeltaP 需要**：由于 Berry 相位涉及 k 空间的非局域量，比磁矩计算复杂：

**方案 B1**：Wannier 函数中心（最直接）
- 构建 Wannier 函数 `|w_nR> = (1/N_k) Σ_k e^{-ikR} |ψ_nk>`
- 计算 Wannier 中心 `r_w = <w_0R|r|w_0R>`
- 分解到原子：γ_I = -(2π/a) × Σ_n w_In[n][I] × r_w[n]
- **问题**：Wannier 函数构建需要 gauge fixing，与现有 `deltap_gauge_mode` 相关

**方案 B2**：Berry 联络 + 原子投影权重（最接近 LCAO 路径）
- PW 下的 Berry 联络：`A_n(k) = i <u_nk|∇_k|u_nk>`（用有限差分）
- 分解到原子：`A_nI(k) = |<alpha_I|psi_nk>|^2 × A_n(k) / Σ_J |<alpha_J|psi_nk>|^2`
- 积分到 γ：γ_I = (1/Ω_BZ) ∫ A_nI(k) dk
- **问题**：PW 的 ∇_k 有限差分需要额外 k-point 数据

**方案 B3**：总 Berry 相位 + 后分解（最简单）
- 用现有 ABACUS Berry 相位功能计算**总 γ**（不分解）
- 后处理：γ_I = (w_In_weight) × γ_total（从 pre-computed 原子投影分解）
- **限制**：牺牲 per-atom 精度，但保留了约束的核心能力
- **推荐为 Phase B 初版**

**TODO B.1**（方案 B3）：
- 在 PW ESolver 中调用 ABACUS 现有 Berry 相位计算（若存在）
- 或实现总 γ 的 Resta-Z / 有限差分计算
- 从原子投影 `|becp|^2` 计算 per-atom 权重
- `gamma_I = gamma_total × w_I / Σ w_J`

**TODO B.2**（方案 B2，长期）：
- 实现 PW 的 Berry 联络有限差分 `A = i<u_k|u_{k+dk}> / dk`
- 在 Atomics projector 基上分解 A_I
- 积分得到 γ_I

**代码量**：B3 ~100 行；B2 ~300 行

---

### Phase C：与 LCAO 路径的代码复用

**目标**：最大化共享 esolver 层逻辑

**现状**：
- `esolver_ks_lcao.cpp` 中的 DeltaP 代码包含大量 LCAO 特定逻辑（`dp_op`, `hamilt_lcao`, `compute_wannier_polarization`）
- `esolver_ks_pw.cpp` 是完全独立的 PW ESolver

**改造方案**（类比 DeltaSpin）：
- 将 λ 更新、两阶段模式、内层 BFGS、约束矩阵、输出格式等移到**独立函数**中
- 不被 ESolver 基类绑定，接受抽象接口

**TODO C.1**：重构 `deltap_update_lambda` → 独立函数
```cpp
void deltap_update_lambda(
    const std::vector<double>& gamma_I,   // 基组无关的 γ 值
    const UnitCell& ucell,
    DeltaP* dp,                             // 基组无关的 DeltaP 对象
    // ... 其他参数
);
```

**TODO C.2**：重构 `deltap_inner_loop` → 独立函数
- 接受一个 `apply_lambda_and_solve(lambda, skip_charge)` 回调
- LCAO: 回调调用 `dp_op->set_lambda()` + `HSolverLCAO::solve()`
- PW: 回调调用 `onsite_proj->set_lambda()` + `HSolverPW::solve()`

**TODO C.3**：在 `esolver_ks_pw.cpp` 中注册 DeltaP
- 在 `before_all_runners` / `iter_finish` 中插入 DeltaP 调用
- 参考 `deltaspin_pw.cpp` 的实现模式

**代码量**：C1-C2 ~120 行（重构）；C3 ~80 行（新增）

---

### Phase D：原子投影器基础设施

**目标**：确保 PW 基组中有可用的原子投影

**现状**：
- `OnsiteProjector` 已存在（用于 DeltaSpin + DFT+U）
- `becp = <alpha|psi>` 计算已实现
- 投影器参数（原子类型、轨道数）从赝势读取

**TODO D.1**：验证 OnsiteProjector 与 DeltaP 的 SMO 投影的兼容性
- DeltaP LCAO 使用 SMO 权重 `w_In`（从 Löwdin 正交化后的全电子轨道投影）
- PW 使用 `becp`（从赝势的 beta 函数投影）
- 两者是**不同基组投影**——需要验证一致性

**TODO D.2**：计算 w_In 等效量
- LCAO: `w_In[n][iat] = Σ_lm |D_I[iat][lm][n]|^2`
- PW: `w_In[n][iat] = Σ_lm |becp_lm[n]|^2 / Σ_J Σ_lm |becp_lm_J[n]|^2`
- 实现函数 `compute_w_In_from_becp()`

**代码量**：~50 行

---

## 4. 实施优先级

| Phase | 任务 | 代码量 | 依赖 | 优先级 |
|-------|------|--------|------|--------|
| **D** | 原子投影器基础设施 | ~50 | — | 先决条件 |
| **B** | per-atom γ 计算 (B3 方案) | ~100 | D | 核心 |
| **A** | 哈密顿修正算子 | ~40 | D | 核心 |
| **C** | 代码复用重构 | ~200 | A, B | 整合 |
| **集成测试** | BN + H2O 跨基组验证 | — | A, B, C | 验证 |

**总计**：~390 行新代码 + ~200 行重构

---

## 5. 与 LCAO 的关键差异总结

| 组件 | LCAO | PW |
|------|------|-----|
| 原子投影 | SMO (Löwdin 正交化) | OnsiteProj (赝势 beta 函数) |
| 哈密顿修正 | `contributeHR()` 修改 NAO HR 矩阵 | 非局部势 `|beta>·ps` 作用在波函数上 |
| γ 计算 | Wilson loop (Berry 联络矩阵元) | 总 γ + 投影权重分解 (B3) / Berry 联络 (B2) |
| 内层加速 | 冻结密度对角化 (HSolverLCAO) | 子空间对角化 (已有) |
| 约束矩阵 | ✅ 已实现 | ✅ 共享逻辑复用 |
| dp_escon | ✅ 已实现 | ✅ 共享逻辑复用 |
| E-field 输出 | ✅ 已实现 | ✅ 共享逻辑复用 |

---

## 6. 风险与验证

| 风险 | 缓解措施 |
|------|---------|
| PW 的 per-atom 投影与 LCAO SMO 权重不一致 → per-atom γ 值不同 | 先用方案 B3，只测总 γ 差分；per-atom 精度留到 B2 |
| PW Berry 相位计算不存在独立的 ABACUS 接口 | 直接实现 Resta-Z 方法（少数 k-point 的有限差分） |
| OnsiteProjector 的 beta 函数数量与 SMO 轨道数不匹配 | 在 Phase D 验证：H2O 的两种投影对比 |

**最低可行版本（MVP）**：Phase D + B3 + A，使 PW 能约束总 γ 并测量 PES 曲率。Per-atom 精确分解和 BEC 留到后续版本。
