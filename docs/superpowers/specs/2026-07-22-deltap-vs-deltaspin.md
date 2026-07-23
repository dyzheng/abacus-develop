# DeltaP vs DeltaSpin 参数对比与改进建议

**日期**: 2026-07-22

---

## 1. 死参数清理

| 参数 | 状态 | 处理 |
|------|------|------|
| `deltap_npk_string` | **死** — 源码完全未引用 | ✅ 已移除 |
| `deltap_rm` | **半活** — 仅在 overlap 中用作截断，不在核心 Wilson loop 中使用 | 保留（overlap 依赖） |

---

## 2. 参数对照表

| 功能维度 | DeltaP | DeltaSpin | 差异 |
|---------|--------|-----------|------|
| **总开关** | `deltap_switch` | `sc_mag_switch` | 功能等价 |
| **约束开关** | `deltap_corr` | — (switch 即约束) | DeltaSpin 无独立约束开关 |
| **靶标** | `deltap_target_file` | STRU 文件中 `sc_mag` | DeltaP 用独立文件，更灵活；DeltaSpin 内嵌 STRU |
| **λ 步长** | `deltap_lambda_step` (固定) | `alpha_trial` (自适应) | DeltaSpin 的 BFGS 自适应调步长，DeltaP 固定步长 |
| **λ 阻尼** | `deltap_lambda_mixing` | — | DeltaP **独有**，DeltaSpin 无阻尼混合 |
| **初始 λ** | `deltap_lambda_init` | STRU 文件中 `lambda` | 功能等价 |
| **内层循环** | `deltap_nscf` | `nsc` + `nsc_min` | DeltaP 缺最小迭代数控制 |
| **内层收敛** | `deltap_conv_thr` | `sc_thr` + `sc_drop_thr` | DeltaSpin 有自适应收敛阈值 |
| **内层触发** | `deltap_inner_thr` | `sc_scf_thr` + `sc_scf_thr_mode` | DeltaSpin 有模式选择 |
| **两阶段** | implicit (0/>0 nscf) | `sc_direction_only` + `sc_dir_phase1_steps` | 完全不同的两阶段策略 |
| **优化方法** | 固定梯度下降 | `sc_lambda_strategy` (bfgs/linear_scan) | DeltaSpin 更丰富 |
| **加速策略** | 无 | `sc_strategy` + `sc_acceleration_mode` | DeltaSpin 有子空间加速，DeltaP 无 |
| **约束组合** | `deltap_constraint_matrix` | — | DeltaP **独有** |
| **多方向** | `deltap_gdir` | —（磁矩天然三维） | DeltaP 仅单方向约束 |
| **Gamma-only** | — | — | 两者均不支持 |

---

## 3. 需调整的参数

### 3.1 建议新增

| 建议参数 | 类比 DeltaSpin | 理由 |
|---------|--------------|------|
| `deltap_nscf_min` | `nsc_min` | 内层 λ 循环的最小迭代数，防止过早退出 |
| `deltap_drop_thr` | `sc_drop_thr` | 当初始残差很大时自动放宽收敛精度 |
| `deltap_scf_thr_mode` | `sc_scf_thr_mode` | 控制 λ 更新时机：`"threshold"` / `"immediate"` |

### 3.2 建议修改默认值

| 参数 | 当前默认值 | 建议值 | 理由 |
|------|----------|--------|------|
| `deltap_lambda_step` | 0.5 | **0.01** | 0.5 对大多数体系过大，H2O 测试中 0.01 更合理 |
| `deltap_lambda_mixing` | 0.0 | **0.1** | 0（不混合）在刚度大的体系中可能震荡 |
| `deltap_inner_thr` | 1e-4 | **1e-3** | 1e-4 太严格，drho 需降至 1e-4 才更新 λ，延长 Phase 1 |

### 3.3 不建议调整的

| 参数 | 理由 |
|------|------|
| `deltap_conv_thr` | 1e-3 对 |γ-target| 收敛合理 |
| `deltap_nscf` | 0（同步两阶段）已验证有效 |
| `deltap_method` | berry_connection 已被大量测试 |
| `deltap_constraint_matrix` | 功能完整 |

---

## 4. 架构层面差异

| 方面 | DeltaP | DeltaSpin | 影响 |
|------|--------|-----------|------|
| λ 优化器 | 固定梯度下降 | BFGS / Polak-Ribiere CG | DeltaSpin 鲁棒性更强 |
| 子空间加速 | 无 | 有 | DeltaSpin 在接近收敛时更快 |
| 靶标输入 | 独立 target.dat | 嵌入 STRU | DeltaP 更灵活，但需额外文件 |
| 约束组合 | 线性约束矩阵 | 无 | DeltaP 独有能力 |
| 扫描模式 | 无 | `linear_scan` | DeltaSpin 有系统诊断模式 |
| 能量修正 | dp_escon（本次新增） | escon（原生） | 已一致 |

---

## 5. 优先整改清单

1. **✅ 移除 `deltap_npk_string`**（死参数） — 已完成
2. **修改默认值** `deltap_lambda_step → 0.01`, `deltap_lambda_mixing → 0.1`, `deltap_inner_thr → 1e-3`
3. **考虑新增** `deltap_nscf_min`（最低内层迭代）
4. **中期**：将 λ 优化器从固定梯度改为 BFGS（对齐 DeltaSpin）
5. **远期**：添加 `deltap_lambda_strategy = scan` 扫描模式
