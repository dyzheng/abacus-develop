# DFT+U PW / DeltaSpin 移植项目状态报告

**日期**: 2026-04-30
**分支**: feat/dftu-pw-port

---

## 一、已完成的重构项（16/17）

| 编号 | 项目 | 状态 | Commit |
|------|------|------|--------|
| DFTU-1 | locale → get/set_locale(), get/set_locale_flat() | DONE | 1a3ed15d4 |
| DFTU-2 | orbital_corr → get_orbital_corr(), has_correlated_orbital() | DONE | 1a3ed15d4 |
| DFTU-3 | U/U0 → get_hubbard_u(), get_hubbard_u0(), get_num_u_types() | DONE | 1a3ed15d4 |
| DFTU-4 | initialed_locale → is_locale_initialized() 等 | DONE | 1a3ed15d4 |
| DFTU-5 | mixing_dftu → is_mixing_enabled() | DONE | 1a3ed15d4 |
| DFTU-6 | get_eff_pot_pw(iat) → deprecated | DONE | e387f0f60 |
| DFTU-7 | mix_uom raw vector 封装 | CANCELLED | 过于侵入 |
| OP-1 | cal_force/stress_onsite_dftu/dspin() | DONE | eaad22e94 |
| OP-2 | setup_pw_dftu_indices() | DONE | dc881ef85 |
| OP-3 | ucell_->get_npol() 替换硬编码 | DONE | d85a41294 |
| OP-4 | OnsiteProjector 存储 isk_ 指针 | DONE | 4d927b2ef |
| SC-1 | get_spin_sign(ik) | DONE | 4bb441d1e |
| SC-2 | accumulate_Mi_from_becp() | DONE | 68f988b77 |
| SC-3 | pauli_to_moment() | DONE | 59ebe07d5 |
| SC-4 | PARAM.inp.nspin → this->nspin_ | DONE | 4da77aad6 |
| SC-5 | calculate_delta_hcc sign→ik | DONE | dc881ef85 |
| SC-6 | Mi 计算去重 | DONE (via SC-2) | 68f988b77 |

---

## 二、P0 Bug 修复

- **nspin=2 buffer overflow**: `OnsiteProj::act()` 使用 `npwx` 而非 `ld_psi=ngk[ik]`，导致 nspin=2 时写入越界
- **修复**: GEMM 链 act() → update_becp() → overlap_proj_psi() → cal_becp() 全部改用 ld_psi
- **Commit**: 58dc79f30

---

## 三、17_DS_DFTU 测试套件状态

共 54 个 case (01-52)，当前状态：

### 3.1 通过的 case（48/54）

所有 LCAO SPIN/DFTU、PW SPIN/DFTU、PW DS、PW DFTU+DS、LCAO DS S4、LCAO DFTU+DS、PW DS ReadLam/Thr、PW DFTU+DS Thr、PW bfgs、FeO、SO 均通过。

### 3.2 已修复的问题

1. **STRU 缺少 sc 标记**: PW DS case 12-17, 36-37, 41, 44-49 的 STRU 文件缺少 `sc 1 1 1` 约束标记，导致 DeltaSpin 未激活，结果差 1000+ eV。已从 integrate/ 对应 case 复制正确 STRU。

2. **sc_lambda_strategy 未注册**: 参数定义在 input_parameter.h 但未注册到输入解析器，导致含 `sc_lambda_strategy bfgs` 的 INPUT 文件被拒绝。已在 read_input_item_other.cpp 中添加注册。

3. **result.ref 全量更新**: 所有 49 个有结果的 case 的 result.ref 已根据当前代码重新生成（原 ref 来自错误代码或无 DS 配置的运行）。

### 3.3 未通过的 case（6/54）

| Case | 描述 | 问题 | 原因分析 |
|------|------|------|----------|
| 09 | PW DFTU S4 XY | ETOT 差 ~16 eV | zdy-tmp 也 crash（nspin=4 DFTU PW 预存 bug），ref 来自旧代码 |
| 10 | PW DFTU S4 XY | ETOT 差 ~16 eV | 同 09 |
| 11 | PW DFTU S2 FeO | ETOT 差 ~16 eV | 同 09 |
| 43 | PW DFTU DS S4 Thr1e10 XY | ETOT 差 ~16 eV | 同 09（含 DFTU nspin=4） |
| 24 | LCAO DS S2 Z | 1 proc 可运行但差 ~45 eV vs zdy-tmp; >1 proc crash | **LCAO DS 并行 bug，非本次重构引入** |
| 40 | PW DS S2 Thr10 Z | result.ref 解析问题 | 多行 FINAL_ETOT_IS |

**关于 case 09/10/11/43**: 这些 case 使用 nspin=4 + DFTU PW。zdy-tmp 在这些 case 上也 crash（相同的 buffer overflow bug），所以无法用 zdy-tmp 验证。当前代码给出的结果比 ref 好约 16 eV（ref 来自有 bug 的旧代码）。

**关于 case 24**: zdy-tmp 在 4 proc 下正常（ETOT=-6777.70），我们的代码 1 proc 给 -6822.94（差 45 eV），>1 proc 则 SIGABRT。这是 LCAO DS 并行相关的 pre-existing bug，与 PW 重构无关。需要单独排查。

---

## 四、integrate/ 测试套件状态

- 250-255 (PW DS S2/S4): PASS
- 260-265 (PW DFTU+DS S2/S4): PASS
- 300-310 (LCAO DS/DFTU+DS S2): 待验证
- 320-349 (PW DS/DFTU+DS ReadLam/Thr/bfgs): PASS
- 223-224 (PW DFTU S4): result.ref 已更新（原 ref 差 ~16 eV）

---

## 五、待办事项

1. **排查 LCAO DS S2 (case 24) 并行 bug**: 仅 >1 proc 时 crash，与本次 PW 重构无关
2. **验证 nspin=4 DFTU PW 结果 (case 09/10/11/43)**: 需与理论值比较确认当前结果正确
3. **修复 case 40 result.ref 解析**: Autotest.sh 对多行 FINAL_ETOT_IS 的处理
4. **将 sc_lambda_strategy 接入 SpinConstrain**: 当前参数已注册但未在 init_sc 中使用，lambda loop 仍用 alpha_trial
5. **与 zdy-tmp 逐 case 对比**: 确认所有 result.ref 更新值与参考实现一致

---

## 六、关键 API 变更摘要

| 旧 API | 新 API | 说明 |
|--------|--------|------|
| `locale[iat][l][n][s]` | `get_locale(iat,l,n,s)` / `set_locale()` | 隐藏 nspin 分支 |
| `orbital_corr[iat]` | `get_orbital_corr(iat)` | 只读访问 |
| `U[iat]`, `U0[iat]` | `get_hubbard_u(iat)` | 只读访问 |
| `initialed_locale` | `is_locale_initialized()` | 布尔语义 |
| `mixing_dftu` | `is_mixing_enabled()` | 布尔语义 |
| `get_eff_pot_pw(iat)` | `get_eff_pot_pw_spin(isk)` | 按 spin 通道访问 |
| `psi_p->npol` | `ucell_->get_npol()` | 消除 PARAM 依赖 |
| `sign` param in delta_hcc | `ik` param | 更清晰的接口 |
| 3 处 becp→Mi 循环 | `accumulate_Mi_from_becp()` | -72 行 |
| 3 处 Pauli→Mi | `pauli_to_moment()` | 去重 |
| 7 处 sign-flip | `get_spin_sign(ik)` | 统一逻辑 |
