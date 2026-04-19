# PW/LCAO DFT+U + DeltaSpin 完整测试集与状态文档

> **分支**: `dftu-pw-port` (开发) vs `zdy-tmp` (参考)
> **二进制**: `abacus_2p` (MPI, commit `92e06a1a1`)
> **测试策略**: `mpirun -n 1` 单进程运行
> **最后执行**: 2026-04-18 11:15
> **详细 Debug 计划**: `DEBUG_PLAN.md`

---

## 一、完整测试矩阵 (34 个)

### 5 个正交维度
| 维度 | 取值 | 说明 |
|------|------|------|
| **Basis** | PW (17) / LCAO (17) | 基组类型 |
| **DFT+U** | ON (20) / OFF (14) | Hubbard U 修正 |
| **DeltaSpin** | ON (24) / OFF (10) | 自旋约束 |
| **nspin** | 2 共线 (20) / 4 非共线 (14) | 自旋极化模式 |
| **MagDir** | z (14) / xy (10) / xyz (10) | 初始磁矩朝向 |

### 测试结果

#### GROUP A: 自旋基准（无 U 无 DS）— 2/4 已运行

| ID | Basis | nspin | Mag | 结果 | 备注 |
|---|---|---|---|---|---|
| 220 | PW | 2 | z | ✅ PASS | — |
| 221 | PW | 4 | xyz | ✅ PASS | — |
| 200 | LCAO | 2 | z | ✅ PASS | — |
| 201 | LCAO | 4 | xyz | 🔴 FAIL(1) | INPUT 参数错误 |

#### GROUP B: DFT+U 单独（无 DS）— 5/6 已运行

| ID | Basis | nspin | Mag | 结果 | 崩溃原因 |
|---|---|---|---|---|---|
| 222 | PW | 2 | z | 🔴 CRASH(134) | eff_pot_pw 10^51 垃圾值 |
| 223 | PW | 2 | xy | 🔴 CRASH(134) | 同上 |
| 224 | PW | 4 | xyz | 🔴 CRASH(134) | 同上 |
| 202 | LCAO | 2 | z | ✅ PASS | — |
| 203 | LCAO | 2 | xy | ⏳ 未运行 | — |
| 204 | LCAO | 4 | xyz | ✅ PASS | — |

#### GROUP C: DeltaSpin 单独（无 U）— 12/12 已运行

| ID | Basis | nspin | Mag | 结果 |
|---|---|---|---|---|
| 250 | PW | 2 | z | ✅ PASS |
| 251 | PW | 2 | xy | ✅ PASS |
| 252 | PW | 2 | xyz | ✅ PASS |
| 253 | PW | 4 | z | ✅ PASS |
| 254 | PW | 4 | xy | ✅ PASS |
| 255 | PW | 4 | xyz | ✅ PASS |
| 300 | LCAO | 2 | z | 🔴 SEGFAULT(139) |
| 301 | LCAO | 2 | xy | 🔴 BLOCKED | 同 300 |
| 302 | LCAO | 2 | xyz | 🔴 BLOCKED | 同 300 |
| 303 | LCAO | 4 | z | 🔴 SEGFAULT(139) |
| 304 | LCAO | 4 | xy | 🔴 BLOCKED | 同 300 |
| 305 | LCAO | 4 | xyz | 🔴 BLOCKED | 同 300 |

#### GROUP D: DFT+U + DeltaSpin — 6/12 已运行

| ID | Basis | nspin | Mag | 结果 | 崩溃原因 |
|---|---|---|---|---|---|
| 260 | PW | 2 | z | 🔴 CRASH(134) | eff_pot_pw 垃圾值 |
| 261 | PW | 2 | xy | 🔴 BLOCKED | 同 260 |
| 262 | PW | 2 | xyz | 🔴 BLOCKED | 同 260 |
| 263 | PW | 4 | z | 🔴 CRASH(134) | 同上 |
| 264 | PW | 4 | xy | 🔴 BLOCKED | 同 260 |
| 265 | PW | 4 | xyz | 🔴 BLOCKED | 同 260 |
| 310 | LCAO | 2 | z | 🔴 BLOCKED | LCAO DS 路径崩溃 |
| 311 | LCAO | 2 | xy | 🔴 BLOCKED | 同上 |
| 312 | LCAO | 2 | xyz | 🔴 BLOCKED | 同上 |
| 313 | LCAO | 4 | z | 🔴 BLOCKED | 同上 |
| 314 | LCAO | 4 | xy | 🔴 BLOCKED | 同上 |
| 315 | LCAO | 4 | xyz | 🔴 BLOCKED | 同上 |

### 功能模块交叉验证

| | PW | LCAO |
|---|---|---|
| 自旋极化 | ✅ | ✅ |
| DFT+U | ❌ crash | ✅ |
| DeltaSpin | ✅ **6/6** | ❌ segfault |
| DFT+U+DS | ❌ crash | ❌ segfault |

---

## 二、两个核心 Bug

### Bug #1: PW DFT+U eff_pot_pw 垃圾值 💥

**崩溃**: `bpcg_kernel_op.cpp:170` — `assert(psi_m_norm > 0.0)`
**根因**: eff_pot_pw 被填入 10^51 量级垃圾值

```
eff_pot_pw[0..9]= (-2.71294e+51,0) (7.46316e+50,0) (8.37986e+49,0) ...
```

**影响**: 15 个测试 (222-224, 260-265)
**对比**:
- ✅ LCAO DFT+U (202, 204) 正常 → bug 仅在 PW 路径
- ✅ PW DeltaSpin (250-255) 正常 → bug 仅在 DFT+U 路径
- ❌ 815 (PW DFTU) 之前通过，现在也崩溃 → bug 可能在代码中或二进制不一致

### Bug #2: LCAO DeltaSpin 段错误 💥

**崩溃**: `density_matrix_io.cpp:191` — NULL 指针 `0x18`
**触发点**: INIT SCF 之后，SCF 循环之前

**影响**: 12 个测试 (300-305, 310-315)
**对比**:
- ✅ LCAO DFT+U (202, 204) 正常 → bug 仅在 DeltaSpin 路径
- ✅ PW DeltaSpin (250-255) 正常 → bug 仅在 LCAO 路径

---

## 三、Debug Todo-List

### P0 — Bug #1: PW DFT+U (15 测试阻塞)

| 步骤 | 任务 | 关键文件 | 状态 |
|------|------|---------|------|
| 1.1 | eff_pot_pw 分配点调查 | `dftu_pw.cpp`, `cal_occ_pw.cpp` | 🔴 |
| 1.2 | 对比 zdy-tmp 初始化逻辑 | zdy-tmp vs current diff | 🔴 |
| 1.3 | vu 数组尺寸/索引验证 | `onsite_op.cpp`, `op_pw_proj.cpp` | 🔴 |
| 1.4 | GPU-CPU sync 路径检查 | `op_pw_proj.cpp:282-292` | 🔴 |
| 1.5 | 修复验证 (222, 224, 815) | 测试运行 | 🔴 |

### P0 — Bug #2: LCAO DeltaSpin (12 测试阻塞)

| 步骤 | 任务 | 关键文件 | 状态 |
|------|------|---------|------|
| 2.1 | density_matrix_io:191 源码分析 | `density_matrix_io.cpp` | 🔴 |
| 2.2 | 对比 LCAO DFT+U 路径 | `esolver_ks_lcao.cpp` | 🔴 |
| 2.3 | DeltaSpin 初始化顺序 | `cal_mw_from_lambda.cpp` | 🔴 |
| 2.4 | 修复验证 (300, 303, 310) | 测试运行 | 🔴 |

### P1 — 小修复

| 步骤 | 任务 | 状态 |
|------|------|------|
| 3.1 | 修复 201 INPUT (移除 seed) | 🔴 |
| 3.2 | 运行剩余未测试 (203, 201) | 🔴 |

---

## 四、下一步推荐

1. **先修 Bug #1 (PW DFT+U)** — 影响面最大 (15 测试)，且 DeltaSpin 单独工作说明基础 PW 路径正常
2. 同时分析 Bug #2 (LCAO DS) — 独立路径，可并行
3. Bug #1 修复后，222-224 + 260-265 全部 15 个测试应自动通过
