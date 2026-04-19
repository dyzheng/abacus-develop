# 测试状态报告 & Debug 计划

> 执行时间: 2026-04-18 10:30-11:15
> 二进制: `abacus_2p` (MPI, commit `92e06a1a1`, 编译时间 08:36)
> 运行方式: `mpirun -n 1 --allow-run-as-root`

---

## 一、34 个测试执行结果汇总

### 测试结果矩阵

| ID | 名称 | 结果 | 崩溃原因 |
|---|---|---|---|
| **自旋基准** | | | |
| 220 | PW_SPIN_S2_Z | ✅ PASS | — |
| 221 | PW_SPIN_S4_XYZ | ✅ PASS | — |
| 200 | LCAO_SPIN_S2_Z | ✅ PASS | — |
| 201 | LCAO_SPIN_S4_XYZ | 🔴 FAIL(1) | INPUT 参数错误 |
| **DFT+U 单独** | | | |
| 222 | PW_DFTU_S2_Z | 🔴 CRASH(134) | eff_pot_pw 垃圾值 → psi_norm assert |
| 223 | PW_DFTU_S2_XY | 🔴 CRASH(134) | 同上 |
| 224 | PW_DFTU_S4_XYZ | 🔴 CRASH(134) | 同上 |
| 202 | LCAO_DFTU_S2_Z | ✅ PASS | — |
| 203 | LCAO_DFTU_S2_XY | ⏳ 未运行 | — |
| 204 | LCAO_DFTU_S4_XYZ | ✅ PASS | — |
| **DeltaSpin 单独** | | | |
| 250 | PW_DS_S2_Z | ✅ PASS | — |
| 251 | PW_DS_S2_XY | ✅ PASS | — |
| 252 | PW_DS_S2_XYZ | ✅ PASS | — |
| 253 | PW_DS_S4_Z | ✅ PASS | — |
| 254 | PW_DS_S4_XY | ✅ PASS | — |
| 255 | PW_DS_S4_XYZ | ✅ PASS | — |
| 300 | LCAO_DS_S2_Z | 🔴 SEGFAULT(139) | density_matrix_io.cpp:191 NULL ptr |
| 301 | LCAO_DS_S2_XY | 🔴 BLOCKED | 同 300 |
| 302 | LCAO_DS_S2_XYZ | 🔴 BLOCKED | 同 300 |
| 303 | LCAO_DS_S4_Z | 🔴 SEGFAULT(139) | 同 300 |
| 304 | LCAO_DS_S4_XY | 🔴 BLOCKED | 同 300 |
| 305 | LCAO_DS_S4_XYZ | 🔴 BLOCKED | 同 300 |
| **DFT+U + DeltaSpin** | | | |
| 260 | PW_DFTU_DS_S2_Z | 🔴 CRASH(134) | eff_pot_pw 垃圾值 (DFT+U 路径) |
| 261 | PW_DFTU_DS_S2_XY | 🔴 BLOCKED | 同 260 |
| 262 | PW_DFTU_DS_S2_XYZ | 🔴 BLOCKED | 同 260 |
| 263 | PW_DFTU_DS_S4_Z | 🔴 CRASH(134) | 同 260 |
| 264 | PW_DFTU_DS_S4_XY | 🔴 BLOCKED | 同 260 |
| 265 | PW_DFTU_DS_S4_XYZ | 🔴 BLOCKED | 同 260 |
| 310 | LCAO_DFTU_DS_S2_Z | 🔴 BLOCKED | LCAO DS 路径 + DFT+U |
| 311 | LCAO_DFTU_DS_S2_XY | 🔴 BLOCKED | 同 310 |
| 312 | LCAO_DFTU_DS_S2_XYZ | 🔴 BLOCKED | 同 310 |
| 313 | LCAO_DFTU_DS_S4_Z | 🔴 BLOCKED | 同 310 |
| 314 | LCAO_DFTU_DS_S4_XY | 🔴 BLOCKED | 同 310 |
| 315 | LCAO_DFTU_DS_S4_XYZ | 🔴 BLOCKED | 同 310 |

### 统计

| 状态 | 数量 | 占比 |
|------|------|------|
| ✅ PASS | 12 | 35% |
| 🔴 CRASH/FAIL | 8 (实际运行) | 24% |
| 🔴 BLOCKED (同类bug) | 14 | 41% |
| ⏳ 未运行 | 1 | 3% |

---

## 二、核心发现

### 发现 1: PW DeltaSpin 完全通过 ✅

**6/6 测试全部收敛** (250-255)，覆盖 nspin=2/4 × z/xy/xyz 全部组合。
- DeltaSpin 模块在 PW 路径上是 **功能正常** 的
- 之前报告的 "堆内存损坏" 是 eff_pot_pw 垃圾值导致的间接崩溃

### 发现 2: PW DFT+U 崩溃根因 — eff_pot_pw 垃圾值 💥

所有 PW DFT+U 测试（222-224, 260-265）崩溃模式一致：

```
eff_pot_pw[0..9]= (-2.71294e+51,0) (7.46316e+50,0) (8.37986e+49,0) ...
  → psi_m_norm <= 0.0
  → assert(psi_m_norm > 0.0) FAILED
  → bpcg_kernel_op.cpp:170
```

**关键特征**:
- eff_pot_pw 值为 10^51 数量级 — 明显是 **未初始化内存 / 越界读取**
- 崩溃发生在 SCF INIT 阶段，第一次 Davidson 对角化时
- LCAO DFT+U（202, 204）正常运行 → bug **仅存在于 PW 路径**
- PW spin-only（220, 221）正常 → bug **仅在 DFT+U 开启时触发**

**addr2line 崩溃栈**:
```
bpcg_kernel_op.cpp:170    ← assert(psi_m_norm > 0.0)
diago_dav_subspace.cpp:443 ← normalize_op() 调用
diago_dav_subspace.cpp:167 ← cal_grad()
hsolver_pw.cpp:426         ← hamiltSolvePsiK
esolver_ks.cpp:158         ← hamilt2rho()
```

### 发现 3: LCAO DeltaSpin 段错误 💥

所有 LCAO DeltaSpin 测试（300-305, 310-315）段错误：

```
Signal: Segmentation fault (11)
Failing at address: 0x18  ← NULL + 0x18 偏移
density_matrix_io.cpp:191  ← 密度矩阵写入时
```

**关键特征**:
- 发生在 "INIT SCF" 之后，进入 SCF 循环前
- 地址 0x18 表明某个对象指针为 NULL，试图访问成员变量
- LCAO DFT+U（无 DeltaSpin）正常运行 → bug **仅在 LCAO + DeltaSpin 组合时触发**

### 发现 4: 功能模块隔离

| 模块 | PW | LCAO |
|------|----|------|
| 自旋极化 (无 U 无 DS) | ✅ | ✅ |
| DFT+U | ❌ crash | ✅ |
| DeltaSpin | ✅ | ❌ segfault |
| DFT+U + DeltaSpin | ❌ crash | ❌ segfault |

**交叉验证**:
- DeltaSpin 的 PW 实现是完整的
- DFT+U 的 LCAO 实现是完整的
- bug 集中在两个交叉点：PW+DFTU 和 LCAO+DS

---

## 三、Debug Todo-List

### Bug #1: PW DFT+U eff_pot_pw 垃圾值 🔴 P0

**影响**: 15 个测试 (222-224, 260-265)
**现象**: eff_pot_pw 在 SCF INIT 阶段被填入 10^51 量级垃圾值

#### 调查步骤

- [ ] **1.1 定位 eff_pot_pw 分配点**
  - 搜索 `eff_pot_pw` 的 `resize`/`new`/`malloc` 调用
  - 确认分配后是否被正确初始化为零
  - 重点文件: `dftu_pw.cpp`, `cal_occ_pw.cpp`

- [ ] **1.2 对比 zdy-tmp 的 eff_pot_pw 初始化**
  - 对比两边 `cal_occ_pw()` 中 eff_pot_pw 的初始化逻辑
  - 检查是否有 `memset` / `std::fill` / `zeros` 缺失
  - 特别关注 `npol==1` vs `npol==2` 分支差异

- [ ] **1.3 检查 vu 数组尺寸匹配**
  - `vu_size=100` (DEBUG 输出) 是否正确对应实际 G 空间大小
  - `onsite_op.cpp` 中 vu 的读写索引是否越界
  - 对比 zdy-tmp `onsite_proj_pw.cpp` 的索引计算

- [ ] **1.4 验证 GPU-CPU 同步路径**
  - `op_pw_proj.cpp:282-292` 的 vu_device sync 是否正确
  - nspin=2 时半量同步是否导致另一半未初始化
  - 检查 `syncmem_var_d2h_op` 的 size 参数

- [ ] **1.5 修复后验证**
  - 运行 222 (PW DFTU S2 Z) 确认收敛
  - 运行 224 (PW DFTU S4 XYZ) 确认非共线通过
  - 与 815 结果数值对比

### Bug #2: LCAO DeltaSpin 段错误 🔴 P0

**影响**: 12 个测试 (300-305, 310-315)
**现象**: `density_matrix_io.cpp:191` NULL 指针 0x18

#### 调查步骤

- [ ] **2.1 读取崩溃点源码**
  - `density_matrix_io.cpp:191` 行内容是什么
  - 哪个对象为 NULL (0x18 偏移暗示访问第 3 个成员)
  - 调用栈分析: 谁调用了密度矩阵 IO

- [ ] **2.2 对比 LCAO DFT+U 路径**
  - LCAO DFT+U (202) 不崩溃 → 找到 DeltaSpin 独有的调用
  - 检查 DeltaSpin 是否跳过了密度矩阵初始化
  - 对比 `esolver_ks_lcao.cpp` 中 DS 和非 DS 分支

- [ ] **2.3 检查 DeltaSpin LCAO 初始化顺序**
  - `run_lambda_loop()` 是否在密度矩阵准备好之前被调用
  - `cal_mw_from_lambda()` 是否正确设置了必要的 state

- [ ] **2.4 修复后验证**
  - 运行 300 (LCAO DS S2 Z) 确认通过
  - 运行 303 (LCAO DS S4 Z) 确认非共线通过
  - 运行 310 (LCAO DFTU+DS S2 Z) 确认组合通过

### Bug #3: 201 INPUT 参数错误 🟡 P1

**影响**: 1 个测试 (201)
**现象**: `seed` 参数不被 LCAO 识别

- [ ] **3.1 修复 201 INPUT** — 移除无效的 `seed 1` 行

---

## 四、执行优先级

```
P0-1.1 eff_pot_pw 分配点调查    ← 影响最大 (15 测试)
P0-1.2 对比 zdy-tmp 初始化逻辑
P0-1.3 vu 数组尺寸/索引验证
P0-1.4 GPU-CPU 同步检查
P0-1.5 修复验证 (222, 224)

P0-2.1 density_matrix_io 崩溃点  ← 影响 12 测试
P0-2.2 对比 LCAO DFT+U 路径
P0-2.3 DeltaSpin 初始化顺序
P0-2.4 修复验证 (300, 303, 310)

P1-3.1 修复 201 INPUT            ← 简单修复
```

## 五、已确认正常工作的功能

- ✅ PW spin-only (nspin=2, 4)
- ✅ PW DeltaSpin (nspin=2, 4 × z, xy, xyz) — **6/6 全部收敛**
- ✅ LCAO spin-only (nspin=2)
- ✅ LCAO DFT+U (nspin=2, 4)
- ✅ 测试框架 (INPUT/STRU/KPT 生成, 运行脚本)
