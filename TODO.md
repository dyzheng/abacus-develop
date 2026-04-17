# ABACUS DFT+U PW Port — 当前任务总览

> 仓库: `/root/abacus-dftu-pw-port` (branch: `feat/dftu-pw-port`)
> 最后更新: 2026-04-17

---

## 核心原则（新增）

### 1. 代码迁移优先级
当 zdy-tmp 与 dftu-pw-port/develop 出现代码冲突时，**优先采用 develop 分支的重构方案**，而非机械复制 zdy-tmp。zdy-tmp 的修改仅作为功能参考，具体实现必须适配 develop 的当前架构。

### 2. 测试一致性要求
**所有修改的物理结果必须与 zdy-tmp 分支保持一致。** 编译通过不等于任务完成，必须通过数值比对验证。

### 3. 测试覆盖矩阵
集成测试必须覆盖以下维度：
- **功能**: DFT+U、DeltaSpin
- **自旋**: nspin=2、nspin=4（非共线）
- **磁矩方向**: z 方向、xy 平面、xyz 倾斜方向
- **基组**: PW（当前重点）+ LCAO（后续）

### 4. 测试集即工作产物
**构建并维护测试集本身就是本项目的工作内容**，不是可选的附加任务。每个新功能或修复都必须伴随集成测试 case。

---

## 一、当前状态快照

### 1.1 编译与测试
- [x] **整库编译**: `cmake --build . -j$(nproc)` **PASS** (100%)
- [x] **DeltaSpin 单元测试**: `ctest -R "deltaspin|lambda_update"` **PASS** (4→6/6)
- [x] **DFT+U 单元测试**: `ctest -R "dftu|operator_dftu"` **PASS** (2→6/6)
- [x] **全量单元测试**: `ctest -R "dftu|delta|lambda|operator_dftu"` **PASS** (6/6)
- [x] **PW 集成测试**: `160_PW_DJ_PK_PU_SO` (DFT+U, nspin=4, SOC) ✅ 运行通过
- [ ] **与 zdy-tmp 的数值比对**: 部分完成（zdy-tmp 无法运行此 case，见 §2.3）

**集成测试基线 (`160_PW_DJ_PK_PU_SO`)**:

| 版本 | 能量 (eV) | 力 | 应力 | 时间 |
|------|-----------|-----|------|------|
| dftu-pw-port | -5662.390604 | 17.627804 | 100838.127667 | 1.15s |
| abacus-develop | -5662.393016 | 14.928918 | 104822.896517 | 19.67s |
| result.ref (旧) | -5662.390886 | 17.965510 | 100582.607209 | 1.26s |

**差异原因**: develop 已重构 force/stress 实现（移除 `npol==1` 特殊分支，统一使用完整 4 项计算）。dftu-pw-port 保留旧分支导致差异。完整适配 develop 重构方案需额外工作量（memory API 签名不兼容），作为独立任务处理。

### 1.2 工作区状态
工作区已清理，**2 个 commit** 已提交：

**已提交（新）:**
1. `060c53d98` — fix: DeltaSpin GPU memory API adaptation and cleanup
   - `cal_h_lambda.cpp`: GPU 索引修正 (pv->nrow/pv->ncol)
   - `cal_mw.cpp`: 移除 debug iostream + 注释掉的死代码
   - `cal_mw_from_lambda.cpp`: GPU memory API 适配 (`ctx` 参数移除)
   - `lambda_update_strategies.cpp`: 移除冗余 iostream

2. `8f7a99778` — fix: correct npol==1 force/stress index and adjust dftu.h include
   - `force_op.cpp`: npol==1 分支 index 从 `nproj` 修正为 `nkb`
   - `stress_op.cpp`: 代码风格清理
   - `dftu.h`: charge_mixing.h 移出 `#ifdef __LCAO`

**未跟踪（保留）:**
- `MIGRATION_PLAN.md` — 迁移计划文档（保留）
- `TODO.md` — 本文件（保留）
- `scripts/` — 迁移工具脚本（保留）

---

## 二、近期提交记录与 zdy-tmp 迁移状态

### 2.1 已完成的迁移批次

| Commit | 内容 | 状态 |
|--------|------|------|
| `82203fe48` | Batch 1: PW OnsiteProjector | ✅ Done |
| `d15c285c2` | Batch 2: PW Force & Stress (partial) | ✅ Done |
| `5b0900be3` | Batch 4: DeltaSpin module | ✅ Done |
| `58997aded` | Batch 5: ESolver mixing_dftu (part 1) | ✅ Done |
| `48134c1b9` | Broyden mixing restore + partial top-7 | ✅ Done |
| `12508850d` | feat: lambda_update_strategies | ✅ Done |

### 2.2 zdy-tmp Top-7 Commits 迁移进度

参考仓库: `/root/abacus-zdy-tmp` (zdy/tmp 分支)

| zdy Commit | 标题 | 涉及文件 | 迁移状态 | 备注 |
|------------|------|---------|---------|------|
| `19ade1859` | Fix: initial error of dngvd on DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ⏸️ **BLOCKED** | 目标仓库无此文件，需调查映射关系 |
| `bce760541` | fix:hip code could not run properly in DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ⏸️ **BLOCKED** | 同上 |
| `b9ce68339` | fix:dngvd.hip.cu run properly in DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ⏸️ **BLOCKED** | 同上 |
| `a9d881c95` | Feature: add conserve_setting for DFTU with DeltaSpin | `charge_mixing.h`, `esolver_ks_pw.cpp` | 🔄 **PARTIAL** | `charge_mixing.h` 已在 `48134c1b9` 中完成；`esolver_ks_pw.cpp` 的 `conserve_setting()` 调用也已加入，但需验证完整上下文 |
| `e9e91d7fe` | Fix: nscf for pw code | `esolver_ks_pw.cpp`, `dftu_occup.cpp`, `dftu_pw.cpp` | 🔄 **PARTIAL** | `esolver_ks_pw.cpp` 可能与 `48134c1b9` 重叠；`dftu_occup.cpp` 和 `dftu_pw.cpp` 的 nscf 逻辑**尚未迁移** |
| `1a6871dca` | fix: nscf error of DFT+U | `dftu.cpp`, `dftu_pw.cpp` | ⏳ **PENDING** | 完全未开始 |
| `34f564ef1` | Fix: deltaspin force error on GPU | `force_op.cu`, `force_op.cpp` | 🔄 **IN PROGRESS** | `force_op.cpp` 的修正已在工作树中（未提交）；`stress_op.cpp` 也有关联修改（未提交）；`force_op.cu` 需检查 |

### 2.3 关键发现

1. **dngvd 阻塞**: `dftu-pw-port` 及上游 `develop` 中均无 `dngvd_op.hip.cu`。`source_hsolver/kernels/rocm/` 下只有 `hegvd_op.hip.cu` 和 `bpcg_kernel_op.hip.cu`。这 3 个 commit 可能：
   - 已被上游重构废弃
   - 功能合并到了 `hegvd_op.hip.cu`
   - 需要创建新文件但目标目录结构已变

2. **force/stress 工作树修改**: 当前未提交的 `force_op.cpp` 和 `stress_op.cpp` 看起来不仅包含 `34f564ef1` 的系数修正，还追加了 `npol == 1` 分支。这可能是 Batch 2 遗留的未完工作，需要与 zdy-tmp 仔细对比确认是否完整，**同时对比 develop 分支确认 `npol==1` 分支是否与上游重构冲突**。

3. **DeltaSpin GPU API**: `cal_h_lambda.cpp` 和 `cal_mw_from_lambda.cpp` 中把 `base_device::memory::xxx(ctx, ...)` 改成了无 `ctx` 版本。这是为了适配 develop 分支中 memory op 的 API 变更。这些修改与编译通过一致，应尽快提交。

---

## 三、行动计划

### Phase 1: 工作区清理与整理（今天，30 min）

- [ ] **1.1 清理误生成的 cmake/测试产物**
  ```bash
  cd /root/abacus-dftu-pw-port
  rm -rf CMakeCache.txt CMakeFiles/ Testing/ commit.h
  rm -f tests/01_PW/test.sum tests/integrate/test.sum
  ```

- [ ] **1.2 处理未提交的 DeltaSpin 清理修改**
  - 文件: `cal_h_lambda.cpp`, `cal_mw.cpp`, `cal_mw_from_lambda.cpp`, `lambda_update_strategies.cpp`
  - 动作: 审阅 diff，确认无功能回退，提交为独立 commit
  - 建议 commit message: `fix: DeltaSpin GPU memory API adaptation and cleanup`

- [ ] **1.3 处理 force/stress 工作树修改**
  - 文件: `force_op.cpp`, `stress_op.cpp`
  - 动作: **三方对比**（zdy-tmp `34f564ef1` vs 当前 diff vs develop 当前状态）
  - 目标: 确认 `npol==1` 分支是 zdy-tmp 既有还是 develop 的新重构；如果是 develop 方案，优先保留 develop 实现，只补回缺失的系数修正
  - 如果确认完整且一致，提交；如果还有缺失，补完后提交

### Phase 2: zdy-tmp Top-7 迁移收尾（接下来 2-3 天）

- [ ] **2.1 T1 — dngvd 映射调查**
  - 使用 `scripts/prompts/task_dngvd_investigate.txt` 启动 subagent
  - **约束**: 调查时必须同时查看 `develop` 中 `hegvd_op.hip.cu` 的当前实现，判断 dngvd 功能是否已被替代
  - 结论输出到 `/tmp/dngvd_investigation_report.md`
  - **决策点**: 这 3 个 commit 是跳过、重定向到新文件，还是手动迁移?

- [ ] **2.2 T2 — nscf esolver 迁移**
  - 文件: `source/source_esolver/esolver_ks_pw.cpp`
  - 参考: `e9e91d7fe` 中的 `esolver_ks_pw.cpp` 修改
  - 动作: **diff 三方对比**（zdy-tmp vs 当前 vs develop），补全尚未应用的部分
  - 重点检查 `calculation == "nscf"` 相关逻辑

- [ ] **2.3 T3 — nscf DFT+U 迁移**
  - 文件: `source/source_lcao/module_dftu/dftu_occup.cpp`, `dftu_pw.cpp`, `dftu.cpp`
  - 参考: `e9e91d7fe` + `1a6871dca`
  - 动作: 按顺序先应用 `e9e91d7fe` 再 `1a6871dca`，处理 nscf 跳过逻辑
  - **如遇 API 冲突，优先采用 develop 当前方案，只保留 zdy-tmp 的业务逻辑**

- [ ] **2.4 T4 — force_op GPU 修复收尾**
  - 文件: `source/source_pw/module_pwdft/kernels/cuda/force_op.cu`
  - 参考: `34f564ef1`
  - 动作: 检查 `force_op.cu` 是否已包含修正；如未包含，应用并验证编译

- [ ] **2.5 T5 — conserve_setting 验证**
  - 文件: `charge_mixing.h`, `esolver_ks_pw.cpp`
  - 动作: 验证 `48134c1b9` 是否已完整覆盖 `a9d881c95`；如已覆盖，标记完成

- [ ] **2.6 T6 — 编译修复（按需）**
  - 如果在 2.1-2.5 中引入编译错误，启动编译修复 subagent

- [ ] **2.7 T7 — 代码审查（每项后）**
  - 对每项修改启动独立 review subagent
  - 输出 `REVIEW_REPORT.md`，必须 `PASS` 才能合并

### Phase 3: 编译与测试 Gate（每项修改后必做）

#### Gate 1: 编译
```bash
cd /root/abacus-dftu-pw-port/build
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
# 目标: 0 error
```

#### Gate 2: 单元测试
```bash
cd /root/abacus-dftu-pw-port/build
ctest -R "deltaspin|dftu|lambda_update" --output-on-failure 2>&1 | tee /tmp/unit_test.log
# 目标: 全部 PASS
```

#### Gate 3: 集成测试（分阶段扩展）

**Stage A — 基础 PW 回归（立即执行）**
```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a ../../build/abacus -n 4 -r "101_PW_.*" 2>&1 | tee /tmp/integration_basic.log
```

**Stage B — DFT+U + DeltaSpin 专项（当前重点）**
需要运行并比对的测试 case：
- DFT+U PW 测试（所有 `dftu` 相关 PW case）
- DeltaSpin PW 测试（所有 `deltaspin` 相关 PW case）
- nspin=2 磁矩沿 z 方向
- nspin=2 磁矩沿 xy 方向
- nspin=4 磁矩沿 z 方向
- nspin=4 磁矩沿 xyz 倾斜方向

** Stage C — LCAO 回归（后续）**
```bash
bash Autotest.sh -a ../../build/abacus -n 4 -r "201_NO_.*"
```

#### Gate 4: 代码审查
- [ ] 逻辑一致性（与 zdy-tmp 意图一致）
- [ ] **优先使用 develop 重构方案**（无冲突时才直接迁移 zdy-tmp）
- [ ] API 适配正确性
- [ ] 无 debug print / WIP 代码
- [ ] 单文件长度合规

### Phase 4: 测试集构建（与代码开发并行）

**目标**: 为 zdy-tmp top-7 修改中每一项功能变更构建最小可复现的集成测试。

#### 4.1 现有测试资源盘点
- [ ] 列出 zdy-tmp 中 `tests/` 目录下与 DFT+U / DeltaSpin 相关的所有 PW case
- [ ] 对比 dftu-pw-port 中是否已存在；如不存在，评估是否需要移植

#### 4.2 新增测试集规格（最小集合）

| 测试 ID | 功能 | nspin | 磁矩方向 | 基组 | 优先级 | 状态 |
|--------|------|-------|---------|------|--------|------|
| `dftu_pw_z2` | DFT+U | 2 | z | PW | P0 | 待建 |
| `dftu_pw_xy2` | DFT+U | 2 | xy | PW | P0 | 待建 |
| `dftu_pw_xyz4` | DFT+U | 4 | xyz | PW | P0 | 待建 |
| `dspin_pw_z2` | DeltaSpin | 2 | z | PW | P0 | 待建 |
| `dspin_pw_xy2` | DeltaSpin | 2 | xy | PW | P0 | 待建 |
| `dspin_pw_xyz4` | DeltaSpin | 4 | xyz | PW | P0 | 待建 |
| `dftu_dspin_pw_z2` | DFT+U + DeltaSpin | 2 | z | PW | P1 | 待建 |

#### 4.3 测试比对流程
每个 case 的运行必须与 zdy-tmp 产出进行数值比对：
1. 在 zdy-tmp 上运行相同 case，保存 `result.ref`
2. 在 dftu-pw-port 上运行，比对关键量：
   - 总能量 (Ry 或 eV)
   - 磁矩 (muB)
   - 力 (Ry/Bohr)
   - 应力 (kbar)
3. **允许偏差阈值**: 1e-6 (能量), 1e-4 (力/应力), 1e-5 (磁矩)
4. 超出阈值必须定位根因

#### 4.4 测试集维护规范
- 每个新 case 必须包含：`INPUT`, `STRU`, `KPT`, `result.ref`, `README.md`
- `README.md` 中必须说明：测试目的、预期结果、与 zdy-tmp 的对应关系
- case 统一放在 `tests/integrate/` 下，按 `dftu_pw_` / `dspin_pw_` 前缀命名

### Phase 5: 更广泛的 DFTU PW Port 未完成项

- [ ] **5.1 Batch 3 — DFTU LCAO 核心**
  - 8 个文件，~1900 行 diff，当前状态未知，需评估是否已在其他 commit 中覆盖

- [ ] **5.2 Batch 5 Part 2 — ESolver + ElecState**
  - `esolver_ks_lcao.cpp` 等仍有大量 diff 待迁移

- [ ] **5.3 Batch 6 — LCAO 基础**
  - `hamilt_lcao.cpp`, `FORCE_STRESS.cpp` 等

- [ ] **5.4 lambda_update_strategies 集成到 SCF**
  - 在 `esolver_ks_lcao.cpp` / `esolver_ks_pw.cpp` 中集成新策略（替换现有 `run_lambda_loop`）
  - 新增 INPUT 参数: `sc_mu_init`, `sc_mu_max`, `sc_mu_growth`, `sc_mix_beta`

---

## 四、Subagent 工作流使用指南

### 4.1 可用脚本
- Orchestrator: `scripts/migrate_zdy_commits_orchestrator.py`
- Prompt 模板: `scripts/prompts/*.txt`

### 4.2 调用规范
**每次只启动 1 个 subagent**（避免 code plan 并发超限）：
1. 主 agent 创建隔离 worktree
2. 填充 prompt 模板并分发
3. 轮询 `PROGRESS.md`（30 秒/次，最多 20 分钟）
4. 检查 `BLOCKERS.md` 是否有阻塞
5. 验收产物（MIGRATION_REPORT.md / REVIEW_REPORT.md）
6. **运行对应集成测试，与 zdy-tmp 数值比对通过后才合并**

### 4.3 并发配置
已在 `~/.hermes/config.yaml` 中设置：
```yaml
delegation:
  max_concurrent_children: 1
```
确保 subagent 串行执行。

### 4.4 新增 Subagent 契约要求
所有派发给 subagent 的迁移任务，prompt 中必须额外包含以下约束：

> "当目标文件与 zdy-tmp 存在 API 冲突时，不要直接复制 zdy-tmp 的旧 API。请先检查 `/root/abacus-develop` 中该文件的当前实现，采用 develop 的重构方案，只保留 zdy-tmp 的业务逻辑。如果无法判断，写入 BLOCKERS.md 并停止工作。"

---

## 五、阻塞与决策清单

| 问题 | 优先级 | 状态 | 下一步 |
|------|--------|------|--------|
| dngvd 文件映射不明 | P3 | ❌ SKIP | ROCm/DCU 专用，目标仓库无此文件，跳过 |
| ~~工作区 cmake 误生成文件~~ | ~~P0~~ | ✅ DONE | 已清理 |
| ~~DeltaSpin 清理修改未提交~~ | ~~P0~~ | ✅ DONE | 已提交 `060c53d98` |
| ~~force/stress 工作树未提交~~ | ~~P0~~ | ✅ DONE | 已提交 `8f7a99778` |
| ~~集成测试基线获取~~ | ~~P0~~ | ✅ DONE | `160_PW_DJ_PK_PU_SO` 运行通过 |
| force/stress develop 重构适配 | P1 | ⏳ | develop 已移除 npol==1 分支，memory API 签名变化大，需作为独立任务处理 |
| ~~nscf DFTU 逻辑未迁移~~ | ~~P1~~ | ✅ DONE | 已提交 `3195855d7` — esolver/dftu/dftu_occup 3 个文件 |
| ~~T4: force_op.cu GPU 修复~~ | ~~P1~~ | ✅ DONE | 已验证：dftu-pw-port 已包含 `34f564ef1` 修正（coefficients1*dbb2 + coefficients2*dbb1）|
| 与 zdy-tmp 的数值比对未进行 | P0 | ⏳ | zdy-tmp 无法运行此 case（参数不兼容），改用 develop 基线 |
| ESolver 与 ElecState 大量 diff 待评估 | P2 | ⏳ | 等 top-7 完成后统一评估 |
| lambda strategies SCF 集成 | P2 | ⏳ | 需要专门的设计决策 |

---

## 六、下一步推荐执行顺序

1. ~~【5 分钟】清理工作区 cmake/测试产物~~ ✅ DONE
2. ~~【10 分钟】提交 DeltaSpin GPU API 适配修改（4 个文件）~~ ✅ DONE
3. ~~【15 分钟】审阅并提交 force_op.cpp / stress_op.cpp 的未提交修改~~ ✅ DONE
4. ~~【20 分钟】运行集成测试获取基线~~ ✅ DONE (`160_PW_DJ_PK_PU_SO` 通过)
5. ~~【当前】启动 nscf DFTU 迁移 subagent (T2 + T3)~~ ✅ DONE (`3195855d7`)
6. ~~force/stress develop 重构适配~~ ✅ 已验证：dftu-pw-port 的 force/stress kernel 已包含 zdy-tmp 34f564ef1 修正
7. **【剩余待办】** T5: conserve_setting 验证 (a9d881c95)

---

*本文件由主 agent 生成，应根据每日进展持续更新。*
