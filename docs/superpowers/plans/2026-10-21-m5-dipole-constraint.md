# M5 偶极约束坐标（Dipole Constraint, saw 优先）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M1 的被动偶极观测量升级为可约束坐标 `ConstraintKind::Dipole`，优先支持周期 saw frame（`p_z` / 界面电势降），并与现有 `ConstraintLoop`、`MuSolver`、ALM、热启动、审计和力链共用同一套机制。

**Architecture:** 偶极不是新的求解器分支，而是新的可观测量权重源。`WeightGrid` 增加 dipole saw / molecular 两种“权重 = dQ/drho”源：saw 的密度权重为 `-Lz*saw(x)`，离子偏移为 `Lz*sum Z*saw(x_i)`；molecular 的密度权重为 `-r_axis`，离子偏移为 `sum Z*r_axis`。观测器把偏移加到积分值上，注入器直接使用同一权重，力核在原有密度导数项之外加入 `-mu*d(offset)/dR`。这样仍然满足“observable == injection operator”，不复制第二套循环。

**Spec:** `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §4（模块 B）。

**前置里程碑:** M1 偶极观测（已合入）、M2 热启动、M3 信赖域、M4 ALM（已合入）。

**代码规范（AGENTS.md）:** 每个 if-else 分支和非平凡 if 块前必须有说明分支目的与原因的注释；禁 `goto` / `setjmp` / `longjmp`；函数超过 300 行必须拆分；优先 RAII / `std::vector`。

**SCF 预算纪律:** 所有集成运行固定 `scf_nmax 100`。100 步仍未收敛时不得调大，必须记录 `drho` 轨迹和约束残差并分析；这一帧作为失败/预算证据进入测试文档。

---

## 物理与单位约定（实现时不得漂移）

设第三晶格方向分数坐标 `x=(r-o).b3/(2*pi)`（ABACUS `taud.z`，`Lz=latvec.e33*lat0`），无量纲 saw 为

```text
s(x) = x - a + 0.5(1-b)                 x <= a
s(x) = 0.5(1-b) - (1-b)(x-a)/b          a < x <= a+b
s(x) = x - a - 1 + 0.5(1-b)             x > a+b
```

与 `DipoleObserver::saw` / `Efield::saw_function` 逐位一致。saw frame：

```text
Q_dip = p_z = Lz * [ sum_I Z_I s(x_I) - integral rho(r) s(x(r)) dr ]
E_con = mu * (Q_dip - t)
v_con(r) = delta E_con / delta rho = -mu * Lz * s(x)
w_dip(r) = -Lz * s(x)                  (Cw 中存储的密度权重)
offset   = Lz * sum_I Z_I s(x_I)       (观测时加到 integral rho*w 上)
F_Iz     = -mu * Z_I * s'(x_I)         (斜坡段 s'=1，衰减段 s'=-(1-b)/b)
```

因此 `mu` 的单位是 `Ry/(e*Bohr)`（均匀电场），目标 `t` 的默认单位是 `e*Bohr`。`deltaV = 4*pi*Q_dip/A`（Hartree）只作报告/`unit:"V"` 输入换算，不改变循环内部单位。

molecular frame：

```text
Q_dip,axis = sum_I Z_I R_I,axis - integral rho(r) r_axis dr
w_dip(r)   = -r_axis
offset     = sum_I Z_I R_I,axis
F_I,axis   = -mu * Z_I
```

原点取 Cartesian 零点；对带电体系绝对 `p` 依赖原点，但 `delta` 模式的 `p-p_ref` 消去该依赖。`absolute + charged/分子` 由文档告警，不在本里程碑额外引入参考点平移。

---

## JSON v2 扩展

```json
{"constraints": [
  {"type": "dipole",
   "target": 2.5,
   "axis": "z",
   "frame": "saw",
   "saw_pos": 0.72,
   "saw_pos_dec": 0.1,
   "unit": "ebohr",
   "mu_max": 0.05}
]}
```

- `axis`: `x|y|z`，默认 `z`。saw frame 本里程碑仅支持 `z`；molecular 支持三轴。
- `frame`: `saw|molecular`，默认 `saw`（周期界面优先）。
- `saw_pos`: 分数坐标，`[-1,1)`；`-1` 或省略表示自动选最低密度平面。
- `saw_pos_dec`: `(0,0.5]`，默认 `0.1`。
- `unit`: `ebohr`（默认）或 `V`；`V` 仅 saw frame，配置阶段按 `A/(4*pi*Hartree_to_eV)` 换成 `e*Bohr`。
- `atoms` 对 dipole 禁止：偶极是全局单约束，多片段分解欠定。
- 全约束列表只能包含一个 dipole，且不可与 charge/spin 混用。
- dipole 与 `efield_amp != 0` 或 `dip_cor_flag = true` 互斥（重复外场）。
- `saw_pos` 自动模式支持冷启动的 reference-first 流程；`fixed_mu` 或 warm start 必须显式给出 `saw_pos`，否则在 init 拒绝，避免在首帧用占位平面注入非零 mu。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `constraint_io.h/.cpp` | `ConstraintKind::Dipole`、spec 的 axis/frame/saw/unit 字段、v2 解析、互斥与单位守卫 |
| `weight_grid.h/.cpp` | dipole saw/molecular 权重源、observable offset、原子偏移导数、saw 平面重定位 |
| `constraint_observe.cpp` | 积分后加 `observable_offset(alpha)`（其它 kind 为 0） |
| `constraint_deriv.cpp` | 密度导数项之外累加 `-mu*d(offset)/dR` |
| `constraint_loop.cpp/.h` | dipole WeightGrid 构建、自动 saw 平面延迟解析、审计单位、warm/fixed 守卫 |
| `constraint_dipole.h/.cpp` | 新增 `saw_slope`（与 saw 的分段导数同源） |
| `test/constraint_io_test.cpp` | 解析、单位、互斥、单约束守卫 |
| `test/weight_grid_test.cpp` | saw/molecular 权重与 offset 的解析对拍 |
| `test/constraint_observe_test.cpp` / `constraint_deriv_test.cpp` | 观测 offset、力原子项 |
| `test/constraint_loop_test.cpp` | 玩具 z 分离晶胞的 saw 约束闭环、审计和力 |
| `tests/constraint_ct_dimer/m5_dipole/` | H2O dimer saw 集成用例（运行目录不入库） |
| `docs/superpowers/specs/2026-10-21-m5-dipole-test.md` | 本轮测试报告 |
| `docs/superpowers/specs/deltap-development-log.md` | 追加 M5 开发日志 |

---

### Task 1: IO / kind / 守卫

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_io.h`
- Modify: `source/source_estate/module_constraint/constraint_io.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_io_test.cpp`

- [ ] **Step 1: 写失败测试**

在 `MixedGuards` 中把“dipole 永久拒绝”改为新语义，并新增 `DipoleConstraintParsingAndGuards`：

```cpp
TEST(ConstraintIOTest, DipoleConstraintParsingAndGuards)
{
    const std::string json = R"({"constraints": [
        {"type":"dipole","target":0.25,"axis":"z","frame":"saw",
         "saw_pos":0.7,"saw_pos_dec":0.08,"mu_max":0.05}]})";
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    std::string error;
    ASSERT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta",
                  json, 5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::OK) << error;
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Dipole);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(specs[0].dipole_frame, "saw");
    EXPECT_EQ(specs[0].dipole_axis, 2);
    EXPECT_DOUBLE_EQ(specs[0].dipole_saw_pos, 0.7);
    EXPECT_DOUBLE_EQ(specs[0].dipole_saw_pos_dec, 0.08);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 0.05);
    EXPECT_EQ(cfg.type, "dipole");
}
```

另写失败断言覆盖：dipole+charge 混用、两个 dipole、dipole 带 `atoms`、`frame` 非法、saw 非 z 轴、`saw_pos_dec` 越界、`unit:"V"` + molecular、未知 unit、`fixed_mu` + 自动 `saw_pos`。

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_io 2>&1 | tail -8
```

预期：编译或断言失败（`ConstraintKind::Dipole` / 新字段不存在）。

- [ ] **Step 3: 实现**

1. `ConstraintKind` 增加 `Dipole`；`build_channel_profile` 对 dipole 返回 charge profile `{1,1,1,1}`；`kind_to_type_string` 返回 `"dipole"`。
2. `ConstraintSpec` 增加：

```cpp
std::string dipole_frame = "saw";   // "saw" | "molecular"
int dipole_axis = 2;                 // 0=x, 1=y, 2=z
double dipole_saw_pos = -1.0;        // fractional; -1 = auto
double dipole_saw_pos_dec = 0.1;
std::string dipole_unit = "ebohr";   // "ebohr" | "V"
```

3. JSON parser 分支：`type=="dipole"` -> kind `Dipole`；解析 `axis/frame/saw_pos/saw_pos_dec/unit`。dipole 默认 `atoms = 全部原子`；显式 `atoms` 直接报错。
4. v2 列表终检：dipole 必须唯一且不能与 charge/spin 混合。
5. `configure_constraint`：对 dipole 检查 saw/molecular、axis/frame、saw 参数、z 对齐（saw 只能在 z），并做 `unit` 换算。`V` 换算需 `ucell`，因此在 `configure_from_inputs` 中完成，换算后把 `spec.target`/`cfg.targets` 同步成 `e*Bohr`。
6. `configure_from_inputs` 互斥守卫：dipole 存在且 (`PARAM.inp.efield_amp != 0.0 || PARAM.inp.dip_cor_flag`) -> ERROR。全 fixed + saw 自动 -> ERROR。
7. v1 的 run-level type 仍只接受 charge/spin；更新错误文案明确 dipole 需要 v2 JSON。

- [ ] **Step 4: 运行通过**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_io
ctest --test-dir build -R MODULE_ESTATE_constraint_io --output-on-failure
```

---

### Task 2: WeightGrid dipole 权重 + offset

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_dipole.h/.cpp`
- Modify: `source/source_estate/module_constraint/weight_grid.h/.cpp`
- Modify: `source/source_estate/module_constraint/constraint_observe.cpp`
- Test: `source/source_estate/module_constraint/test/weight_grid_test.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_observe_test.cpp`

- [ ] **Step 1: 写失败测试**

在 `weight_grid_test.cpp` 增加 `DipoleSawWeightMatchesObserver`：

```cpp
constraint::WeightGrid wg(*ucell, rhopw, radii,
                          constraint::WeightType::DipoleSaw, 0.0,
                          2, 0.7, 0.1);
wg.build();
ASSERT_EQ(wg.nconstraint(), 1);
double q = wg.observable_offset(0);
for (int ir = 0; ir < rhopw->nrxx; ++ir) {
    q += wg.constraint_weight(0)[ir] * rho[ir];
}
q *= dV;
EXPECT_NEAR(q, constraint::DipoleObserver::saw_dipole(
                   *ucell, rhopw, rho_ptr, 1, 0.7, 0.1), 1e-9);
```

同文件增加 molecular 版解析对拍；增加 `saw_slope` 三段值与中心差分对拍（避开分段点）。

`constraint_observe_test.cpp` 增加：非 dipole 的 offset 为 0，dipole 的 `Q = offset + integral`。

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_weight_grid MODULE_ESTATE_constraint_observe 2>&1 | tail -8
```

- [ ] **Step 3: 实现**

1. `WeightType` 增加 `DipoleSaw`, `DipoleMolecular`。构造函数尾部追加可选 `dipole_axis=2, dipole_pos_max=0.0, dipole_pos_dec=0.1`，默认值让现有调用逐位不变。
2. 构造函数对 dipole 类型把 `constraint_atoms_` 设为单个全局片段 `{0,...,nat-1}`。
3. `build()` 首部分支到 `build_dipole()`：

```text
saw:        cw[0][ir] = -Lz * saw(a,b,z_frac)
offset[0]             =  Lz * sum_I Z_I saw(a,b,z_I)
molecular:  cw[0][ir] = -r_axis(ir)
offset[0]             =  sum_I Z_I R_I,axis
maxdev_               =  0
```

4. `build_derivatives()` 对 dipole 类型直接生成全零 `dw_`（权重不依赖原子坐标），避免走 Becke 的 O(N_g N_at^3) 路径。
5. 增加访问器：

```cpp
bool is_dipole() const;
double observable_offset(int alpha) const;
double dipole_atom_offset_derivative(int alpha, int J, int d) const;
void set_dipole_saw_pos(double pos_max);
double dipole_saw_pos() const;
```

`dipole_atom_offset_derivative` 返回 `Z_J`（molecular 且 `d==axis`）或 `Z_J*saw_slope(x_J)`（saw 且 `d==2`），其余 0。
6. `ConstraintObserver::observe` 在 `q*dV` 后加 `wg.observable_offset(alpha)`；现有 kind 的 offset 恒为 0，历史输出不变。

- [ ] **Step 4: 运行通过**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_weight_grid MODULE_ESTATE_constraint_observe
ctest --test-dir build -R 'MODULE_ESTATE_constraint_(weight_grid|observe)' --output-on-failure
```

---

### Task 3: 约束力

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_deriv.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_deriv_test.cpp`

- [ ] **Step 1: 写失败测试**

构造单原子 toy cell、saw dipole weight、零密度、`mu=1`，断言：

```cpp
EXPECT_NEAR(F(J, 2), -ucell->atoms[0].ncpp.zv * saw_slope, 1e-12);
EXPECT_DOUBLE_EQ(F(J, 0), 0.0);
EXPECT_DOUBLE_EQ(F(J, 1), 0.0);
```

再加一个非零均匀密度用例，确认密度导数项对 dipole 为零（权重与原子位置无关），总力仍只有离子偏移项。molecular 用例断言 `F(J,axis) = -mu*Z_J`。

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_deriv 2>&1 | tail -8
```

- [ ] **Step 3: 实现**

在 `constraint_force` 的 pool reduction **之后**加入：

```cpp
for alpha:
  if (!wg.is_dipole_alpha(alpha)) continue;  // comment: only dipole has an offset
  for J,d:
    f[J*3+d] -= mu[alpha] * wg.dipole_atom_offset_derivative(alpha,J,d);
```

该位置很重要：原子偏移项是 rank-independent 的，放在 allreduce 之后可避免按 MPI rank 数重复累加。密度导数项仍由原路径处理（dipole 的导数网格为零）。

- [ ] **Step 4: 运行通过**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_deriv
ctest --test-dir build -R MODULE_ESTATE_constraint_deriv --output-on-failure
```

---

### Task 4: ConstraintLoop 集成

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h/.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

新增 `DipoleSawConstraintLoop`：用 H2O toy cell 但把两分子在 z 方向拉开，令 `zv={6,1,1}`；spec 为单个 `Dipole/saw`，`target=0.5 e*Bohr`，`saw_pos` 显式给在真空。mock SCF 响应：

```cpp
Q(mu) = Q_ref + chi * mu,  chi = -1.0
rho(mu) = rho_ref + chi*mu*w0/S
```

驱动 loop 到 CONVERGED，断言：

1. `loop.charges()[0] == loop.targets()[0]` 在 `thr` 内；
2. `loop.charges()[0]` 与 `DipoleObserver::saw_dipole(...)` 的差值 `<1e-8`；
3. `mu` 落在解析根 `(target-Q_ref)/chi`；
4. 审计行含 `kind=dipole`；
5. `compute_force` 对 dipole 输出非零且等于 `-mu*Z_I*saw_slope` 加解析密度项（dipole 密度项为 0）。

另加两个生命周期测试：

- 自动 `saw_pos`：参考帧 `observe` 后平面解析，注入阶段使用同一平面；`Q` 与 observer 对拍。
- warm start + 自动 `saw_pos` -> init 拒绝；显式 `saw_pos` warm 正常。

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -8
```

- [ ] **Step 3: 实现**

1. `build_geometry` 检测 `specs.size()==1 && specs[0].kind==Dipole`：

```cpp
WeightType wt = frame == "saw" ? DipoleSaw : DipoleMolecular;
wg_.reset(new WeightGrid(ucell, rho_basis, radii, wt, 0.0,
                         axis, saw_pos, saw_pos_dec));
wg_->build();
```

非 dipole 保持现有 Becke 构造和 `set_constraint_atoms` 路径逐位不变。
2. 记录 `dipole_constraint_`、`dipole_constraint_auto_saw_`、`dipole_constraint_saw_resolved_`。
3. `observe()` 在任何积分前分支：若 dipole saw 自动且未解析，调用 `lowest_density_plane`，`wg_->set_dipole_saw_pos(pos)`，打印一次解析日志并执行 `<1 Bohr` 警告。之后 observer 用新权重积分，保证 reference Q 与后续注入同一平面。
4. `init()` 在 warm-start 分支前：dipole saw 自动且 `warm_start_eligible` -> `WARNING_QUIT`，要求显式 `saw_pos`。fixed_mu 自动已在 IO 拒绝。
5. warm start 的 spec 兼容性比较加入 dipole frame/axis/saw_pos/saw_pos_dec，防止跨帧换坐标仍复用 mu。
6. `final_status_line()` 的 CONVERGED 文案按 kind 选择单位：dipole -> `e*Bohr`，其它 -> `e`。
7. `final_report()` 对 dipole 增加一行：frame/axis/saw_pos/target 当前单位与 `deltaV`（saw）。
8. 现有 charge/spin 路径不得改变分支顺序、重置策略或默认输出。

- [ ] **Step 4: 运行通过**

```bash
cmake --build build -j 14 --target MODULE_ESTATE_constraint_loop
ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure
```

---

### Task 5: 集成验证（`scf_nmax=100`）

**Files:**
- Add: `tests/constraint_ct_dimer/m5_dipole/INPUT`
- Add: `tests/constraint_ct_dimer/m5_dipole/STRU`
- Add: `tests/constraint_ct_dimer/m5_dipole/KPT`
- Add: `tests/constraint_ct_dimer/m5_dipole/constraint_target.json`
- Add: `docs/superpowers/specs/2026-10-21-m5-dipole-test.md`

用例：H2O dimer（O...O 沿 z 分离，长 z 盒，真空足够），PW/LCAO 任选已稳定路径；先跑偶极约束一个小 `delta p`，`scf_nmax=100`。记录：

```text
[constraint] final status
CONSTRAINT_AUDIT nconstraint=1
CONSTRAINT_AUDIT c[0] ... kind=dipole ...
[constraint] dipole report
drho 轨迹（若不收敛）
```

判据：

1. `Q_dip` 命中 target，且与 `constraint_dipole_report` 的 saw `p_z` 同值（同一 saw 平面）；
2. `deltaV` 与 `4*pi*p/A` 一致；
3. 力为有限值，且固定 mu 的解析原子项与增量能量斜率同号；
4. 默认 charge/spin 回归仍通过；
5. 若 100 步不收敛，记录每步 `drho`、残差和是否在振荡，禁止调大 `scf_nmax`。

运行命令（按 build 前缀清理 OUT 目录）：

```bash
mpirun --allow-run-as-root -np 8 build/abacus_basic_para > run.log 2>&1
```

- [ ] 记录测试设置、原始日志摘要、通过/失败和收敛性分析。
- [ ] 运行 `ctest -R constraint`，要求全部通过。

---

### Task 6: 文档与收尾

**Files:**
- Modify: `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §4（实现细节回写）
- Modify: `docs/superpowers/specs/deltap-development-log.md`
- Add: `docs/superpowers/specs/2026-10-21-m5-dipole-test.md`

- [ ] **Step 1:** 测试文档包含测试计划 / 设置 / 结果 / 分析 / 下一步五节。
- [ ] **Step 2:** 开发日志追加 M5 条目：文件清单、关键公式、守卫清单、单元/集成结果、遗留风险。
- [ ] **Step 3:** 设计文档 §4 回写实际约定：`w=-Lz*saw`、offset、saw 仅 z、自动平面 warm/fixed 限制、target 单位。
- [ ] **Step 4:** 最终回归：

```bash
ctest --test-dir build -R constraint --output-on-failure
cmake --build build -j 14 --target abacus_basic_para
```

- [ ] **Step 5:** 按逻辑任务拆分提交；不提交 `tests/.../m5_*` 运行目录、`Bohr-CLI 用户手册.md` 或无关未跟踪文档。

---

## 验收映射（设计 §9 模块 B）

| 设计要求 | 本计划落点 |
|---|---|
| q_dip 目标收敛后 p 命中 | Task 4 闭环测试 + Task 5 集成 |
| 链式法则 `mu_q = mu_p*dp/dq` | Task 4 用同一 toy 密度对拍（同一响应下解析斜率） + Task 5 文档 |
| FD 力通过 | Task 3 解析原子项 + Task 4 `compute_force` 对拍 |
| saw 长盒 + 带电体系 ΔV | Task 5 长 z 盒；带电体系参考点/单位在文档中给出边界 |
| 默认关闭/回归 | 所有新行为仅在 `type:"dipole"` 出现时触发；charge/spin 单测和 `ctest -R constraint` |

