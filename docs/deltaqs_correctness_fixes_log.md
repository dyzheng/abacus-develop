# DeltaQS 正确性修复开发日志

**日期**: 2026-07-01  
**分支**: `feat/deltaqs-unified-framework`  
**基线**: `origin/develop`

---

## 一、问题背景

DeltaQS（电荷约束DFT）寄生在DeltaSpin（自旋约束DFT）之上，但存在多处正确性问题。审查发现：

1. **CSZ投影方案物理上错误**：全zeta投影过完备，不满足幂等性，first-zeta才是正确方案
2. **约束原子列表不含电荷约束**：仅有自旋约束的原子才加入`constraint_atom_list`
3. **纯DeltaQ模式完全崩溃**：DeltaSpin算符仅在`sc_mag_switch=true`时注册
4. **能量修正公式不一致**：`cal_charge_escon`多了target项
5. **spin+charge串行优化逻辑错误**：先收敛spin再收敛charge，mu改变后spin约束被破坏
6. **CSZ死代码+内存泄漏**：构建了CSZ projector但从未使用

## 二、已实施的修复

### Phase 1: 让DeltaQS在纯电荷模式下可运行

| 修改 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 1a | `hamilt_lcao.cpp` | 397 | `sc_mag_switch` → `sc_mag_switch \|\| sc_charge_switch` |
| 1b | `dspin_lcao.cpp` | 258 | `constraint_atom_list`合并`constrain_charge` |
| 1c | `deltaspin_lcao.cpp` | 197 | `use_qs`去掉`&& sc_mag_switch`条件 |
| 1c+ | `deltaqs.cpp` | 523 | 去掉`if(has_spin_constraint)`对`update_lambda()`的保护 |

**效果**: 纯DeltaQ (`sc_charge_switch=1, sc_mag_switch=0`) 不再崩溃，`p_operator`有效，pre_hr正确构建。

### Phase 2: 清除CSZ死代码

| 修改 | 文件 | 说明 |
|------|------|------|
| 2a | `deltaqs.cpp` | 删除`init_deltaqs`中整个CSZ构建块（~50行）+ 4个CSZ include |
| 2b | `spin_constrain.h` | 删除`csz_projector_`、`csz_configs_`成员和upf_valence_parser include |

**效果**: 消除内存泄漏，减少~84行死代码。

### Phase 3: 修复能量修正公式

| 修改 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 3 | `deltaqs.cpp` | 309 | `-mu*(N-N_target)` → `-mu*N`，与`cal_escon()`一致 |

### Phase 4+5: 清理cal_ni_lcao

| 修改 | 文件 | 说明 |
|------|------|------|
| 4 | `deltaqs.cpp` | 删除误导性CSZ注释 |
| 5 | `deltaqs.cpp` | `constrain_all`→`constrain_charge`，仅计算约束原子 |

### Phase 6: 统一CG内循环（核心修改）

**旧实现**: `run_lambda_loop()`收敛spin → 单独mu循环收敛charge（串行两步法）

**新实现**: 单一Polak-Ribiere CG循环同时优化`(lambda, mu)`联合向量

算法结构:
- 状态向量: `Vector3[nat]`(spin) + `double[nat]`(charge) 并行数组
- 残差: `delta_spin = Mi - M_target`, `delta_charge = Ni - N_target`
- RMS: `sqrt(sum(delta²) / n_active_dims)`
- CG方向: `search = delta + beta * search_old`
- **独立步长**: `alpha_spin`用于自旋维度，`alpha_mu`用于电荷维度
- 线搜索: 基于RMS线性插值的`alpha_factor`
- 自适应: `alpha *= g^0.7`

退化行为:
- 纯自旋: 委托给原`run_lambda_loop`
- 纯电荷: CG仅优化mu维度
- 混合: 同时优化所有活跃维度

---

## 三、构建验证

```
5 files changed, 241 insertions(+), 124 deletions(-)
编译: 通过，无新增错误
LCAO测试: 21/21 通过
PW测试: 0/21 通过（预存sc_scf_thr_mode参数问题，与本次修改无关）
```

---

## 四、集成测试发现的问题

### 测试配置
- 用例: `99_LCAO_DQS_S2_Z`（基于`24_LCAO_DS_S2_Z`）
- 体系: Fe BCC, 2 atoms, nspin=2, LCAO
- 约束: spin (mag ±2.0 uB) + charge (tc 16.2/15.8 e)

### 问题1: MPI 4进程 Segfault

**现象**: `mpirun -np 4` 在SCF循环开始时段错误（地址nil）

**根因**: 4进程时部分进程在`init_deltaqs`完成前进入SCF循环，`p_operator`尚未设置。1进程下正常运行。

**可能方案**:
- 在`cal_ni_lcao`和`cal_mi_lcao`中添加`p_operator`空指针保护
- 确保`init_deltaqs`在所有MPI进程上同步完成后再进入SCF

### 问题2: CG收敛速度慢

**现象**: RMS从~2.7降到~0.14后停滞，20步内无法收敛到阈值(0.2)

**根因分析**:
1. **target charge远离自然投影电荷**: 初始RMS≈2.7意味着电荷误差~5e/atom。first-zeta投影的自然电荷可能远非Z_val=16
2. **CG不稳定**: RMS先降后升，呈现振荡行为。可能原因:
   - 线搜索(`alpha_factor`)在RMS非单调时失效
   - Polak-Ribiere beta在RMS振荡时产生不良共轭方向
   - 步长自适应(g因子)响应过慢
3. **每次mu/lambda更新都做全对角化**: 性能瓶颈（同DeltaSpin无子空间加速时的情况）

**可能方案**:
- **确定自然投影电荷**: 先运行无约束SCF，输出Ni，据此设定合理target
- **改进线搜索**: 使用更robust的Armijo/Wolfe条件替代简单线性插值
- **重启机制**: 当RMS连续增加时重置CG方向（beta=0，回到最速下降）
- **子空间加速**: 复用DeltaSpin的SubspaceDiagonalizer，避免每步全对角化
- **阻尼CG**: 在RMS振荡时减小alpha或增加阻尼因子

### 问题3: nspin=4 CG循环中dynamic_cast可能失败

**现象**: `apply_and_solve`中使用`dynamic_cast<DeltaSpin<OperatorLCAO<complex<double>, double>>>`，对于nspin=4，operator类型是`DeltaSpin<OperatorLCAO<complex<double>, complex<double>>>`

**可能方案**: 根据`nspin_`选择正确的template参数进行cast

---

## 五、当前代码状态

### 修改文件清单

```
source/source_lcao/hamilt_lcao.cpp                 |   2 +-
source/source_lcao/module_deltaspin/deltaqs.cpp    | 336 +++++++++++-------
source/source_lcao/module_deltaspin/deltaspin_lcao.cpp |   2 +-
source/source_lcao/module_deltaspin/spin_constrain.h   |  12 -
source/source_lcao/module_operator_lcao/dspin_lcao.cpp |  13 +-
5 files changed, 241 insertions(+), 124 deletions(-)
```

### 待解决事项（优先级排序）

| 优先级 | 问题 | 方案 |
|--------|------|------|
| **P0** | 确定first-zeta自然投影电荷 | 添加无约束Ni输出；根据实际Ni设定合理target |
| **P0** | CG收敛不稳定 | 添加CG重启机制（RMS上升时beta=0）；改进线搜索 |
| **P1** | MPI 4进程segfault | `p_operator`空指针保护 |
| **P1** | nspin=4 unified loop的cast类型 | 根据nspin选择正确的template |
| **P2** | 子空间加速 | 复用DeltaSpin的SubspaceDiagonalizer |
| **P2** | 集成测试用例 | 基于正确target charge创建可收敛的测试 |

---

## 六、公式正确性确认

| 组件 | 状态 | 公式 |
|------|------|------|
| H' = H + (μ±λ_z)P_σ | ✅ 正确 | `cal_coeff_lambda_qs` |
| N_I = Tr[P_I^(1st-zeta) · ρ] | ✅ 正确 | first-zeta, switch_dmr(1)=ρ_total |
| E_scon = -Σμ·N - Σλ·M | ✅ 正确 | `cal_escon()` |
| cal_charge_escon | ✅ 已修复 | `-μ·N` (去掉了target项) |
| μ=0 → 无约束DFT | ✅ 正确 | H'=H_DFT, 自然恢复 |
| 投影算符 | ✅ first-zeta | 近似幂等，物理意义清晰 |

---

## 七、下一步行动建议

1. **诊断自然投影电荷**: 在无约束SCF后输出Ni，确定first-zeta投影的典型值范围
2. **CG稳定性修复**:
   - 添加RMS上升检测：连续2步RMS增加时重启CG（beta=0）
   - 使用Armijo线搜索：`f(alpha) < f(0) + c * alpha * f'(0)`
3. **nspin=4支持**: 在`apply_and_solve`中根据`nspin_`选择正确的cast类型
4. **MPI安全**: 在`cal_ni_lcao`中添加`if(!this->p_operator) return;`保护
5. **创建可收敛的测试用例**: 基于实际自然投影电荷设定target，验证CG收敛
