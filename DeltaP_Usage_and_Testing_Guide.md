# DeltaP 算法使用与测试指南

> **目的**: 总结 DeltaP（逐原子极化分解）算法的当前代码状态，并给出使用与测试的完整流程
> **分支**: `feat/deltap-wilson-per-atom`
> **代码版本**: ABACUS v3.11.0-beta.1 + DeltaP 模块
> **相关文档**: `DeltaP_Incremental_Design.md`（设计）, `docs/superpowers/specs/2026-07-02-deltap-critical-evaluation-report.md`（批判性评估）

---

## 1. 代码状态总结

### 1.1 功能成熟度

| 功能 | 状态 | 说明 |
|------|------|------|
| 总电子极化（Wilson loop 特征值法） | ✅ 可用 | 与 Berry Phase 偏差 ≤0.1%，三体系验证 |
| 逐原子极化 sum rule（Σ P^I = P_total） | ✅ 精确 | SMO 完备时严格成立 |
| 逐原子极化分配 | ⚠️ 不唯一 | SMO 权重给出连续分配，与 Wannier90 差异 7–22%（本质不唯一） |
| Born 有效电荷 Z*（有限差分） | ❌ 不可靠 | 2π 分支跳变 + 误差被 1/δ 放大 |
| SCF 哈密顿量修正算符（`deltap_corr`） | ❌ 不可用 | 能编译，从未成功运行；lambda 更新为占位代码 |

**结论**: DeltaP 当前作为 **NSCF 后处理工具**计算总电子极化和逐原子分解（参考性）是可用的。SCF 约束极化（constrained DFT）尚未实现。

### 1.2 源码结构

核心模块位于 `source/source_lcao/module_deltap/`：

| 文件 | 职责 | 关键函数 |
|------|------|---------|
| `deltap.h` | `DeltaP` 类定义、`AtomicPolarization` 结果结构 | — |
| `deltap.cpp` | 初始化 + k-string 构造 | `init()`, `setup_kstring()` |
| `deltap_wannier.cpp` | **Wilson loop 特征值法（主方法）** | `compute_wannier_polarization()`, `compute_S_dk_link()`, `compute_resta_z()` |
| `deltap_berry.cpp` | 旧 Berry connection 法（已弃用，44% 误差） | `compute_S_k()`, `compute_D_I()`, `compute_berry_connection()` |
| `deltap_overlap.cpp` | SMO 重叠矩阵 | `compute_real_overlaps()`, `compute_smo_overlap_matrix()` |
| `deltap_gauge.cpp` | SMO-anchored 规范固定 | `gauge_fix_smo_anchored()` |
| `deltap_io.cpp` | 输出 + sum rule 验证 | `write_results()`, `verify_sum_rule()` |

集成点：

| 文件 | 位置 | 作用 |
|------|------|------|
| `source/source_io/module_ctrl/ctrl_scf_lcao.cpp` | line 365 | **NSCF 后处理入口**（可用）：`calculation=nscf && deltap_switch` |
| `source/source_lcao/hamilt_lcao.cpp` | line 411 | SCF 算符链注册（`deltap_corr`，未验证） |
| `source/source_esolver/esolver_ks_lcao.cpp` | line 663 | lambda 更新占位代码（未实现） |
| `source/source_lcao/module_operator_lcao/deltap_lcao.cpp` | — | `DeltaPOperator`（H^λ 修正，未验证） |
| `source/source_io/module_parameter/read_input_item_other.cpp` | line 1104 | 输入参数解析 |
| `source/source_io/module_parameter/input_parameter.h` | line 617 | 参数声明 |
| `source/source_basis/module_nao/two_center_bundle.cpp` | — | `overlap_orb_onsite` / `overlap_onsite_onsite` 积分器 |

### 1.3 Git 状态

当前分支 `feat/deltap-wilson-per-atom` 有未提交修改（已提交 30+ commits 覆盖算法 D）：
- 已修改：`deltap.cpp/.h`、`deltap_io.cpp`、`deltap_overlap.cpp`、`deltap_wannier.cpp`、`esolver_ks_lcao.cpp`、`ctrl_scf_lcao.cpp`、`hamilt_lcao.cpp/.h`、`input_parameter.h`、`two_center_bundle.cpp/.h`、两个 CMakeLists
- 未跟踪：`deltap_lcao.cpp/.h`（修正算符）、三体系测试目录（66/67/68）、6 篇评估文档

---

## 2. 输入参数参考

所有参数在 INPUT 文件的 `INPUT_PARAMETERS` 块中设置，`category = DeltaP`。

### 2.1 核心参数

| 参数 | 类型 | 默认值 | 取值 | 说明 |
|------|------|--------|------|------|
| `deltap_switch` | Boolean | False | — | 启用 DeltaP |
| `deltap_method` | String | `berry_connection` | `berry_connection` / `wannier` | **推荐 `wannier`**（Wilson loop 特征值法，精确）；`berry_connection` 已知 44% 误差 |
| `deltap_rm` | Real | 3.0 | Bohr | SMO 调制半径；若为 0 复用 `onsite_radius` |
| `deltap_gdir` | Integer | 3 | 1/2/3 | 极化方向 x/y/z；需与 `gdir`（Berry phase 方向）一致 |
| `deltap_gauge_mode` | String | `none` | `none` / `smo_anchored` | 规范固定模式；**推荐 `smo_anchored`** |

### 2.2 高级参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_dk_fd` | Real | 1e-6 | Berry connection 有限差分验证的 Δk |
| `deltap_npk_string` | Integer | 0 | 覆盖 k-string 密度；0 表示用 KPT 网格 |
| `deltap_anchor_thr` | Real | 1e-8 | 锚定 SMO 重选阈值（`smo_anchored` 模式） |

### 2.3 SCF 修正参数（⚠ 未实现，勿用）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_corr` | Boolean | False | 启用 H^λ 修正算符——**当前不可用** |
| `deltap_lambda_step` | Double | 0.5 | lambda 更新步长（占位） |
| `deltap_nscf` | Integer | 5 | 内层 SCF 最大迭代（占位） |

### 2.4 必需的配套参数

DeltaP 依赖 Berry phase 基础设施，必须同时设置：
- `berry_phase 1` — 启用 Berry phase
- `gdir <1/2/3>` — Berry phase 方向（应与 `deltap_gdir` 相同）
- `basis_type lcao` — 仅支持 LCAO 基组
- `gamma_only 0` — 必须多 k 点（不支持 gamma_only）
- `symmetry 0` — 推荐关闭对称性（避免 k 网格约化导致 nmp=[0,0,0]）

---

## 3. 使用方法

### 3.1 标准工作流（NSCF 后处理）

DeltaP 当前仅在 `calculation=nscf` 时触发，在 Berry phase 计算之后运行。完整流程为两步：

**步骤 1：SCF 自洽计算**（产生电荷密度）

```
INPUT_PARAMETERS
suffix          mysys
calculation     scf
basis_type      lcao
ecutwfc         100
gamma_only      0
nspin           1
scf_thr         1.0e-8
out_chg         1            # 必须输出电荷密度
ks_solver       genelpa
symmetry        0
pseudo_dir      /root/pporb/apns-pseudopotentials-v1
orbital_dir     /root/pporb/apns-orbitals-efficiency-v1
```

运行：`mpirun -np 4 /root/abacus-develop/build/abacus_basic_para`

**步骤 2：NSCF + Berry phase + DeltaP**

```
INPUT_PARAMETERS
suffix          mysys
calculation     nscf
basis_type      lcao
ecutwfc         100
gamma_only      0
nspin           1
scf_thr         1.0e-8
out_chg         0
ks_solver       genelpa
symmetry        0
init_chg        file
read_file_dir   ./OUT.mysys     # 指向 SCF 输出

berry_phase     1
gdir            3

deltap_switch       1
deltap_method       wannier          # 推荐
deltap_rm           3.0
deltap_gdir         3                # 与 gdir 一致
deltap_gauge_mode   smo_anchored

pseudo_dir      /root/pporb/apns-pseudopotentials-v1
orbital_dir     /root/pporb/apns-orbitals-efficiency-v1
```

运行：`mpirun -np 4 /root/abacus-develop/build/abacus_basic_para`

### 3.2 KPT 网格要求

- DeltaP 沿 `deltap_gdir` 方向构造 k-string，k-string 密度来自 KPT 网格
- 推荐 Monkhorst-Pack 网格，各方向 ≥4（晶体），分子体系可用 4×4×4
- `symmetry 0` 避免 nmp 被约化为 [0,0,0]（若发生，代码会从 k 点列表推断网格）

### 3.3 输出文件

运行后在 `OUT.{suffix}/` 下生成：

| 文件 | 内容 |
|------|------|
| `deltap_results.dat` | 逐原子极化 P^I（a.u.）、Total、ABACUS 参考、逐原子电子中心位移 |
| `deltap_smo_weights.dat` | SMO 投影权重矩阵 w_In（行=能带 n，列=原子 I） |
| `deltap_branch.dat` | 跨 SCF 分支跟踪数据 |
| `deltap_zeta_debug.dat` | 每个 k-string 的 ζ = det(W) 调试值 |

`deltap_results.dat` 格式示例：

```
# DeltaP atomic polarization decomposition
# Direction: 3 (1=x, 2=y, 3=z)
# SMO radius: 3 Bohr
# Atom    Px          Py          Pz          (a.u.)
  Ba  0   0.00e+00    0.00e+00    4.817e-03
  Ti  0   0.00e+00    0.00e+00   -1.217e-04
  ...
# Total   ...    4.292e-03
# ABACUS  ...    0.000e+00      # Berry phase 参考值
```

运行日志中会打印 **Sum Rule Check**：

```
* DeltaP Sum Rule Check:
  P_total (DeltaP)  = 4.292e-03
  P_total (ABACUS)  = 0.000e+00
  Relative error    = ...
  [PASS] Sum rule satisfied (< 1%)
```

> **注**: `P_total (ABACUS)` 来自 Berry phase 的电子部分。若两者偏差 >1%，检查 k 网格密度和 `berry_phase` 是否启用。

---

## 4. 测试方法

### 4.1 集成测试（体系级）

测试目录：`tests/17_DS_DFTU/`，每个测试为独立子目录。

| 测试 | 体系 | 方法 | k-mesh | 用途 |
|------|------|------|--------|------|
| `18_LCAO_DELTAP_BTO` | BaTiO3 | berry_connection | — | 早期测试（3% 误差版） |
| `19_LCAO_DELTAP_SI` | Si | berry_connection | — | 规范固定验证 |
| `66_LCAO_DELTAP_BN` | BN 闪锌矿 | wannier | 8×8×8 | **主测试**：含 Wannier90 对比，含 `run.sh` |
| `67_LCAO_DELTAP_H2O` | H2O 分子 | wannier | 4×4×4 | 分子体系验证 |
| `68_LCAO_DELTAP_WATER` | 液态水 4H2O | wannier | 4×4×4 | 多原子分子晶体 |
| `20_LCAO_BTO_BORN` | BaTiO3 | berry_connection | — | Born 有效电荷（11 子目录：平衡 ± 5 原子位移） |

**运行单个测试**（以 BN 为例，推荐起点）：

```bash
cd tests/17_DS_DFTU/66_LCAO_DELTAP_BN
bash run.sh    # 自动完成 SCF → NSCF(ref/disp) → Wannier90 对比
```

`run.sh` 步骤：
1. SCF（`scf/`）→ 电荷密度
2. NSCF+Berry+DeltaP 参考结构（`nscf_berry_ref/`）
3. NSCF+Berry+DeltaP 位移结构（`nscf_berry_disp/`）
4. Wannier90 预处理（`wannier90/`）→ `.nnkp`
5. NSCF+Wannier90 接口（`nscf_wann/`）→ `.mmn/.amn/.eig`
6. Wannier90 主运行（`wannier90/`）→ `.wout`（WF 中心）

**手动运行单步**（任意测试）：

```bash
cd tests/17_DS_DFTU/67_LCAO_DELTAP_H2O/scf
mpirun -np 2 /root/abacus-develop/build/abacus_basic_para > scf.log 2>&1
cd ../nscf_berry
mpirun -np 2 /root/abacus-develop/build/abacus_basic_para > nscf.log 2>&1
cat OUT.H2O/deltap_results.dat
```

**运行整个 17_DS_DFTU 测试套件**（需 CTest 构建启用测试）：

```bash
cd build
ctest -R "17_DS_DFTU" --output-on-failure
```

该套件通过 `tests/integrate/Autotest.sh` 驱动，对比 `result.out` 与 `result.ref`。

### 4.2 Born 有效电荷测试（`20_LCAO_BTO_BORN`）

有限差分法计算 Z*：

$$Z^*_{I,\alpha\beta} = \frac{\Omega}{e}\frac{P_\alpha(+\delta\tau_{I,\beta}) - P_\alpha(-\delta\tau_{I,\beta})}{2\delta\tau}$$

目录结构：`equilibrium/` + `atom{0..4}_{plus,minus}/`（5 原子 × ±位移）。

自动化脚本：`tests/17_DS_DFTU/run_born_charges.py`

```bash
cd tests/17_DS_DFTU
python3 run_born_charges.py    # 位移每个原子，跑 SCF+NSCF，提取 P_z，计算 Z*
```

> **警告**: 当前 Z* 结果不可靠（见 §5.2），仅用于研究 2π 分支跳变问题。

### 4.3 单元测试（数学层面）

位于 `source/source_lcao/module_deltap/test/`，需 `ENABLE_LCAO` + `ENABLE_MPI` + `BUILD_TESTING`：

| 测试文件 | 验证内容 |
|---------|---------|
| `deltap_math_test.cpp` | 相位求和公式 S(k)=Σ e^{ikR}·overlap；解析 dS/dk 与有限差分一致；Berry connection 规范不变性 |
| `deltap_gauge_test.cpp` | SMO-anchored 规范固定：锚定投影为正实数；相位连续无 π 跳变；term1 规范不变 |
| `deltap_smoothness_test.cpp` | Wilson loop 扰动平滑性；规范不变性；锚定跳变导致非平滑 |

构建并运行：

```bash
cd build
cmake -DENABLE_LCAO=1 -DENABLE_MPI=1 -DBUILD_TESTING=1 ..
make -j MODULE_LCAO_deltap_math_test MODULE_LCAO_deltap_gauge_test MODULE_LCAO_deltap_smoothness_test
ctest -R "MODULE_LCAO_deltap" --output-on-failure
```

### 4.4 验证检查清单

运行测试后，按以下顺序检查：

1. **Sum rule**：日志中 `Relative error < 0.01`（[PASS]）
2. **总量精度**：`deltap_results.dat` 的 `# Total` Pz 与 Berry phase 电子极化偏差 ≤0.1%
3. **逐原子合理性**：极性键中电负性大的原子（如 BN 中的 N、H2O 中的 O）应有非零贡献
4. **ζ 值**：`deltap_zeta_debug.dat` 中 |arg(ζ)| 应在 (−π, π] 内，无异常跳变

---

## 5. 已知限制与注意事项

### 5.1 逐原子分解不唯一

Berry phase 是整体量，分解为 Σ P^I = P 没有唯一答案。DeltaP 用 SMO 投影权重 $w^I_n = \sum_{a\in I}|\langle v_n|\alpha_a\rangle|^2$（连续 0–1 分配），Wannier90 用最近原子硬归属（0 或 1）。两者都满足 sum rule 但给出不同分配（7–22% 差异）。**这是数学本质，非 bug。**

### 5.2 Born 有效电荷不可靠

- 单次极化误差（~0.1%）被 1/δ 放大：δ=0.001 Bohr 时放大 ~126 倍
- Wilson loop 特征值 $\arg(\lambda_n)$ 的 2π 分支跳变在逐原子量中不抵消，可导致符号反转
- 小位移避免分支跳变但放大误差，大位移避免放大但有分支跳变——两难

### 5.3 D_mat 基矢不匹配（实现层阻塞）

SMO 投影矩阵 D（`intor_`：LCAO⊗onsite 空间）与 SMO 重叠矩阵 S（`onsite_onsite_intor_`：onsite⊗onsite 空间）基矢不同，导致 Löwdin 正交化 $S^{-1/2}D$ 不自洽，实测 $|D_{a,n}|^2$ 之和远超 1（违反 Cauchy-Schwarz）。当前用归一化非正交权重作为 fallback。

### 5.4 SCF 修正算符未实现

`deltap_corr=1` 会注册 `DeltaPOperator`（$H^\lambda = -\sum_I \lambda_I \tau^I_\alpha \hat{P}^I$）到算符链，但：
- lambda 更新是占位代码（`esolver_ks_lcao.cpp:663` 仅回传当前值）
- SCF 内 Wilson loop 计算未集成
- 从未成功运行任何测试

**请勿在生产计算中设置 `deltap_corr=1`。**

### 5.5 基组依赖

SMO 权重类似 Mulliken 布居，依赖轨道截断半径、zeta 数量、l_max。跨基组可转移性未验证。

---

## 6. 快速开始（最小可运行示例）

以 H2O 分子为例（最小体系，~1 分钟）：

```bash
cd /root/abacus-develop/tests/17_DS_DFTU/67_LCAO_DELTAP_H2O

# 1. SCF
cd scf && mpirun -np 2 /root/abacus-develop/build/abacus_basic_para > scf.log 2>&1 && cd ..

# 2. NSCF + DeltaP
cd nscf_berry && mpirun -np 2 /root/abacus-develop/build/abacus_basic_para > nscf.log 2>&1

# 3. 查看结果
cat OUT.H2O/deltap_results.dat
grep -A4 "Sum Rule Check" nscf.log
```

预期：`P_total` 与 Berry phase 电子极化偏差 ≤0.1%，Sum Rule [PASS]。

---

## 附录：文档索引

| 文档 | 内容 |
|------|------|
| `DeltaP_Incremental_Design.md` | 基于 DeltaSpin 的增量开发设计、映射分析、路线图 |
| `DeltaSpin_vs_DeltaP_Essential_Difficulties.md` | DeltaSpin 与 DeltaP 的本质差异分析 |
| `docs/superpowers/specs/2026-06-28-deltap-algorithm-evaluation.md` | 6 种候选算法的理论分析与评估 |
| `docs/superpowers/specs/2026-06-30-deltap-three-system-test-results.md` | BN/H2O/液态水三体系完整测试数据 |
| `docs/superpowers/specs/2026-06-30-deltap-berryphase-wannier90-comparison-guide.md` | 三方法对比公式与方法论 |
| `docs/superpowers/specs/2026-07-02-deltap-critical-evaluation-report.md` | 批判性评估报告（最新、最全面） |
