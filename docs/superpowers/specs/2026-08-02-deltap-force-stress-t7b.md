# DeltaP Force/Stress T7-b：FD 双组协议实证 —— 首次量化 A1-only 力与 E' 的缺口

> 2026-08-02 · 依据 `2026-08-02-deltap-force-stress-dev-guide.md` §4-T7-b 执行。
> 结论先行：**当前 deltap_corr 的 relax 受力不可信**——组①/组② FD 残差超出判据
> 2–3 个数量级；根因定位为 **H_HK 解析力缺失（B 项，主导）** + **τ 单位与规格不符
> （代码用 lat0 单位位置，文档用分数坐标）** + A2/C 未实现。指南 O1 得到答案：
> **H_HK 力项必须解析实现（F6 从"不实现"改为"必须实现"）**。

---

## 1. Test plan

1. 补 escon 记账验证：确认 `!FINAL_ETOT_IS` 已含 `dp_escon`（代码链审计）；
2. 实现 FD 双组协议脚本（组① 冻结 λ / 组② 每位移点重收敛 λ），判据
   |F_FD − F_ana| < 5e-4 Ry/Bohr = 0.01286 eV/Å；
3. 组①：λ 冻结在 base 收敛值，±δ=0.005 Bohr 位移，全原子×3 方向 FD；
4. 组②：同位移、λ 重收敛（relax 实际受力路径）；
5. 对照实验隔离缺口来源：
   - 纯 SCF（deltap_corr=0）FD —— 验证协议/提取本身自洽；
   - λ=0（deltap on）—— 验证机制无 λ=0 泄漏；
   - 分数 τ（taud）重跑组① —— 验证 τ 单位假设；
   - 禁用 H_HK 重跑组① —— 定量 B 项缺口。

## 2. Test setup

- 系统：本机（Intel Ultra 5 225H，14 核），`mpirun -np 1`，常规构建
  （`-O3 -DNDEBUG -std=gnu++14`）。
- 算例 `tests/deltap_fd_force/h2o1`：单 H₂O（O 2s2p1d / H 2s1p），15.873 Å 立方盒，
  1×1×2 Gamma（2 k），`calculation=scf`，`deltap_gdir=3`，`deltap_corr=1`，
  `deltap_target_file`（3×0），scf_thr 1e-7，genelpa，cal_force=1。
  （弃用原 4 分子/10 Å 盒：2×2×2 下 γ 分支跳变 SCF 不收敛，64 k 网格成本过高。）
- δ = 0.005 Bohr = 0.002646 Å（STRU 为 Cartesian_angstrom）；E' 取
  `!FINAL_ETOT_IS`（eV，已含 dp_escon）；F_ana 取 `#TOTAL-FORCE`（eV/Å）。
- λ 冻结机制：新增 `deltap_lambda_init_file`（逐原子 λ 初值）+ `deltap_lambda_step 0.0`。

### escon 记账验证（指南 §4 的"补 escon 项"修正）

`fp_energy.cpp:19` `calculate_etot()` 已把 `dp_escon` 计入 `etot`；
`esolver_fp.cpp:234` 打印 `f_en.etot` ⇒ **`!FINAL_ETOT_IS` 即 E' = E_KS + escon**，
FD 提取**不需要再补 escon**（指南"run_fd.sh 需补 escon 项"基于 etot 不含 escon 的
误读）。λ=0 对照实测：`E'(deltap, λ=0) = -481.6911714598026` = 纯 SCF
`-481.691171459806`（12 位一致）⇒ 记账链路无泄漏。为可审计性，`[DeltaP P3]`
报告行已加 `escon=... Ry` 打印。

## 3. Results

### 3.1 组①（冻结 λ*，绝对 τ 现状代码）—— 9 方向全矩阵

base：E'(R0) = -488.44815193535 eV，λ* = (-5.513928e-3, -3.601925e-3, -3.601925e-3)，
γ = (-5.518, -3.600, -3.600)，escon = -0.056360 Ry。

| 原子 | 方向 | F_FD (eV/Å) | F_ana (eV/Å) | Δ | 判据 0.0129 |
|------|------|-------------|--------------|-----|-----|
| O1 | x | +1.1e-5 | +3.3e-6 | 8e-6 | PASS |
| O1 | y | +1.4e-5 | +2.0e-6 | 1.2e-5 | PASS |
| O1 | z | **+4.5615** | -0.7733 | **+5.3348** | FAIL |
| H1 | x | **+1.0491** | -0.3209 | **+1.3700** | FAIL |
| H1 | y | 0.0 | -1e-6 | 1e-6 | PASS |
| H1 | z | **-0.4829** | +0.3867 | **-0.8696** | FAIL |
| H2 | x | **-1.0491** | +0.3209 | **-1.3700** | FAIL |
| H2 | y | 0.0 | -1e-6 | 1e-6 | PASS |
| H2 | z | **-0.4829** | +0.3867 | **-0.8696** | FAIL |

5/9 FAIL，残差 0.87–5.33 eV/Å（判据的 68–415 倍）。y 方向全 PASS（H₂O 分子在
xz 平面，y 方向对称零力）。

### 3.2 FD 可信度验证

- **δ 线性度**（O1-z）：±δ 斜率 -4.5616 eV/Å，±2δ 斜率 -4.5633 eV/Å —— 0.04%
  一致 ⇒ E'(R) 平滑，FD 值可信（无分支跳变 kink）。
- **λ 线性度**：λ×10 → 斜率 -49.09 eV/Å（×10.8）⇒ 缺失项 ≈ λ 线性。
- **纯 SCF 对照**：O1-z F_FD = -0.759 vs F_ana = -0.753（0.8% 一致）⇒ 协议、
  STRU 位移、E'/力提取全部自洽；异常只在 deltap 路径。

### 3.3 组②（每位移点重收敛 λ，z 方向实测）

| 原子 | F_FD (eV/Å) | F_ana (eV/Å) | Δ |
|------|-------------|--------------|-----|
| O1 | +5.863 | -0.7733 | **+6.636** |
| H1 | -0.716 | +0.3867 | **-1.102** |
| H2 | -0.691 | +0.3867 | **-1.077** |

组② 残差比组① 更大（O1 +6.6 vs +5.3）。γ≈t 远未满足（|γ−t|=5.5），H1/H2 的
γ 分支在 ±δ 间交换（-3.593/-3.608）⇒ λ 随之交换 ⇒ E' 快速变化。**relax 可用性
判决：FAIL**（与指南 §4 预期一致：约束未激活时 Pulay-only 不精确）。

### 3.4 隔离实验（根因定位）

| 配置（均 O1-z，组①） | F_FD | F_ana | Δ |
|----------------------|------|-------|-----|
| 现状（绝对 τ + H_HK on） | +4.562 | -0.773 | **+5.335** |
| 分数 τ（taud）+ H_HK on | +2.190 | -0.710 | **+2.900** |
| 分数 τ + H_HK off | -0.495 | -0.759 | **+0.264** |
| 纯 SCF（无 deltap） | -0.759 | -0.753 | **-0.006** |

- **H_HK 力缺失（B 项）≈ 2.64 eV/Å**（= 2.90 − 0.26）：主导缺口。
- **τ 单位**：绝对→分数使 O1-z 残差 -2.4 eV/Å；分数 τ 下 H1/H2-z 残差 ≈ 0.003
  （绝对 τ 下 -0.87）⇒ τ 单位错误同时放大 H 原子残差。
- 剩余 +0.26 eV/Å ≈ A2 + C + B 残余（指南预测组① 残差量级，方向一致）。

## 4. Analysis（根因链）

### B-7（新，主导）：H_HK 解析力缺失

- H_HK（`deltap_wannier.cpp compute_hk_correction`）进入 hsk 的矩阵元 ~2.5e-3 Ry
  （`[DeltaPOp] contributeHk max|corr|`），其 E_band 贡献随 O1-z 变化
  ~-0.0156 eV/(2δ) —— 但 `deltap_force_stress.hpp` 头注释自认只覆盖 H_HR
  （"H_HK does not have an analytic force contribution"），指南 F6 原判"不实现，
  FD 定量"。
- **定量结果：B 项 ~2.6 eV/Å（分数 τ 组①），非可忽略小项**。原因是 H_HK =
  sym[(i/2)w_eff·S_dk·c·c†] 依赖 SMO 系数 D（随 R 快速变化，尺度为轨道尺寸而非
  晶胞），且 w_eff = Σλ|D|² 直接进 Hamiltonian ⇒ 其 ∂/∂R 是约束主导力。
- **O1 裁决：F6 升级为实现项**（T7-c 必做）；在此之前 relax 数值不可信。

### B-6（τ 单位与规格不符，量级错误）

- 指南 §2.2a/F5/F8：τ_α = **分数坐标**；实现（`deltap_lcao.cpp contributeHR`、
  `deltap_force_stress.hpp`）用 `atoms[T].tau[I][α]` = **lat0 单位笛卡尔位置**
  （数值 ≈ Å，`read_atoms_helper.cpp:215`），= 分数坐标 × L_α（本盒 15.87）。
- 后果：H_HR（及 A1/A2 全部 λτ 项）被放大 L 倍；E'(λ*)−E'(0) = -6.76 eV
  （分数 τ 为 -1.06 eV ≈ escon + 响应量级）；P2 后电子响应 -0.44 Ry vs -0.022 Ry。
- 该改动用 `taud` 实现已实测（§3.4）；但修改会改变全部 deltap 数值结果
  （SCF 端、test_C_I 类锚点）⇒ **列为 BLOCKER，需与 O2（分数 vs Cartesian 应变
  耦合语义）一并决策后修**，不擅自并入本轮。

### A2/C 未实现（指南已知，量级确认）

- 分数 τ + H_HK off 下残余 +0.26 eV/Å ≈ A2（∂τ/∂R HF 项）+ C（λ·dγ/dR，实测
  γ 变化 dγ/dz ~1.7 Å⁻¹ ⇒ C ~0.24 eV/Å）+ B 残余 —— 仍超判据 20×，T7-c 实现后
  复验。

### 组② 额外噪声：γ 分支交换

- H1/H2 的 γ（-3.60±0.01）在 ±δ 间交换归属（分支简并），λ 随 γ 重收敛后交换
  ⇒ E' 对位移的响应含 λ 跳变成分；约束激活（γ≈t）前组② 残差无意义地大。

## 5. 代码改动（T7-b 交付）

- `source/source_esolver/deltap_scf.cpp`：`[DeltaP P1|P3]` 报告行加
  `escon=... Ry`（6 位）打印；λ 打印精度 2→6 位有效数字（组① 冻结需精确 λ*）。
- `source/source_io/module_parameter/input_parameter.h` +
  `read_input_item_other.cpp` + `source/source_lcao/module_operator_lcao/deltap_lcao.cpp`：
  新增 INPUT `deltap_lambda_init_file`（逐原子 λ 初值文件，覆盖标量
  `deltap_lambda_init`；文件每行一个值，nat 行；缺失/超短 WARNING_QUIT）。
- `tests/deltap_fd_force/run_fd.sh`：重写为双组协议（base 重收敛 λ → λ* 提取 →
  组① 冻结 / 组② 重收敛；STRU 解析含 Direct 晶格矩阵换算；判据自动判定）。
- `tests/deltap_fd_force/h2o1/`：单 H₂O 算例（STRU/KPT 1×1×2/target 3×0）。
- `tests/deltap_fd_force/.gitignore`：忽略 base/、disp_*/、lambda_star.dat、run.log。
- 运行时行为（绝对 τ、H_HK on）**未改**——对照实验的分数 τ / H_HK off 均为
  临时构建，已还原。

## 6. Next steps（T7-c 修订）

1. **实现 B 项（H_HK 解析力）——从"不实现"升级为必做**：∂H_HK/∂R_J 需
   S_dk/D_I 对 R 的导数（w_eff 的 ∂|D|²/∂R 可用现有 snap cal_deri=1 通道），
   力路径 tmp 算符需能访问 k-string 数据。
2. **A2 实现**（∂τ/∂R：-λ⟨P̂⟩(L⁻¹)_{αβ}，用 taud/L 约定）+ λ 诊断打印（对标
   dspin "magnetic force"，`spin_constrain.cpp:857`）。
3. **τ 单位决策（BLOCKER）**：分数坐标（指南约定）vs 保持绝对坐标（现状数值）；
   若改分数，全量回归（SCF 锚点、bn_sampling、relax）并更新 E-field 等效换算。
4. B+A2 后重跑组①/组② FD 复验；应力 FD（变胞 ±0.1%）验 S1。
5. LIMITATION 记录：组② 在 γ≫t（约束未激活）时残差含 λ 分支交换噪声；
   y 方向 PASS 仅反映对称性，不代表一般方向。

## 7. 数据文件（/tmp/fd_t7b）

- `h2o1/` base（绝对 τ）；`g1/` 组①（O1±δ、±2δ、λ×10）；`g1b/` 组① H1/H2±δ；
- `frac/` 分数 τ 组①（base、O1/H1/H2 ±δ）；`nohk/` 分数 τ+H_HK off 组①；
- `g2/` 组②（O1/H1/H2 ±δ，z）；`plain/` 纯 SCF ±δ；`lam0/` λ=0 对照。
- 脚本复跑：`tests/deltap_fd_force/run_fd.sh h2o1 0.005 1 1|2`。

### 3.5 组② 完整 3×3 位移矩阵（脚本复跑结果）

（补充 2026-08-03：`run_fd.sh h2o1 0.005 1 2` 全 18 位移点结果，判据 0.0129 eV/Å）

| 原子 | 方向 | F_FD (eV/Å) | F_ana (eV/Å) | Δ | PASS/FAIL |
|------|------|-------------|--------------|-----|-----|
| O1 | x | +1.12e-5 | +3.3e-6 | 7.9e-6 | PASS |
| O1 | y | +1.77e-5 | +2.0e-6 | 1.6e-5 | PASS |
| O1 | z | +5.863 | -0.7733 | **+6.636** | FAIL |
| H1 | x | +1.321 | -0.3209 | **+1.642** | FAIL |
| H1 | y | +4.8e-8 | -1.0e-6 | 1.0e-6 | PASS |
| H1 | z | -0.6907 | +0.3867 | **-1.077** | FAIL |
| H2 | x | -1.303 | +0.3209 | **-1.624** | FAIL |
| H2 | y | -9.4e-8 | -1.0e-6 | 9.0e-7 | PASS |
| H2 | z | -0.6907 | +0.3867 | **-1.077** | FAIL |

5/9 FAIL。组② 残差比组① 略大（O1-z +6.64 vs 组① +5.33），失败模式一致
（x/z 方向、y 对称零力 PASS）。`script_exit=1`，与"relax 不可用"判定一致。
组② 的 γ≫t 分支交换噪声（H1/H2 γ 在 ±δ 间互换）使残差无意义地大。
