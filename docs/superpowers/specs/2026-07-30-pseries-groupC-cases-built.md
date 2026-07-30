# P 系列算例构建 C 组（P06/P07/P10/P16/P17/P18）

> 日期：2026-07-30 ｜ 分支：feat/deltap ｜ 依据：`2026-07-30-pseries-test-cases-design.md` §4（C 组：收敛/等效）
> 状态：构建完成，未运行 ABACUS

## 1. 测试计划

为 C 组 6 个测试构建自包含算例目录（README.md + run.sh + cases/），覆盖：
- P06 盒尺寸收敛（μ 有效 / α 阻塞）、P07 基组与截断双收敛（无阻塞）、P10 E–D 曲线（阻塞）、
  P16 PW 约束核整改验证（无阻塞）、P17 场致弛豫对照（阻塞）、P18 实空间密度对照（阻塞）。

验证项（不运行 ABACUS）：
1. `bash -n` 全部 6 个 run.sh；
2. 内嵌 awk 程序语法检查（逐程序提取后用 `awk -f` 对 /dev/null 编译）；
3. 关键分析逻辑用合成数据冒烟：efield 二次拟合、cube 平面平均 + Pearson/RMS（python3 与 awk 双路径）、
   STRU_ION_D 几何解析、NUMERICAL_ORBITAL 存在性检测；
4. 提取键与真实输出样例比对（`[rawG]`、`[DeltaP-PW]`、`E_KohnSham`、deltap_results.dat）。

## 2. 测试设置

- H₂O 标准几何：O 居盒心，H 在 (c±0.7571, c, c+0.5858) Å（r_OH=0.9573 Å，∠HOH=104.5°，C2 沿 z）。
- 盒：P06 L=12/15/18/21/24 Å；P07 15 Å；P10/P17/P18 30 Bohr（15.873 Å）；P16 15/30 Bohr 双盒。
- STRU：LATTICE_CONSTANT 1.8897261254578284，LATTICE_VECTORS 用 Å，Cartesian_angstrom。
- 赝势/轨道：O.upf/H.upf，O_gga_6au_100Ry_2s2p1d.orb / H_gga_6au_100Ry_2s1p.orb
  （已核实存在于 /root/pporb/apns-{pseudopotentials,orbitals-efficiency}-v1）。
- LCAO 基线（任务书公共约定）：ecutwfc 100、scf_thr 1e-8（P18 收紧 1e-9）、scf_nmax 200、
  mixing_beta 0.4（负 λ 自动 0.3）、genelpa、symmetry 0、deltap_rm/onsite_radius 6.0、
  lambda_step 0、lambda_mixing 0.1、inner_thr 1e-2、total 模式。
- PW：ecutwfc 80（P07 扫 40/60/80/100）、nbands 8、berry_phase 1、gdir 3、deltap_switch true。
- efield：efield_flag 1、efield_dir 2（z）、dip_cor_flag 1。
- 源码事实核对：
  - PW 输出行 `[DeltaP-PW] ... γ_total=... rad λ_avg=...`（source_pw/module_pwdft/deltap_pw.cpp:284）；
  - relax 每个离子步覆写 `OUT.*/STRU_ION_D`（source_relax/relax_driver.cpp:118-131，print_stru_file Direct）；
  - `OUT.*/STRU.cif` 是运行前初始结构（source_esolver/esolver_fp.cpp:55），P17 不可用；
  - `out_chg 1`（nspin=1, scf）产物 `OUT.*/chg.cube`（source_io/module_ctrl/ctrl_output_fp.cpp:61-77）；
  - PW/LCAO 当前均无 SMO 覆盖率直接输出键 → P16 留 TODO。

## 3. 结果

| 项 | 结果 |
|---|---|
| `bash -n` 6 个 run.sh | 6/6 通过 |
| 内嵌 awk 语法编译 | 全部通过（夹具报出的"错误"均为空输入运行时除零，非语法错误；P10 已补零分母保护） |
| P06 分析 awk（μ/α_ref 五点二次拟合/α_dp） | 合成 data.tsv 冒烟通过 |
| P18 cube 分析 python3 路径 | 合成 4×4×6 cube：Pearson=1.000000、RMS=0（设计值）✓ |
| P18 cube 分析 awk 回退路径 | 同上，1.000000 / 0 ✓ |
| P17 STRU_ION_D 解析 + cases/STRU 参考几何 | 合成文件 0.958270/104.385°；STRU 0.957268/104.539° ✓ |
| P07 NUMERICAL_ORBITAL 检测 | 正确列出 tzdp 两个轨道文件 ✓ |
| 提取键 vs 真实样例 | `[rawG] Σγ_raw=`、`γ_total=`、`E_KohnSham`（$2）格式一致 ✓ |

## 4. 分析

无失败项。判据判定逻辑（check_le/blocked/SUMMARY/退出码）与 A/B/D 组同构。

## 5. 假设与偏差说明

1. **P17 的 E 对应值**：任务书 F1 工作值 E=−πλ/(2a) 使 λ=±0.02/±0.04 ↔ E=∓0.001047/∓0.002094 Ha/Bohr，
   与设计文档 P17 §3 的 E=±0.002/±0.004 不一致（即 F1 的 1/2 因子之争）——P17 本就阻塞于 F1，
   按任务书实现并把 e_of_lam 集中在 run.sh 头部常量区，备忘录定稿后单点修改。
2. **P10 κ 三方对比**：三条 κ 量纲不同（Ry/rad² vs Ry/Ha²），run.sh 同时输出各自 α 等价量供比对；
   备忘录定稿后改统一单位直接比 κ。
3. **P18 分析降维**：判据在 z 轴平面平均 Δρ(z) 剖面上计算（任务书允许）；逐体素 3D 相关留作升级路径。
4. **P18 只认 chg.cube**：旧版 `<suffix>-CHARGE-DENSITY.restart` 格式不同不解析，缺失则 SKIP（README 说明）。
5. **P16 LCAO 对照组 radius 固定 6 Bohr**：LCAO 无 PW 式覆盖率赤字（R6 背景），"同设置"理解为同盒/同 λ/同收敛。
6. **P16 覆盖率**：无直接输出键，脚本依次尝试 run.log 关键词、deltap_results.dat 头部，均失败打印 N/A + TODO。
7. **P07 PW 偶极**：用 `[DeltaP-PW] γ_total`（wrapped）而非 berry_phase 输出；J3 跨通道比较在 γ 域做 mod 2π
   最小差后换算 μ（README 说明分支边界风险）。
8. **PW 显式 nbands 8**（参考 tests/deltap_h2o_polarizability/pw_total 既有算例）。
9. **P17 scf_thr 用 1e-8**（任务书 LCAO 基线），设计文档"SCF/力阈值必须收紧"由 force_thr_ev=1e-3 承担。
10. **P06/P10 α_DeltaP 差分窗 ±0.02 Ry**、P10 λ 七点按任务书（与设计文档 P10 §3 一致）。

## 6. 下一步

1. 冒烟运行 P07（无阻塞，tzdp 层级 + PW ecut40）验证端到端提取；
2. F1/F2 备忘录定稿后回填 P06/P10/P17/P18 常量区并启用阻塞判据；
3. P07 dzp/qzdp 层级轨道文件需用户补充至 $ORBITAL_DIR（缺失自动 SKIP）；
4. 首次实跑 P16 核对 r=20 Bohr 大半径 SMO 的 SCF 稳定性。
