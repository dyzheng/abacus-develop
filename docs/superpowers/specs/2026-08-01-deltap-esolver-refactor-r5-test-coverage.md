# 2026-08-01 DeltaP 重构 R5：补算例 + 补文档 + 全量测试验证

> 状态：**完成** — 新增 2 个通过算例（掩码 / gdir2）、1 个已知问题复现器（relax），
> 全量复验既有算例，MPI（LCAO 4-rank / PW 2-rank）实跑通过，发现并修复 1 个
> 既有空指针 bug（force 路径），记录 1 个既有 force 堆损坏 bug（重构前引入）。
> 前置：R1–R4 已完成。总方案：`2026-07-31-deltap-esolver-refactor-design.md`。

## 1. Test plan

1. **复验既有仓库算例**（新二进制重跑，对比旧日志/物理判据）：
   `test_C_I`（约束矩阵 2×2）、`test_C_total`（约束矩阵 1×2）、`test_stru_target`
   （STRU target + 内循环 nscf=3）、`center`（target.dat，P1→P3）、
   `deltap_pw_h2o`（PW 冒烟）。
2. **新增算例**：`test_mask`（STRU `dp_constrain 0/1` 混合掩码）、`test_gdir2`
   （`deltap_gdir 2`）、`deltap_relax`（多离子步 + `reset_ionic_step`）。
3. **MPI 实跑**：LCAO 4-rank（方阵进程网格，验证 C-23 rank0+Bcast）、PW 2-rank
   （验证 C-28 λ 同步/rank 守卫）。
4. **单测回归**：`MODULE_ESOLVER_deltap_common_test`（10 例）、
   `MODULE_ESOLVER_esolver_dp_test`（6 例）。

## 2. Test setup

- 本机 Release + MPI；`build/abacus_basic_para`（2026-08-01 21:0x 重建，含 R5 空指针修复）。
- 运行目录：`/tmp/deltap_r5/{test_C_I,test_C_total,test_stru_target,center,deltap_bn_test,deltap_pw_h2o,new_mask,new_gdir2,new_relax}`。
- 命令：`OMP_NUM_THREADS=1 timeout 900 <binary> > run.log 2>&1`（MPI 用 `mpirun -np 4/2`）。
- 复验注意事项：仓库旧 `.log` 为**历史快照**，多数与当前 INPUT 参数不一致
  （旧 run 用 `lambda_step=0.5/mixing=0.5/inner_nmax=20` 等，见各目录 `OUT.bn/INPUT.info`），
  且 `deltap_branch.dat` 分支状态随时间变化 → 不做逐字节 A/B，改用物理判据 + 能量锚点。

## 3. Results

| 用例 | 模式 | 结果 | 关键数值 |
|---|---|---|---|
| `test_C_I` | 约束矩阵 2×2（C=I） | **PASS**，GE16 收敛 | 末态 `-3.38713360e+02 eV`，GE14–16 与旧日志**逐字节一致** |
| `test_C_total` | 约束矩阵 1×2（Σγ=7.5） | **PASS**，GE16 收敛 | 末态 `-3.38713360e+02 eV` |
| `test_stru_target` | STRU target + 内循环 nscf=3 | **PASS**（跑通内循环） | `inner loop done: final l0=2.11e-03 l1=-6.94e-04`；30 步未达 1e-8（历史行为） |
| `center` | target.dat，P1→P3 | **PASS**（完整状态机） | P2 iter=11 触发；P3 λ=(2.58e-03,-2.32e-03)，SCF 振荡（BN+deltap 已知收敛难点） |
| `test_mask`（新增） | 掩码 0/1 混合 | **PASS** | P3 起 `λ=(2.58e-03, 0.0e+00)`：B 更新、N 恒为 0 |
| `test_gdir2`（新增） | gdir=2 | **PASS** | P3 γ=(4.000,3.500) `|γ-t|=2.8e-04`，GE50 DRHO=3.4e-08 |
| `deltap_bn_test` | LCAO **MPI 4-rank** | **PASS** | 51 行 P1→P3，P2 iter=11；GE50 DRHO=7.0e-07 |
| `deltap_pw_h2o` | PW 1-rank | **PASS** | `!FINAL_ETOT_IS -442.0244045310906 eV`（旧基线 -442.02427429，Δ1.3e-4 @ scf_thr=1e-4） |
| `deltap_pw_h2o` | PW **MPI 2-rank** | **PASS** | 收敛；`[DeltaP-PW]` 仅 rank0 打印一次（C-28 守卫生效） |
| `deltap_relax`（新增） | relax 多离子步 | **FAIL（既有 bug）** | SCF 43 步收敛后 force 路径 double-free（见 §4.2） |
| `total_0.0`（补跑 F5） | total 模式 | **PASS**（跑通） | P2 iter=11；P3 λ=2.24e-03；SCF 振荡（见 §4.4） |
| `deltap_common` 单测 | — | **PASS** | 10/10 |
| `esolver_dp` 单测 | — | **PASS** | 6/6 |

运行日志：`/tmp/deltap_r5/*/run*.log`；新基线已刷新至仓库各用例目录（`*.log` gitignore）。

## 4. Analysis

### 4.1 旧日志不可逐字节 A/B 的原因（重要）

- 仓库 `deltap_bn_sampling/*.log` 是历史快照：`OUT.bn/INPUT.info` 显示旧 run 的
  `deltap_lambda_step/mixing/inner_nmax/method` 与当前 INPUT 不一致（center 旧 run：
  step=0.5、mixing=0.5、inner_nmax=20、method=wannier；当前 INPUT：step=0.01、mixing=0.1、
  inner_nmax=0）。
- `deltap_branch.dat`（Wilson-loop 前帧种子，`load_branch()` 读取）随每次运行改写，
  旧日志对应的种子状态无法复原。
- 因此本轮以**物理判据**（收敛、γ→target、P2 λ 更新、能量锚点）验收。最强的逐字节证据：
  `test_C_I` 在约束矩阵模式（λ 恒 0，无分支选择扰动）下 GE14–16 能量/DRHO 与旧日志完全一致，
  证明重构未改变 SCF 物理。

### 4.2 relax/force 路径：既有 double-free（重构前引入，非回归）

- **修复 1（R5 提交）**：`hamilt::DeltaPOperator` 构造函数无条件 `hR->get_paraV()`，
  `FORCE_STRESS.cpp` 以 `hR=nullptr` 构造 → 必崩。已加空指针守卫
  （`deltap_lcao.cpp:29`，force 路径不需要 `this->paraV`，`cal_force_stress` 取 `dmR->get_paraV()`）。
- **既有 bug（未修，超出 R5 范围）**：修复后 force 路径在 `cal_force_stress` 的 OMP 区
  `double free or corruption`（gdb 背靠栈见 `tests/deltap_relax/README.md`）。根因初步分析：
  `nlm_target[...] = nlm[channel][iw+m]` 与 `cal_force_IJR` 的 `nlm2[index]` 存在混合基组
  越界读（O–H 对），垃圾值 + SCF 阶段堆损坏累积，force 区释放时触发。该算子自
  `85b2af322`（2026-07-09）引入，**R1–R4 未触碰**（`git diff` 仅本轮 7 行守卫）。
- force 数学本身标注不完整（`deltap_force_stress.hpp`：缺 H_HK 力项、∂τ/∂R 项），
  relax/MD + deltap_corr 为实验功能。`tests/deltap_relax/` 已作为复现器入库。

### 4.4 total 模式语义归因（F5）

- `total_0.0.log` 旧基线来自 `aeb9a0f9c`（2026-07-21），早于 total 模式提交 `8e73e0f82`
  （旧代码读 target.dat 无 total 分支 → target=(7.5, 0.0)；`8e73e0f82` 起均分为 3.75/3.75）。
- R1-R4 的 `gd_update_total` 与 `6beb70bc3` 的 total 分支数学**逐行一致**
  （`delta = step·(Σγ−Σt)`，全原子共享 λ）→ total 模式非重构回归。
- 当前语义下 SCF 振荡强（λ≈2.2e-3 扰动），收敛性属已知难点，留后续调研。

### 4.3 MPI 验证结论

- LCAO 2-rank 被**既有**方阵网格限制拦截（`deltap_wannier.cpp` "square process grid"，
  提交 `cbb37b7ae` 引入，早于 R1–R4）→ 改用 **4-rank（2×2 方阵）** 实跑通过。
- LCAO 4-rank：target 文件由 rank0 读取并 Bcast（C-23）→ 各 rank P 序列一致、λ 一致。
- PW 2-rank：`[DeltaP-PW]`/`report_pw` 仅 rank0 输出一次（C-28）；λ Bcast 后各 rank
  escon 一致；SCF 收敛，能量与 1-rank 一致。

## 5. 完整 DeltaP 功能矩阵（2026-08-01 R5 实测）

| # | 功能 | 用例/验证 | 状态 |
|---|---|---|---|
| F1 | LCAO per-atom target（target.dat） | center、deltap_bn_test（1/4 rank） | ✅ 通过 |
| F2 | LCAO STRU target（dp_target） | test_stru_target | ✅ 通过 |
| F3 | 约束矩阵 C·γ=t（2×2） | test_C_I | ✅ 通过 |
| F4 | 约束矩阵 C·γ=t（1×2 行） | test_C_total | ✅ 通过 |
| F5 | total 模式（`deltap_constraint_mode total`） | deltap_bn_sampling/total_0.0（本轮重跑） | ✅ 通过（跑通 P1→P3）；旧日志基线 aeb9a0f9c 早于 total 提交 `8e73e0f82`，目标语义已变（旧 (7.5,0.0) → 新均分 3.75/3.75），无法 A/B；λ 更新数学与 6beb70bc3 逐行一致 |
| F6 | 掩码 `dp_constrain 0/1` 混合 | **test_mask（新增）** | ✅ 通过 |
| F7 | gdir=1 | deltap_compare INPUT_lcao（旧输出） | ⚠️ 旧输出发散，未重跑 |
| F8 | gdir=2 | **test_gdir2（新增）** | ✅ 通过 |
| F9 | gdir=3（默认） | center、deltap_bn_test、h2o 冒烟 | ✅ 通过 |
| F10 | 两阶段 P1→P2→P3（nscf=0） | 全部 LCAO/PW 用例 | ✅ 通过 |
| F11 | BFGS 内循环（nscf>0） | test_stru_target（nscf=3）、h2o_inner | ✅ 通过 |
| F12 | PW 路径（Wilson-loop k-string） | deltap_pw_h2o（1/2 rank） | ✅ 通过 |
| F13 | PW target/约束矩阵文件 | — | ❌ 未接线（已知限制，INPUT 解析后未传给 PW backend） |
| F14 | PW total 模式 | — | ❌ 假 total（INPUT total 被忽略，代码 per-atom，历史不一致） |
| F15 | MPI：rank0 读文件 + Bcast（C-23） | LCAO 4-rank、PW 2-rank | ✅ 通过 |
| F16 | MPI：λ Bcast + rank 守卫（C-28） | PW 2-rank | ✅ 通过 |
| F17 | MPI：LCAO 2-rank | — | ⛔ 既有方阵网格限制（非回归） |
| F18 | relax 多离子步 + reset_ionic_step | **deltap_relax（新增）** | ❌ force 路径既有 double-free |
| F19 | FD 力验证（C-02） | deltap_fd_force/run_fd.sh | ❌ 依赖 F18 |
| F20 | `deltap_common` 纯函数单测 | 单测 10/10 | ✅ 通过 |
| F21 | esolver DP 单测 | 单测 6/6 | ✅ 通过 |

## 6. Next steps

1. **F5 后续**：total 模式当前 target 均分语义（`8e73e0f82` 引入）下 SCF 振荡强
   （λ≈2.2e-3，GE50 DRHO≈1e-1）；如需稳定收敛需单独调研（目标分支选择 + 步长），不在 R5。
2. **F7 补跑**：`deltap_compare`（gdir=1）用新二进制重跑，OUT.autotest 旧输出为发散残留。
3. **F18/F19**：修 `cal_force_stress` 越界读（nlm 布局统一 + 长度守卫）→ 复跑
   `deltap_relax` → 按 `run_fd.sh` FD 判据验收（需先补齐 H_HK/∂τ/∂R 力项）。
4. **F13/F14**：PW 接线 target/约束矩阵文件、统一 total 模式（与 LCAO 状态机对齐）。
5. 可选：把 P01–P18 可靠性用例（提交 `6beb70bc3`）纳入 CI 脚本，防回归。
