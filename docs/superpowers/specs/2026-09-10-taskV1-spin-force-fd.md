# 2026-09-10: Task V1 —— 单自旋通道力 FD（stationary4，PW 先行）

> 依据：`docs/superpowers/plans/2026-09-09-mixed-force-fd-and-stageB.md` Task V1。
> 前置事实：2.6 的 18 腿力 FD **全部是电荷通道**；自旋通道力（M6 核的 spin 折叠）
> 只有单测（`SpinChannelReadsMagnetization`、A5 混合牛三）覆盖，**从未过 FD**。
> 判决门 G-V1：9/9 轴 PASS + 净力标准检查，判据 0.0128555 eV/Å 不豁免。

## 1. 测试方案（Test plan）

**目标**：判决单自旋通道（`constraint_type spin`，nspin=2）解析力是否等于能量导数。

**协议（stationary4 沿用，写死）**：
- 体系：`tests/01_PW/212_PW_constraint_h2o_spin`（H2O，15 Å 盒子，nspin=2，
  STRU 中 O 的起始 mag=0.5，约束靶 `{"targets": [0.1], "atoms": [[0]]}`，delta 模式）；
- 处方：ecutwfc=100 / ecutrho=400 / scf_thr=1e-8（R7 网格前提）、δ=0.005 Bohr、
  **绝对模式冻结 t\***（首条 audit 的 t）、**raw-E 口径**（`!FINAL_ETOT_IS`，
  禁用 E′=E−μt\*）；
- F_FD = −(E₊−E₋)/(2δ)；F_ana = R0 约束态打印 TOTAL-FORCE（eV/Å）；判据 0.0128555 eV/Å；
- Step 2 先跑 `ONLY="0_2"`（O-z，∂m/∂R 最强方向）冒烟，再跑 18 腿。

**脚本改动（V1 Step 1，唯一代码改动）**：`tests/constraint_fd_force/tools/run_constraint_fd.sh`
1. `CASE=<dir>` 覆盖测试例目录（默认仍是 211/212_NAO 电荷例）；
2. `nspin` / `constraint_type` / `nelec` 从该例 INPUT 继承（自旋例 → nspin 2 + spin），
   缺省回退电荷口径 → 211/212_NAO 的 INPUT **逐字节不变**（回退自测见 §3.1）；
3. `TEST_FORCE=1` 额外写 `test_force 1`，用于 2.7 标准检查（净力/补偿前力）；
4. 新增 `std_checks()`：从 running log 的分项力块算 Σ CONSTRAINT 力、Σ 补偿前力、
   compen、Σ 打印总力与 max|pre−compen−printed|。

**Step 4 标准检查（2.7 起永久）**：全轴净力、Σ CONSTRAINT 力（平移不变性）、
F_ana 补偿前后双值入档。

## 2. 测试设置（Test setup）

### 2.1 脚本改动验证（轻量，本地产物）

| 项 | 方法 | 结果 |
|---|---|---|
| 电荷路径 INPUT 逐字节回退 | 用 `git show HEAD:` 取改动前脚本，抽出 `write_input()` 在同一 harness 下生成 INPUT，`diff` 改动前后（delta 与 absolute 两种模式） | **PASS**：`diff` 无输出（两种模式均逐字节一致） |
| 自旋路径 INPUT | 同上，`CASE_NSPIN=2 CASE_CTYPE=spin TEST_FORCE=1` | **PASS**：新增 `nspin 2` / `constraint_type spin` / `test_force 1`，其余行与电荷例一致 |

### 2.2 冒烟（ONLY="0_2"）

| 项 | 值 |
|---|---|
| 命令 | `CASE=tests/01_PW/212_PW_constraint_h2o_spin TEST_FORCE=1 ONLY=0_2 bash run_constraint_fd.sh pw 0.005 4 2` |
| 二进制 | `build/abacus_basic_para`（A5/A6 后，2026-09-09 22:17） |
| 并行 | np4，MAXJOBS=2 |
| 工作目录 | /tmp/cfd_pw_<id> |
| 开始/结束 | 2026-09-10 10:39:51 → 11:03:33（**23 min 42 s**） |

## 3. 结果（Results）

### 3.1 脚本自测

harness（`/tmp/v1_selftest/harness.sh`）把改动前（`git show HEAD:...`）与改动后的
`write_input()` 抽到同一环境里，分别生成 delta / absolute 两个 INPUT 再 `diff`：

```
$ bash harness.sh /tmp/old_fd.sh /tmp/v1_selftest/old pw
$ bash harness.sh tests/constraint_fd_force/tools/run_constraint_fd.sh /tmp/v1_selftest/new pw
$ diff old.delta new.delta && diff old.abs new.abs
LEGACY INPUT BYTE-IDENTICAL (charge path)
```

自旋路径（`CASE_NSPIN=2 CASE_CTYPE=spin CASE_NELE=8 TEST_FORCE=1`）生成的 absolute INPUT：

```
suffix      s
calculation scf
basis_type  pw
ecutwfc     100
ecutrho     400
scf_thr     1e-8
scf_nmax    300
nbands      8
symmetry    0
init_wfc    atomic
init_chg    file
read_file_dir /tmp/restart
nspin       2
nelec       8
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     0.4
pseudo_dir  /root/abacus-develop/tests/PP_ORB
constraint        true
constraint_type   spin
constraint_weight_type becke
constraint_target_file constraint_target.json
constraint_target_mode absolute
constraint_mu_max  5.0
constraint_thr     1e-4
cal_force       1
test_force      1
```

结论：电荷/NAO 既有路径零扰动（逐字节），自旋例按 `CASE=` 显式选择，
nspin/constraint_type 由测试例 INPUT 数据驱动，无硬编码分支。

### 3.2 冒烟：base（R0，delta→冻结 t\*）

- 工作目录 `/tmp/cfd_pw_l2uZ`，np4，2026-09-10 10:39:51 → 11:03:33（**23 min 42 s**）。
- delta 模式参考相：m_ref(O) ≈ −1.4e-6 μB（与 212 例 README 在 ecutwfc=20 的
  m_ref=5.9e-6 同量级，只是网格/口径差异）；外环一次约束步即收敛。
- **t\* = 0.09999858905 μB**（首条 audit，= Q_free(R0)+0.1）
- **μ\* = −0.07188979607 Ry**（212 例在 ecutwfc=20 + scf_thr=1e-7 的参考值 −0.07234，
  差距 0.6%，量级一致——自旋通道 μ 为负的符号模式与电荷通道相同，符合 README 记录）
- E0（raw `FINAL_ETOT_IS`）= **−466.9021000194886710 eV**
- 末次 audit：q=0.09996454448、res=−3.404457591e-05、maxdev=2.22e-16，
  `[constraint] final status: CONVERGED (targets reached within 0.0001 e)`

**F_ana（R0 约束态打印 TOTAL-FORCE，eV/Å）**

| atom | x | y | z |
|---|---|---|---|
| O | +0.0000847 | −0.0001651 | **−0.7259442** |
| H1 | −0.5037454 | +0.0000826 | +0.3629653 |
| H2 | +0.5036607 | +0.0000825 | +0.3629789 |

**标准检查（`test_force 1` 分项力块）**

| 项 | Σx | Σy | Σz |
|---|---|---|---|
| Σ CONSTRAINT force (eV/Å) | −0.000000 | +0.000000 | −0.023966 |
| Σ 补偿前总力 (eV/Å) | +0.000168 | −0.000286 | **+0.015392** |
| compen (eV/Å) | +0.000056 | −0.000095 | +0.005131 |
| Σ 打印总力 (eV/Å) | +0.000000 | −0.000000 | +0.000000 |

- 一致性自证：`max|pre − compen − printed| = 1.75e-07 eV/A`（打印精度 1e-10 × 3 原子
  的量级），说明分项块与总力块自洽、`std_checks` 解析正确。
- **净力（补偿前）Σz = +0.0154 eV/Å ≈ 0**（打印的 Σz=0 是 compen 均值的构造结果，
  不是判据；真正有信息量的是补偿前的 +0.0154）。
- 对照：电荷通道 R0 同项（`/tmp/cfd_fixedmu/oz_0`，修复后）Σ 补偿前 z = +0.0089 eV/Å；
  修复前同目录 `.old` = **+11.62 eV/Å**（即 2.6 的 μw-Pulay 缺失指纹）。
  自旋通道 +0.0154 eV/Å 与修复后电荷通道同量级（1e-2 eV/Å），**无缺失 Pulay 特征**。
- Σ CONSTRAINT force（z）自旋 = −0.024 eV/Å，而电荷通道同样为 +6.19 eV/Å（修复前后
  都是 6.19，说明该项本身不是平移不变量，其非零部分由 SCC 项配平）——故该项记录为
  诊断量，不作判据；判据看补偿前净力。

### 3.3 冒烟：O-z 两腿 + 判据（2026-09-10，PASS）

| 项 | 值 |
|---|---|
| 命令 | `CASE=tests/01_PW/212_PW_constraint_h2o_spin TEST_FORCE=1 ONLY=0_2 bash run_constraint_fd.sh pw 0.005 4 2` |
| 二进制 | `build/abacus_basic_para`（Debug） |
| F_FD (O-z) | **−0.72247438 eV/Å** |
| F_ana (O-z) | **−0.72594423 eV/Å** |
| \|d\| | **0.0034698 eV/Å < 0.0128555**（判据），**3.7× 富余 → PASS** |
| 净力标准检查 | Σ 补偿前总力 z = +0.0154 eV/Å ≈ 0（电荷通道修复后参考 +0.0089；修复前 +11.62 ⇒ 自旋通道**无 μw-Pulay 缺失特征**） |
| 一致性自证 | max\|pre − compen − printed\| = 1.75e-07 eV/Å |

**判定**：单自旋通道（`constraint_type spin`，nspin=2）解析力在 O-z 轴与能量导数一致，
G-V1 的 9/9 轴判决完成 1/9 轴（最强的 ∂m/∂R 方向）。

### 3.4 18 腿重跑（2026-09-14，**未完成**）

用户批复"并行补完 V1"后重启：同载体（PW `212_PW_constraint_h2o_spin`）、同协议
（frozen t* + μ 重收敛 + raw-E）、只把二进制换成 **release**（`build_rel/abacus_basic_para`），
np4 / MAXJOBS=3 / OMP_NUM_THREADS=1。

**base（R0）结果（已完成）**：

| 项 | Debug（2026-09-10） | Release（2026-09-14） |
|---|---|---|
| t\* (μB) | 0.09999858905 | 同（首条 audit） |
| μ\* (Ry) | −0.07188979607 | **−0.07188979603** |
| E0 (eV) | −466.9021000194886710 | 同量级 |
| F_ana(O-z) (eV/Å) | −0.7259442 | **−0.72595136860**（差 7.2e-6） |
| Σ compen z (eV/Å) | +0.005131 | +0.005127 |
| max\|pre−compen−printed\| | 1.75e-07 | **1.750e-07** |
| base 墙钟 | 1422 s | **1247 s** |

- **跨二进制一致性**：μ\* 一致到 **4e-11 Ry**、F_ana(O-z) 到 **7.2e-6 eV/Å**、
  std-check 逐位复现 —— release 重构没有改动自旋通道的物理。
- **release 提速只有 1.14×**（1422 s → 1247 s），远低于 `2026-09-09-fd-cost-analysis.md`
  里"3–10×"的预期（那是按对角化密集的 LCAO 载体估的）。⇒ **"release 即可把 18 腿压到
  1–2 h"在本 PW/ecut=100 载体上不成立**；真正省时的是换 LCAO 载体或 fixed-μ 腿。
- 18 腿 14:54 起跑，**未跑完**；产物目录是 `mktemp -d /tmp/cfd_pw_XXXX`，随会话环境
  清理**全部丢失**（仓库内无归档）⇒ 本轮无 leg 级结果，需重跑。

## 4. 分析（Analysis）

- **G-V1 的状态**：O-z 轴 PASS（自旋通道解析力 = 能量导数，3.7× 富余），但其余 8 轴
  仍缺；两次 18 腿起跑（09-10 / 09-14）都**没有留下归档结果**，所以力相关物理
  （约束 relax/MD）目前仍不能解锁——这与评审给出的"闸门 1"判断一致。
- **可复现性**：base 两次（debug/release）逐位级一致（μ\* 4e-11 Ry），说明 base 本身
  可复现；问题纯粹是**跑批持久化**。
- **根因（流程，不是物理）**：`tests/constraint_fd_force/tools/run_constraint_fd.sh:70`
  用 `mktemp -d /tmp/cfd_${BASIS}_XXXX` 作工作目录，且**不像新脚本**
  （`run_inner_thr_scan.sh`/`run_ct_scan.sh`）那样把每条腿的 audit 归档进 `results/`。
  ⇒ 长跑一旦中断/环境清理，18 腿的计算全部作废。这是 09-10 那次"无归档结果"的同一个坑。
- **成本画像（实测）**：base ~21 min（release），其中含自由相 + 约束外环；18 腿
  /MAXJOBS=3 的实测未完成，但按 base 量级估计 ≥1 h。若换 LCAO 载体（2.6 的 charge
  通道已在该载体上 18 腿 PASS）应显著更省，且能与既有 pair 直接对照。

## 5. 下一步（Next steps）

1. **先修跑批持久化**（1 行级）：给 V1 runner 加 `RESDIR`（或 `WORKROOT` 覆盖 + 归档），
   leg 级 audit / 力分量 / 时间 一律落到仓库内（对齐 `run_ct_scan.sh` 的写法）；
2. 重跑 18 腿（建议 LCAO 载体，PW 作对照 base 已有），补齐 9/9 轴；
3. 之后才能对"约束 relax/MD"类力相关物理解锁；TM d 矩解读仍需 II-1b/4b 先收口。
