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

### 3.5 归档修复（V1 补跑前置，入库 `62819bd46`）

- runner 新增 `RESDIR`/`TAG`：base 与每条腿的 **audit + 力块 + 计时**逐条落到
  `tests/constraint_fd_force/results/<TAG>/`（`summary.txt` 记 case/二进制/网格/协议/
  E0/t*/μ*/base 力/FD 表，另存 `legs.tsv`）——直接堵住 §3.4 的 `/tmp` 丢失坑；
- `FIXED_MU`（默认 1 = 腿冻结 μ=base μ*，经 `ABA_CONSTRAINT_FIXED_MU`；0 = legacy 重优化）
  与 `KS_SOLVER`（可选钉求解器）两个开关，默认值下 INPUT 逐字节不变；
- `std_checks` 改为**块名自动发现**：PW 打 `LOCAL/NONLOCAL/NLCC/ION/SCC`，
  LCAO 打 `OVERLAP/T_VNL/VL_dPHI/VL_dVL/EWALD/NLCC/SCC` —— PW 侧求和块集与修复前逐字相同；
- 新载体 `tests/constraint_fd_force/cases/212_NAO_constraint_h2o_spin`（212 几何、
  LCAO/gamma-only、nspin 2、O `mag 0.5`；runner 从该 INPUT 继承 nspin/constraint_type/nelec）；
- 回归自证：`ABACUS=/bin/true` 短跑对比修复前后生成的 base INPUT ——
  电荷（PW/LCAO）与自旋两条路径均**逐字节一致**。

### 3.6 协议等价性：spin 通道 fixed-μ vs 重优化-μ（首发对照，LCAO O-z，R7）

| 协议 | F_FD (eV/Å) | F_ana (eV/Å) | \|d\| | 判定 |
|---|---|---|---|---|
| fixed-μ（= base μ*=−0.08212212424 Ry） | −1.336305240734221 | −1.3364833783 | 1.781e-4 | PASS |
| 重优化-μ（legacy） | −1.3365405564557589 | 同上 | 5.718e-5 | PASS |

- 两协议 F_FD 差 **2.35e-4 eV/Å**（判据 0.0128555 的 1.8%，相对差 0.018%）
  ⇒ **等价性在自旋通道成立**（电荷通道此前已双证），全轴扫描可安全采用 fixed-μ；
- 成本：fixed-μ 8m28s vs 重优化 16m20s（同 base+2 腿）≈ **1.9×**，与"省 50%"一致。

### 3.7 结果 A：LCAO spin 全轴 9/9 PASS（fixed-μ + R7 + release，np4 / MAXJOBS=2）

base：`t*=0.1`、`μ*=−0.08212212424 Ry`、`E0=−466.2989480199703394 eV`；
Σ 补偿前 z = **+0.000490 eV/Å ≈ 0**（无 μw-Pulay 缺失特征），一致自证 1.286e-07 eV/Å。
墙钟 **27m36s**（base + 18 腿）。

| 原子轴 | F_FD (eV/Å) | F_ana (eV/Å) | \|d\| | 富余 |
|---|---|---|---|---|
| O-x | −1.2985349e-05 | −5.2948e-05 | 3.996e-5 | 322× |
| O-y | −1.1355284e-05 | −5.1629e-05 | 4.027e-5 | 319× |
| **O-z** | −1.336305240734 | −1.3364833783 | **1.781e-4** | **72×** |
| H1-x | −0.886644656650 | −0.8865516363 | 9.302e-5 | 138× |
| H1-y | −1.8362117e-07 | +2.5815e-05 | 2.600e-5 | 494× |
| H1-z | +0.668377914008 | +0.6682416838 | 1.362e-4 | 94× |
| H2-x | +0.886644388995 | +0.8866045843 | 3.980e-5 | 323× |
| H2-y | −1.0934128e-07 | +2.5815e-05 | 2.592e-5 | 496× |
| H2-z | +0.668377710911 | +0.6682416945 | 1.360e-4 | 95× |

**9/9 PASS，最大残差 1.781e-4（O-z）= 判据的 1/72。**

### 3.8 结果 B：PW 冒烟 2 轴 PASS（dav_subspace + release + fixed-μ）

base：`t*=0.1000010364`、`μ*=−0.07189321089 Ry`、`E0=−466.9020975520369348 eV`；
Σ 补偿前 z = +0.013651 eV/Å（与修复后电荷通道同量级 1e-2），一致自证 6.120e-08 eV/Å。

| 原子轴 | F_FD (eV/Å) | F_ana (eV/Å) | \|d\| | 富余 | 墙钟 |
|---|---|---|---|---|
| **O-z**（权重导数最强） | −0.722392527036 | −0.72659147690 | 4.199e-3 | 3.1× | 28m17s |
| **H1-x**（历史变化最大轴） | −0.503574974627 | −0.50345528700 | 1.1969e-4 | 107× | 27m50s |

- 两轴均 PASS ⇒ PW 嵌入路径在自旋通道**无新的力签名**（净力指纹）。
- **dav_subspace 未带来收益**：每轴 ~28 min，而 09-10 的 debug+DiagoCG 同轴冒烟为 23m42s
  ⇒ 反而约 1.2× 慢。重设计 §1 的"dav_subspace 2–5×"在本 PW/ecut=100 小分子载体上
  **未被实测支持**（与 release 仅 1.14× 同族：该载体的瓶颈是基组规模/厂商库，不是求解器实现）。
- 跨求解器 base 差异：dav_subspace μ*=−0.07189321089 vs DiagoCG(release) −0.07188979603
  （Δ3.4e-6 Ry，0.005%）；FD 是差分量，两求解器下判决都 PASS。

## 4. 分析（Analysis）

- **G-V1 判决：闭合（在声明的覆盖包络下）**。覆盖 = LCAO 载体**全 9 轴 PASS**
  （最大残差 1.781e-4，72× 富余）+ PW 载体**冒烟 2 轴 PASS**（O-z 权重导数最强、
  H1-x 历史变化最大轴，且净力指纹无异常）。减轴依据与 2.6 电荷通道同型：
  机制已根因（自旋通道与电荷共用 M6 折叠核）、最坏轴已验、PW 侧风险签名
  （μw-Pulay 缺失 → 净力 +7.08 eV/Å 级）由净力标准检查兜底（本两轴 ≈ +0.014 eV/Å）。
- **力相关物理（约束 relax/MD）过闸门**；对外的受限声明保持不变：自旋通道力只在
  "LCAO 全轴 + PW 2 轴"包络内验证，包络外须按同一协议补验。
- **成本结论（两处更正）**：① `release` 仅 1.14×（已入 cost-analysis §5）；
  ② `dav_subspace` 在本载体上约 1.2× **慢**，"2–5×"未获实测支持。真正的杠杆是
  **载体选择**——本轮 LCAO 全 18 腿 27m36s vs PW 单轴 ~28 min。
- **可复现性**：本轮三次独立 base（LCAO fixed/reopt 两次 + PW 一次）与 09-14 的
  PW base 交叉一致（μ* 同求解器下逐位级一致、跨求解器 0.005%），归档 41 个证据文件。
- **流程**：§3.5 的 `RESDIR` 归档是本轮不再丢数据的直接原因；协议开关（fixed-μ）
  把 18 腿压到半小时级，使"补跑"从 8.7 h 量级变成一次会话内可完成的工作。

## 5. 下一步（Next steps）

1. **混合通道（charge+spin 同原子）力 FD** ——同族最后一环：LCAO 全轴 + PW 冒烟
   （213 几何；协议与守卫复用本轮）；顺带补混合通道的 fixed-vs-重优化对照 1 条；
2. **力相关物理解锁后的首个冒烟**：约束 relax 单步（几何驱动）——验证跨离子步的
   λ/μ 生命周期（C-12/C-13 族）在自旋通道不复发；
3. 力矩 FD 的 LCAO 复查（PW 已 PASS，LCAO 未做；可选、低成本）；
4. 回到主队列：4b 半径敏感性 + II-1 重锚定 → 阶段 B 立项评审。
