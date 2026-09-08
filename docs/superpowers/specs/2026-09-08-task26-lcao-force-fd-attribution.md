# Task 2.6.2 LCAO 力 FD 归因——测试方案 + 已完成部分结果（供审查）

> 日期：2026-09-08
> 队列：二期 Task 2.6.2（PW≡LCAO 三判决之一）·LCAO 力 FD。
> 前置：PW 力 FD 已 9/9 PASS（SCC-μw 根因修复，见
> `2026-09-08-task26-r7-scc-fix.md`）。本文件记录 LCAO O-z 冒烟腿 FAIL 的
> 归因进展与**待审定的剩余实验方案**。
> 状态：归因中，判据未过、未豁免。**重型测试已按用户要求停止**；下文
> §5 的实验 (1) 只跑了一半即被叫停。

---

## 1. 测试方案（判据与协议，写死不放松）

来自计划 `docs/superpowers/plans/2026-08-31-task26-judgment-validations.md`
Task 2.6.2 处方（R7）：

- 网格/收敛：`ecutwfc=100`、`ecutrho=400`、`scf_thr=1e-8`（scf_nmax 300）；
- 位移：δ = 0.005 Bohr = 0.0026458861 Å，中心差分；
- 靶点：绝对模式冻结 `t*`，每腿重优化 μ（`constraint_thr=1e-4`）；
- 观测量：`FINAL_ETOT` 原始值中心差分（raw-E 口径，PW 判决同款；
  FD 工具 `run_constraint_fd.sh` 已改为 raw-E 输出）；
- 判据：`|F_FD − F_ana| < 0.0128555 eV/Å`（5e-4 Ry/Bohr），不豁免；
- F_ana：R0 约束态（μ* 收敛、|Q−t|≈1e-4 内）打印 TOTAL-FORCE。

LCAO 归因还用到两个实验分支：

- **固定-μ 判决腿**（`ABA_CONSTRAINT_FIXED_MU`=base μ* 冻结，常数外势，
  无外环）：把"重优化-μ 路径的 μ(R) 漂移"从分歧中切掉；
- **plain 对照**（同网格同 δ、无约束）：验证 LCAO 力/能量通用机制自洽，
  把"约束特有 vs LCAO 通用"切分。

### 1.1 正在跑但被叫停的实验（轻量判别）

为定位差值的空间结构，设计了 O-x 与 H1-x 各一对 FD 腿（4 次 SCF，
np4，每腿约 5–10 min，估计总 ~30–40 min）：

- `ox_minus/ox_plus`：O 原子沿 x（对称方向，Q_O 对 x 响应≈0）；
- `h1x_minus/h1x_plus`：H1 沿 x（大分量 F_x≈−4.43 eV/Å，但 Q 响应弱）。

目的：若差值是"z/电荷响应方向特有"（O-z 腿差 1.27 eV/Å 而 O-x/H1-x 腿差
≈0）→ 指向 μ·∂Q/∂R 类项；若差值在 H1-x 上也 ~1.27（均匀常数）→ 指向
COM/约定类项。**该实验未跑完（ox_minus 进行中被终止），数据作废**，
目录 `/tmp/cfd_lcao_morelegs2/` 仅保留了可复用的 STRU/INPUT 生成脚本
（`/tmp/ab_mkleg.py`、`/tmp/cfd_lcao_more_legs2.sh`）。

---

## 2. 测试设置

- 体系：H₂O，15 Å 立方盒，O 于 (7.5,7.5,7.5)，H1/H2 于 z=8.086；
  γ-only、np=4、串行 OMP=1；数值轨道 `O_gga_7au_60Ry_2s2p1d` /
  `H_gga_8au_60Ry_2s1p`；smearing gauss σ=0.002。
- 约束：charge、Becke 权重、O 原子、绝对靶 `t*=6.505559879`；
  从 base（R0 约束收敛态，μ*=−0.2193905904 Ry）密度 restart。
- 二进制：`/root/abacus-develop/build/abacus_basic_para`
  （2026-09-08 11:04 构建，含 SCC-μw 修复）。
- 数据目录：
  - R7 协议腿：`/tmp/cfd_lcao_4lt7/{base,disp_s_0_2_minus,disp_s_0_2_plus}`；
  - 固定-μ 腿：`/tmp/cfd_lcao_fixedmu/{oz_minus2,oz_plus2,oz_0,oz_0_tf}`；
  - plain 对照：`/tmp/cfd_lcao_plain/{plain_r0,plain_minus,plain_plus}`。

---

## 3. 结果（已完成部分）

### 3.1 R7 协议 O-z 腿（μ 每腿重优化）——FAIL

| 量 | R0（base） | R−δ | R+δ |
|---|---|---|---|
| FINAL_ETOT (eV) | −466.2070054139356 | −466.2137612395323 | −466.1998489632887 |
| μ (Ry) | −0.2193905904 | −0.2162621247 | −0.2227964099 |
| Q (res) | 6.505467804 (−9.2e-5) | 收敛 | 收敛 |

- `F_FD = −(E₊−E₋)/(2δ) = −2.629 eV/Å`；
- `F_ana(R0) = −1.3602401011 eV/Å`（打印 TOTAL-FORCE O-z）；
- **|d| = 1.269 eV/Å ≫ 判据 0.0128555 → FAIL**（差 ≈ 0.0067 eV/2δ 窗）。

### 3.2 固定-μ 判决腿（μ 冻结 = −0.2193905904）——复现同 FD

| 腿 | FINAL_ETOT (eV) | Q | res |
|---|---|---|---|
| R−δ | −466.2137876462331 | 6.506868505 | +1.31e-3 |
| R+δ | −466.1998835520661 | 6.504064673 | −1.50e-3 |

- `F_FD = −2.629 eV/Å`（与重优化-μ 协议**逐位同值**）；
- `dQ/dz = −0.53 e/Å`（电荷响应真实存在）。
- 结论：FD 面稳健，差是**记账性**的（不是 μ(R) 路径/外环产物）。

> ⚠️ 数据卫生：固定-μ 第一批腿（oz_minus/oz_0/oz_plus）STRU 生成有 bug，
> 三个文件几何全同 R0（E 全 −466.20700x、Ewald 全 103.13292609）——早期
> "Ewald 不随几何变化"的怀疑来自这批坏数据，**已作废**。正确位移腿
> （oz_minus2/oz_plus2）见上表。

### 3.3 plain 对照（无约束，同网格同 δ）——PASS

- E(R0)=−466.3547593993459；F_ana(R0) O-z = −1.2103979595 eV/Å；
- E(R±δ)=−466.3577954901457 / −466.3513909042449 →
  `F_FD = −1.21027 eV/Å`，|d| = 0.00013 → **PASS**。
- 结论：LCAO 无约束力/能量完全自洽；不一致是**约束态特有**。

### 3.4 R0 力分解（fixed-μ、test_force 打印，O-z，eV/Å）

| 分量 | 值 |
|---|---|
| OVERLAP | −18.450874 |
| TVNL_DPHI | −43.943540 |
| VNL_DBETA | +0.880636 |
| VL_dPHI | +125.289819 |
| VL_dVL | +49.997455 |
| EWALD | −115.578927 |
| NLCC | 0 |
| SCC | −3.4e-7（修复后≈0）|
| CONSTRAINT | +2.805058（=0.1090989770 Ry/Bohr）|
| 分量和 | +0.9996 |
| COM 修正（−net/3） | −2.3594 |
| TOTAL（打印） | **−1.360240**（= 分量和 + 修正，自洽）|

- 打印力与其全部分解自洽 → 分析力侧不存在"隐藏项"；
- Ewald 分量跨腿正确变化（102.8272/103.1329/103.4388 eV）→ 排除 Ewald 缓存。

### 3.5 能量口径核查：E_KohnSham vs FINAL

- plain：两者逐位相等（1e-10）→ 无分歧。
- 约束 R0：末次 SCF 迭代打印 E_KohnSham = −466.1814363750，FINAL =
  −466.2070054139356，差 0.0256 eV 且**随几何变**（± 腿差 0.0212/0.0003）。
- 分析：该差是 μ 末步（−0.2→−0.2194）能量组件刷新滞后造成
  （Q 变 8.6e-3 ≈ 0.0256 eV/μ），打印 E_KohnSham 为滞后快照；
  **FINAL 是变分值**。FD(E_KohnSham 腿)≈−1.32 碰巧接近 F_ana 属巧合，
  不作为判据依据。

---

## 4. 分析

**已排除的假说（物证链）**

1. 观测口径（E′=E−μt* 的伪包络项）：PW 判决与 LCAO 归因均用 raw
   FINAL_ETOT；✓ 排除。
2. SCC-μw 污染（PW 根因）：修复后 SCC 分量 −3e-7；✓ 排除。
3. Ewald/晶格版本缓存：正确位移腿 Ewald 跨腿变化与 PW 物理一致；
   原怀疑基于坏 STRU 数据；✓ 排除。
4. 电子收敛噪声：腿内能量收敛到 ~1e-9 eV（>100 次 SCF 步稳定）；
   1.269 eV/Å = 0.0067 eV/2δ 窗是系统性记账差；✓ 排除。
5. 外环容差（|Q−t|=9e-5 停）：曲率代价 ~O(1e-8 Ry)；✓ 排除。

**剩余候选（未定案）**

- 差的方向：F_FD 比 F_ana 更负 1.27 eV/Å；只在约束态、只在
  ∂Q/∂R 强方向（O-z）出现；plain/PW 均不受影响。
- 结构线索：LCAO 的 μw 是**经 veff 网格注入**（PW 同款），但 LCAO 力侧
  多一条 PW 没有的路径——`fvl_dphi`（Pulay 局域力，读 `pelec->pot`，
  其中含 μw），与 `forcecon = −μ∫ρ·∂w/∂R` 的互补闭合在 LCAO 基下
  是否精确成立是首要候选（A）；次要候选（B）是 escon/forcecon 的
  Q 与 w 快照在 LCAO 注入生命周期里不同步。判别实验见 §5。

**结论状态**：LCAO 力 FD **FAIL、未豁免、未定案**；力维持"未验收"。

---

## 5. 下一步（待用户批准后再执行）

1. **轻量判别实验**（先把 §1.1 的 O-x + H1-x 四腿跑完，~30–40 min np4）：
   - O-x/H1-x 差≈0 → 电荷响应方向特有 → 查 μw 位置导数在
     fvl_dphi/forcecon 间的闭合（候选 A）；
   - H1-x 也差 ~1.27 → 均匀/约定项 → 查 COM 修正与 escon 记账（候选 B）。
2. **低网格对照**（ecutwfc=50，归因链 #3 egg-box）：残差同量级则判据被
   网格噪声淹没（预期不成立，但计划要求走一遍）。
3. 定点修复候选 A 或 B → 同一二进制重跑 O-z 冒烟腿（判据不豁免）。
4. 通过后接计划队列：LCAO 18 腿全量 → 残余∝|Q−t| 档位检查 →
   2.6.3 力矩 FD → 2.6.4 → 2.6.1（PW≡LCAO 逐位 + M3b 审计数据块）→ 2.7 判决门。
5. 闭合前 `make` 重建：`constraint_inject_lcao.{h,cpp}` 的 np=4 空目标
   audit 修复（11:51 修改）晚于当前二进制（11:04），尚未入 bin；
   该改动只影响 M3b 运行时审计路径，不影响力/能量。

---

## 6. 文件与数据

- 本 spec：`docs/superpowers/specs/2026-09-08-task26-lcao-force-fd-attribution.md`
- 前置 spec：`2026-09-07-task26-r7-judgments.md`、
  `2026-09-08-task26-r7-scc-fix.md`
- 数据：/tmp/cfd_lcao_4lt7、/tmp/cfd_lcao_fixedmu、/tmp/cfd_lcao_plain
- 脚本：`tests/constraint_fd_force/tools/run_constraint_fd.sh`（raw-E 口径）
- 半成品实验（被叫停，待续）：`/tmp/cfd_lcao_morelegs2/`、
  `/tmp/ab_mkleg.py`、`/tmp/cfd_lcao_more_legs2.sh`
