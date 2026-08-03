# DeltaP Force/Stress T7-a：修 B-1 nlm 越界 + B-5 跨离子步 UAF + B-3 λ 轨迹（实证）

> 2026-08-02 · 依据 `2026-08-02-deltap-force-stress-dev-guide.md` §3-B1/§9-T7-a 执行。
> 本轮是 force/stress 开发的第一轮**实证修复**：B-1（力路径 nlm 布局越界，必崩）、
> 新发现 B-5（backend 捕获 dp_op 指针跨离子步悬垂 → UAF）、B-3（λ 跨离子步轨迹，
> 顺带关闭）。O3（多 ζ 丢轨道）经源码实证后裁定为**设计语义，非 bug**（见 §5）。

---

## 1. Test plan

1. 定位 `deltap_force_stress.hpp` B-1 精确根因（nlm 提取/消费两端索引协议）；
2. 修复后 `tests/deltap_relax`（relax_nmax=3）跑通 ≥3 离子步，exit=0；
3. ASAN 构建复跑同一算例 + cell-relax（isstress 路径）——零内存错误；
4. 核对 B-3：λ 跨离子步无重复累加/无丢失；
5. 回归：单测 16/16、`tests/deltap_mpi_smoke/run.sh` 三用例、test_C_I SCF 能量锚点。

## 2. Test setup

- 系统：本机（Intel Ultra 5 225H，单机）；`mpirun -np 1`；`-O3 -DNDEBUG -std=gnu++14`
  常规构建 + `ENABLE_ASAN=ON` 独立构建（`build_asan`）。
- 主算例 `tests/deltap_relax`：H2O（O 2s2p1d / H 2s1p，单 ζ/角动量），
  calculation=relax，relax_nmax=3，deltap_gdir=3，deltap_inner_nmax=0（同步模式），
  deltap_lambda_init=0.0，deltap_lambda_step=0.01，mixing=0.1，inner_thr=1e-3，
  genelpa，8 k 点（2×2×2 超胞 15.87 Å）。
- 应力冒烟：同一算例改 calculation=cell-relax，relax_nmax=1（触发 isstress 路径）。
- 回归：`MODULE_ESOLVER_deltap_common_test`（10）+ `MODULE_ESOLVER_esolver_dp_test`（6）；
  `tests/deltap_mpi_smoke/run.sh`；`tests/deltap_bn_sampling/test_C_I`（SCF 锚点，
  commit 6beb70bc3 基线 `GE16 -3.38713360e+02 eV`）。

## 3. Results

| 项 | 结果 |
|----|------|
| 修复前（HEAD bfe0aa45d） | `double free or corruption (!prev)`，GE43（step1 SCF 末段/力阶段） |
| 修复后 relax 3 步 | exit=0，`RELAX STEP: 1/2/3` 全部出现，TOTAL Time 175s |
| ASAN relax 3 步 | exit=0，`ERROR: AddressSanitizer` 计数 **0** |
| ASAN cell-relax 1 步（isstress） | exit=0，ASAN 0 错误；输出 TOTAL-STRESS 3×3 张量 |
| 单测 | 10+6 = **16/16 PASSED** |
| MPI 冒烟 | PW 2-rank / BN 4-rank / 内循环 4-rank 三用例 **PASS** |
| test_C_I 锚点 | `GE16 -3.38713360e+02` **逐字节一致**（无 SCF 回归） |

### λ 跨离子步轨迹（B-3，deltap_relax run.log）

```
STEP 1: [DeltaP P2] iter=12 → λ=(-5.51e-3, -3.60e-3, -3.60e-3) 冻结至 GE43
STEP 2: [DeltaP P2] iter=6  → λ=(-1.10e-2, -7.19e-3, -7.19e-3)  （= step1 值 + 一次 GD 步）
STEP 3: [DeltaP P2] iter=7  → λ=(-1.65e-2, -1.08e-2, -1.08e-2)  （= step2 值 + 一次 GD 步）
```

各步首迭代即承接上一步末值（无归零、无重复累加）；每离子步 P2 阶段再执行一次
梯度下降（γ≈−5.5 恒定、t=0 ⇒ 残差恒定 ⇒ λ 线性累积），符合"两阶段 + 跨步延续"
设计意图。

## 4. Analysis（根因链，三条独立缺陷）

### B-1（阻塞，85b2af322 引入）：力路径 nlm 索引协议破坏

- `snap()`（`two_center_integrator.cpp:148-230`）的 ket 侧按 **(L→N→m) 展平**，
  平铺索引 = 原子全局 `iw`（两者同构：L 外、N 中、m 内，m 为 ABACUS 约定序
  0,1,−1,2,−2…；`atom_spec.cpp:set_index` 与 `nchi_ket==l_nchi` 保证相等）。
- 旧提取循环按 `iw` 遍历全部轨道、`index` 持续自增 + 条件重置 `L²`：每个 L 的
  首轨道（N=0,m=0）之外的所有 iw 都写入**块外**位置；最后一个 L 的 m>0 轨道
  直接越界（channel=3 时 `index+3·length ≥ 4·length`）。单 ζ 也崩（nwl≥1 即触发），
  多 ζ 崩得更远。OMP 区内表现为 double-free/heap corruption（README 背靠栈吻合）。
- 消费端 `cal_force_IJR/cal_stress_IJR` 用 `l*l+m` + `length=nlm_size/4` 读取——
  与 dspin 模板一致；故修复只需**把提取端对齐消费端协议**。

### O3（多 ζ 丢轨道）裁定：设计语义，非 bug

- γ 的投影基由 `deltap_overlap.cpp` 定义：`nproj_per_atom_=(nwl+1)²`，
  "Select first zeta of each l, same as DeltaSpin"（:91），SMO 矩阵、S_k/D_I、
  gauge 全部沿用 → **每个 (l,m) 恰一个径向函数（N=0）**。
- SCF 路径 `cal_pre_HR` 与 dspin 同款 first-ζ 提取，**与 γ 自洽**；多 ζ 的 N>0
  轨道不进入约束投影基是刻意选择（force 必须对同一 H_HR 求导）。
- 结论：**不改语义**，B-1 修复保持 first-ζ 通道集；O3 从 TODO 移除，
  记为 LIMITATION（若未来需全 ζ 投影基，需同步改 γ 全链：nproj_per_atom、
  SMO 维数、D_I、escon）。

### B-5（新发现，跨离子步 UAF）：backend 捕获 dp_op 悬垂

- `before_scf`（esolver_ks_lcao.cpp:161-170）在**每个离子步** `delete p_hamilt`
  并重建（含新 `DeltaPOperator`）；而 `deltap_make_backend` 的
  `set_lambda/get_lambda/sync_lambda/apply_hk_correction` 在 `deltap_init`
  （仅首次）时**捕获了 step1 的 dp_op 裸指针**。
- step2 iter1 `iter_finish→update_lambda_gd→get_lambda()` 命中已释放对象 ⇒
  `lambda_` 读出垃圾 ⇒ `std::vector<double>` 构造抛 `bad_alloc/bad_array_new_length`
  （两次运行异常类型不同 = 堆已被污染/悬垂读，gdb bt 指向
  `deltap_make_backend::{lambda()}` 直接抛出）。
- 修复：backend 各 lambda 改为每次调用经 `get_dp_op`（查当前 `p_hamilt`）解析；
  `before_scf` 重建后把 `state_.lambda_eff`（新持久化点，`apply_lambda` 写入）
  播种进新算子，`contributeHR` 以 dλ=λ−0 全量补加重建 hR（与既有注释设计一致）。

### B-3（λ 跨离子步）关闭

- 上述播种 + `state_.lambda_eff` 持久化使 λ 跨步连续（§3 轨迹表）；
  无重复累加（同一算子内 lambda_save_ 机制不变）、无丢失（新算子继承末值）。

### 顺带修复（消费端审计发现）

- `cal_stress_IJR` 原按 3×3（9 元）写 `stress[ipol*3+0..2]`，但 buffer 只有 6 元
  （对称存储）→ **越界写**。改为 6 元对称累积 [xx,xy,xz,yy,yz,zz]（isstress 路径
  的 ASAN 冒烟验证）。
- 两个消费端补 `dm_pointer += (npol-1)*col_size` 行尾步进（dspin 模板同款，
  修复 npol=2 自旋极化时 DM 错位读取的潜在 bug）。

## 5. Files modified

- `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`：B-1 提取
  改 first-ζ(l,m) 显式 l²+m 写入；消费端布局注释 + 长度守卫 + npol 行尾步进 +
  stress 6 元对称累积。
- `source/source_esolver/esolver_ks_lcao.cpp`：backend 动态解析 dp_op（B-5）；
  `before_scf` 播种 λ（B-3）。
- `source/source_esolver/deltap_scf.cpp`：`apply_lambda` 持久化 `state_.lambda_eff`。

## 6. Next steps（T7-b 起）

1. **T7-b**：FD 双组协议（`tests/deltap_fd_force/run_fd.sh` 补 escon 提取；
   组① frozen λ / 组② 每位移点 λ 重收敛；判据 5e-4 Ry/Bohr）——A1 符号与
   A2 量级的首次实证。
2. **T7-c**：实现 A2（∂τ/∂R HF 项，−λ_J⟨P̂_J⟩(L⁻¹)_{αβ}）+ λ 诊断打印
   （对标 dspin `spin_constrain.cpp:857`）；应力 FD（变胞 ±0.1%）验 S1。
3. **T7-d**：P17 relax 对照（deltap vs efield）端到端。
4. 记录 LIMITATION：O3（first-ζ 投影基）、B-1 修复不改变 γ 语义。
