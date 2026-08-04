# Tier-1 判据体系筛选 —— 执行状态总结与中断恢复清单

> 2026-08-04 · 任务：测试三个新构建的 DeltaP 候选体系（hf / co / h2o_asym），
> 按 `tests/deltap_fd_force/README.md` 三项筛选协议选定判决体系。
> 本文件是**状态快照**（两次系统崩溃后的现场盘点），供评估后续 TODO，非正式结果文档。

---

## 0. 一句话现状

筛选 3（同步模式收敛）三个体系**全部 PASS**（串行完成，但 /tmp 数据被崩溃清空，仅仓库侧残留 MPI 运行）；
筛选 1（γ(λ) 平滑性）9 个冻结 λ 运行已构造并启动但**未跑完**；
筛选 2（分支稳定性）**未开始**。
同时发现一个**阻塞级 MPI 正确性 bug**（D_I Allreduce 交错混带，co/h2o_asym 4-rank 直接崩溃），
以及一个串行 FFTW-OMP 卡死（环境问题，`OMP_NUM_THREADS=1` 可绕过）。

---

## 1. 测试目标（README Tier-1 筛选协议）

| 筛选项 | 协议 | 状态 |
|--------|------|------|
| 1. γ(λ) 平滑性 | 3 冻结 λ 点（0, ±5e-3 Ry），逐原子 γ_report 单调、无 2π 跳变 | ❌ 未完成（运行被崩溃中断） |
| 2. 分支稳定性 | 逐原子 ±δ(z) 冻结 λ，γ/branch 连续、零分支翻转 | ❌ 未开始 |
| 3. 同步收敛 | base run（λ step 0.01，无内循环）scf_nmax=100 内收敛、无振荡 | ✅ 三个体系均 PASS（串行） |

---

## 2. 已完成的执行与结果（串行，OMP_NUM_THREADS=1，ecutwfc=100/ecutrho=400/scf_thr=1e-8）

> 以下数据来自崩溃前 /tmp/dp_screen 串行 base 运行的现场记录，**现已丢失**，需重跑补录。

| 体系 | SCF 迭代数 | 收敛判定 | E′(R0) / eV | λ* (Ry) | γ(base) | escon (Ry) |
|------|-----------|---------|-------------|---------|---------|-----------|
| hf | 37 | ρ 偏差 7.4e-9 < 1e-8 ✓ | −689.34097 | (−5.910e-3, −6.030e-3) | (−5.909, −6.031) | −0.07129 |
| co | — | 收敛 ✓（iter<100） | −614.96885 | (−7.306e-3, −8.614e-3) | (−7.291, −8.630) | −0.12761 |
| h2o_asym | — | 收敛 ✓（iter<100） | −483.83957 | (−5.477e-3, −3.869e-3, −3.381e-3) | (−5.479, −3.867, −3.382) | −0.05640 |

- 三个体系 λ 均正常收敛（λ* ≈ −γ(base)，与 h2o1 行为一致），无振荡迹象 → **筛选 3 PASS**。
- 运行耗时：hf 203 s / co 240 s / h2o_asym 290 s（串行单核）。

---

## 3. 新发现（阻塞级，须先修）

### 3.1 MPI 崩溃：co / h2o_asym 4-rank 运行崩溃 —— D_I Allreduce 计数 rank 相关

**现象**
- co：`iter_finish → compute_gamma_scf → compute_wannier_polarization` 的
  D_I Allreduce（deltap_wannier.cpp:449）抛 `MPI_ERR_TRUNCATE: message truncated`，gdir=1 首条串即崩。
- h2o_asym：iter 1 的 γ 计算完整跑完（rawG/P1 已打印），在 `[E-field]` 打印后崩溃，
  具体 Allreduce 未定位（疑似同族：line 1708 或 iter 2）。

**根因（证据链完整，已用 shim/gdb/插桩确认）**
- MPI 下 `psi` 由 `Setup_Psi::allocate_psi` 以 `ncol = para_orb.ncol_bands` 分配
  （setup_psi.cpp），`psi->get_nbands()` 返回的是**本地列数**，不是全局 NBANDS。
- 带维度按 nb=1 块循环分布：NBANDS=15（co）→ 本地 8（col-rank 0）/ 7（col-rank 1）；
  NBANDS=14（hf/h2o_asym）→ 7/7。
- `compute_D_I` 用该本地值 `resize(nbands)` → D_I 尺寸 rank 相关；
  D_I Allreduce（`MPI_IN_PLACE, 2*sz`）在 15 % 2 = 1 时 rank 间 count=16 vs 14 → TRUNCATE。
- 实测：shim 记录 `seq=36 count=16(rank 0,2)/14(rank 1,3)`；插桩打印 `rank=0 nbands=8`。
- **为什么 MPI smoke 没抓到**：deltap_bn_test（BN）的 NBANDS 恰好被列块均匀整除，
  count 一致 → 只暴露了"不崩溃"，掩盖了数值错误（见 3.2）。

### 3.2 MPI 数值错误（即使不崩溃）：D_I Allreduce 交错混带

- 带按 nb=1 交错分布（col-rank 0 持 0,2,4,…, col-rank 1 持 1,3,5,…），
  当前全通信子 Allreduce 把**不同带**的投影按同一本地索引相加 → 逐原子 γ 被污染。
- **定量对照（hf，仓库侧 MPI base 仍在）**：
  - MPI 4-rank：γ = (−5.107, −6.831)，Σγ = −11.938
  - 串行：γ = (−5.909, −6.031)，Σγ = −11.940
  - **Σγ（总极化）几乎守恒，但逐原子分量漂移 ~0.8 rad** —— 总和的巧合掩盖了错误，
    之前"MPI 结果合理"的判断不可信。
- 附带症状：hf MPI base 跑到 iter=100（scf_nmax 未收敛）仍出 FINAL_ETOT，
  串行 37 iter 收敛 —— 混带 γ 破坏了 λ 更新路径。

### 3.3 串行 FFTW-OMP 卡死（环境，非 DeltaP bug）

- 串行（未设 OMP_NUM_THREADS）时，co/h2o_asym 在 `atomic_rho → recip2real`
  FFTW OMP 线程池处全线程屏障自旋（gdb 确认：主线程+7 worker 全在 gomp barrier）。
- `OMP_NUM_THREADS=1` 立即正常（co 1.18 s 过 CHARGE）→ 筛选与 FD 一律显式设 1。
- 建议：run_fd.sh 增加 `export OMP_NUM_THREADS=1` 或文档注明。

---

## 4. 数据与工作区清单（崩溃后）

| 路径 | 状态 | 内容 |
|------|------|------|
| `tests/deltap_fd_force/{hf,co,h2o_asym}/` | ✅ 仓库侧幸存 | STRU/KPT/target.dat（骨架，未入库，untracked） |
| `tests/deltap_fd_force/hf/base/` | ✅ 幸存 | **MPI 4-rank 完整运行**（E′ −689.5229，γ band-mixed，iter=100），可作 3.2 对照复现 |
| `tests/deltap_fd_force/co/base/`、`h2o_asym/base/` | ✅ 幸存 | MPI 崩溃日志（TRUNCATE / non-zero exit） |
| `tests/deltap_fd_force/README.md` | ✅ 仓库侧 | Tier-1 节已写入（本次未改） |
| `/tmp/dp_screen/`（串行 base 三体系 + 9 冻结 λ 扫描 + shim/gdb 分析产物） | ❌ **两次崩溃后丢失** | 需重跑；分析结论已在本文件固化 |
| `build/abacus_basic_para` | ✅ 幸存 | 与 HEAD 源码一致（临时插桩已回退，无残留修改） |
| 源码 | ✅ 干净 | `git status` 无 source 改动（仅 docs/tests untracked/modified） |

---

## 5. 下一步 TODO（供评估，按依赖排序）

### P0 — 修 D_I MPI 路径（先于一切筛选，否则多 rank 不可用且 1-rank 才是唯一可信面）
> **已修复并验证（2026-08-04 Stage 0.1/0.2）**：方案 A'（D_I 全局带槽位 + 统一
> 计数 Allreduce + nlm 全局键 + 5 处全局 nbands）已提交；co 奇数 NBANDS=15 4-rank
> 不再崩溃；hf/co λ=0 逐原子 γ 与串行一致。另发现并修复**相位配对 sort 负距离 bug**
> （见 dev log 2026-08-04 轮）——co 串行 γ 修正为 (-6.702,-9.219)。
- [ ] 方案裁定：A) 带维度列复制（MPI_Allgatherv）后按**行通信子**归约；
      B) 全量复制 C 矩阵本地算 D_I（内存 nlocal×nbands/rank，小体系无压力）。
- [ ] `nbands` 改用全局值（`PARAM.globalv.nbands` 或 `pelec->nbands`），不再用 `psi->get_nbands()`。
- [ ] 修复后回归：hf 4-rank γ 应与串行逐原子一致（±0.01 rad）；
      co/h2o_asym 4-rank 不再崩溃。
- [ ] MPI smoke 扩用例：补一个 NBANDS 不能被列块整除的体系（如 co），
      否则该 bug 会被"恰好整除"掩盖。

### P1 — 重跑三项筛选（工作区迁到仓库侧 gitignored 目录，避免 /tmp 再次被清）
- [ ] 串行 base（筛选 3）重跑补录数据（hf/co/h2o_asym，OMP_NUM_THREADS=1）。
- [ ] 筛选 1：9 个冻结 λ 运行（0, ±5e-3 Ry，init_chg=base）。
- [ ] 筛选 2：逐原子 ±δ(z) 冻结 λ（hf 2×2、co 2×2、h2o_asym 3×2 = 14 次）。
- [ ] 汇总三项 PASS 者 → 选定判决体系（预期 hf）。

### P2 — 固化 3.2 的 MPI 对照为回归锚点
- [ ] hf MPI-vs-串行 γ 对照表写入正式文档（Σγ 守恒 + 逐原子漂移），
      修复后该对照必须闭合。

### P3 — 文档与环境
- [ ] run_fd.sh / README 注明 `OMP_NUM_THREADS=1`（串行 FFTW 卡死规避）。
- [ ] 本文件内容正式化：筛选结果落 `2026-08-04-deltap-tier1-screening.md`，
      MPI bug 落 dated 文档（含证据链），dev-log 追加。

---

## 6. 给评审的关键问题

1. D_I 修复方案选 A（列复制+行通信子归约，内存小、改动中等）还是 B（全复制，改动最小、内存大）？
2. 是否同意"MPI smoke 需补非均匀整除用例"作为 D5 扩展？
3. 筛选工作区从 /tmp 迁到仓库侧 gitignored 目录（`tests/deltap_fd_force/<sys>/base` 已是）——确认。
