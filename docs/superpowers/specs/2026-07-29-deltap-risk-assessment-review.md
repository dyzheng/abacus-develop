# 2026-07-29 DeltaP 算法理论与实现风险评估报告

> 分支：`feat/deltap`（merge-base vs develop: `b9669d37`，HEAD `93a0d782`）
> 方法：**静态代码审查 + 设计文档研读 + git 历史分析**。本轮未做构建/运行动态验证（见文末"下一步"）。
> 范围：DeltaP 专用代码（LCAO `module_deltap`、`deltap_lcao` 算符、esolver 集成；PW `deltap_pw`、`op_pw_proj`；公共 `bfgs.h`、`deltap_common.h`）及被修改的共享文件。
> 既往评审：2026-07-12 `risk-points-and-solutions`（18 项）、2026-07-13 `risk-review-evaluation`/`rebuttal`。本报告与其关系见 §8——**本轮新发现 20 项代码级问题，多数在既往评审覆盖范围之外**（PW 实现、约束矩阵、力/应力、MPI 文件 I/O 均为 07-13 之后新增或未被审查的部分）。

---

## 1. 总体结论（TL;DR）

DeltaP 当前处于**"特定配置下演示成功、泛化使用高风险"**的状态：

- 已演示的路径非常窄：LCAO + nspin=1 + 正交晶胞 + 1×1×N k 网格 + gdir=3 + 小体系（nocc≤4）+ 固定构型 SCF + 串行/特定 MPI 网格。
- **8 项 Critical 级确认 bug**：其中力/应力与哈密顿量不自洽（C-02）、MPI 能量跨 rank 不一致（C-01）、gdir≠3 时约束算符方向错配（C-05）、MPI 非方形进程网格越界写（C-06）直接影响结果正确性或程序稳定性。
- **relax/MD 场景当前必错**：λ 跨离子步不更新（C-12）、hR 重建后约束项丢失（C-13）、力本身就不自洽（C-02）三重叠加。
- **PW 路径为半成品**：初始 λ 单位混淆（C-03）、`deltap_corr=0` 时仍施加微扰（C-04）、内循环假循环过冲（C-08）。
- 理论层面 3 项本质限制（分解不唯一、分支切割破坏跨构型连续性、约束模式测不到物理极化率）已在既往文档中定论，**任何对外宣称的物理结果都必须带这些限定条件**。

---

## 2. 架构与数据流（审查基线）

**LCAO 约束路径（deltap_switch=1 && deltap_corr=1）**：
`iter_finish` → `deltap_init`（首次，建 SMO 积分器 + unkOverlap_lcao）→ `deltap_compute_gamma`（`compute_gamma_scf` → `compute_wannier_polarization`：3 方向 × 多 string Wilson loop + Hungarian 匹配 + 分支选择）→ `deltap_update_lambda`（两阶段：drho 低于阈值时一次性梯度步）→ `compute_hk_correction`（Berry 联络算符）→ `DeltaPOperator::contributeHR/contributeHk`（进哈密顿）→ `FORCE_STRESS`（力/应力修正）。

**LCAO 分解路径（corr=0）**：`ctrl_scf_lcao`（仅 `calculation=nscf`）→ `compute_atomic_polarization` → 按 `deltap_method` 分派 berry_connection（默认）或 wannier。

**PW 路径**：`esolver_ks_pw` 初始化读 STRU `dp_target` → `OnsiteProj::act`（`cal_ps_deltap`：λ·|α⟩⟨α| on-site 投影，复用 ΔSpin kernel）→ `iter_finish` → `deltap_iter_finish`（drho 门控，一次性 λ 更新 + escon）。

---

## 3. Critical — 确认 bug（阻塞生产使用）

### C-01 [Critical][MPI] `dp_escon` 能量修正只在 rank 0 设置，etot 跨 rank 不一致
- 证据：`esolver_ks_lcao.cpp:793`（`this->pelec->f_en.dp_escon = deltap_common::compute_dp_escon(...)`）位于 `if (GlobalV::MY_RANK == 0)` 块内（块范围 725–808 行）；`fp_energy.cpp:19` 中 `etot = ... + dp_escon`。
- 影响：MPI 运行时 rank 0 的总能比其它 rank 低 Σλ·γ，破坏了"etot 全局一致"这一 ABACUS 基本不变量。**已核实不会因此死锁**：SCF 收敛标志由 rank 0 广播统一（`chgmixing.cpp:100` `MPI_Bcast(&conv_esolver,...)`）。真实后果：(a) 非 rank-0 的 `pelec->f_en.etot` 缺修正项，任何按 rank 读本地 etot 的消费者（restart、弛豫/MD 驱动器的能量比较、`cal_energy()`）拿到不一致的值；(b) `scf_ene_thr>0` 时能量收敛判据用 rank 0 的 etot——除 MPI 不一致外，**串行同样受影响的衍生问题**：λ 一次性更新（P2 切换）时 dp_escon 突变，`etot_delta` 在该步出现跳变，可能错误推迟能量收敛。PW 侧（`esolver_ks_pw.cpp:305-306`）在所有 rank 设置，说明 LCAO 侧是疏忽而非设计。
- 修复：将赋值移出 rank 守卫（打印保留在守卫内）。

### C-02 [Critical][物理] 约束力/应力与哈密顿量不自洽
三个互相独立的不自洽叠加：
1. **τ_α 因子缺失**：`deltap_lcao.cpp:80-81` HR 算符为 `coeff = dλ·τ_α(iat)`（τ_α 为原子分数坐标），即 H 中含 λ·τ_α·P_I；而 `deltap_force_stress.hpp:255-260` 的力公式为 `λ·(∂nlm·nlm)·DM`，**既无 τ_α 因子，也无 ∂τ_α/∂R_J = δ_IJ 的齐次项**。
2. **×2 因子无出处**：`deltap_force_stress.hpp:180` `force = force * 2.0`（注释 "Hermitian conjugate contribution"），但其模板来源 DeltaSpin（`dspin_force_stress.hpp:298-306`）**没有**该因子。两份代码的算符结构相同（|α⟩⟨α| 型），因子差异无文档依据。
3. **HK（Berry 联络）部分完全没有力/应力贡献**：`cal_force_stress` 只覆盖 HR 投影算符的 Pulay 项；而物理算符的能量导数 −Σλ·∂γ/∂R 中 γ 由 Wilson loop（HK 部分）承载。
4. 应力公式自述 "simplified"（`deltap_force_stress.hpp:301`），且 `stress[ipol*3+k] += F·r_vector[k]` 用的是**整数晶格矢量**（无量纲），未做笛卡尔变换与 lat0 换算，量纲错误。
- 影响：deltap_corr=1 时 `cal_force/cal_stress` 输出错误；能量-力不自洽 ⇒ relax/MD 轨迹错误且无法用 FD 通过。**当前所有固定构型的测量（PES、刚度）不受此影响，但任何结构弛豫结果无效。**
- 修复：先完成算符理论闭环（HK 部分的 ∂/∂R），再用有限差分对总能做力验证。

### C-03 [Critical][PW] 初始 λ 被赋值为目标 γ（rad 当 Ry 用）
- 证据：`deltap_pw.cpp:33` `s_targets = lambda;`（同一 vector 既当 λ 又当 target）；`esolver_ks_pw.cpp:107-114`：无 STRU target 时用 `deltap_lambda_init` 填充，有 STRU target 时 `set_deltap_pw_lambda(dp_target, ...)`——**λ 初值 = 目标 γ 值**。
- 影响：PW + STRU `dp_target` 非零（如 0.5 rad）时，Phase-A SCF 从第一步就施加 0.5 Ry 的 on-site 势，严重污染电荷密度；后续 escon 全部错误。LCAO 侧由构造函数 lambda_=lambda_save_ 规避了此问题（`deltap_lcao.cpp:30-31`），PW 侧没有。
- 修复：λ 与 target 分开存储，λ 初值只用 `deltap_lambda_init`。

### C-04 [Critical][PW] `has_deltap` 传错开关：deltap_corr=0 也施加约束微扰
- 证据：`hamilt_pw.cpp:124-127`：`OnsiteProj(..., PARAM.inp.sc_mag_switch, (PARAM.inp.dft_plus_u>0), PARAM.inp.deltap_switch)`——第三个功能参数应为 `deltap_switch && deltap_corr`。
- 影响：PW 纯分解模式（corr=0）下 `cal_ps_deltap` 仍把 λ（此时 = target 值，见 C-03）加进哈密顿。**PW 的"只分解不约束"模式不存在，任何 deltap_switch=1 的 PW 计算都被静默微扰。**
- 修复：改为 `PARAM.inp.deltap_switch && PARAM.inp.deltap_corr`。

### C-05 [Critical][LCAO] gdir≠3 时 HK 修正算符的 k-string 方向与 S_dk 方向错配
- 证据：`compute_wannier_polarization` 的 alpha 循环（`deltap_wannier.cpp:340-344`）每次以 gdir=3 结束，故 `k_index_` 保留的是 **z 方向** string（`gdir_` 在 1526 行恢复为输入值）；`compute_hk_correction`（1627 行）用 `k_index_[0]` 的 link 对，但 `compute_S_dk`（1604-1607 行触发）按**输入 gdir** 构造位移。
- 影响：`deltap_gdir=1` 或 `2` 时，约束算符 = z 方向 link 的波函数 × x/y 方向位移的 S_dk——**算符物理意义错误**，约束仍可能"看起来收敛"（γ 被分支选择拉到 target 附近，见 T-12）。默认 gdir=3 恰好自洽，是该 bug 未在测试中暴露的原因。
- 修复：`compute_hk_correction` 前按输入 gdir 重跑 `setup_kstring` 并重建对应 string 的 D_I（或缓存每方向数据）。

### C-06 [Critical][MPI] hk_correction 假设 nrow==ncol，非方形 2D 进程网格越界写
- 证据：`deltap_wannier.cpp:1687-1711` 构造 `M/H_sym` 为 `nrow×nrow`；`deltap_lcao.cpp:280-291` `add_hk_correction` 按 `H_sym.size()`（= nrow²）写入 `hsk->get_hk()`，而 hk 本地块为 **nrow×ncol**。2D 块循环分布下 nrow≠ncol 很常见（如 nlocal 不能被进程网格均分）。`deltap.h:92` 注释自承 "Serial only for now (nrow == ncol)"，但 `compute_hk_correction` 在 SCF 路径中被无条件调用（`esolver_ks_lcao.cpp:1412`）。
- 影响：ncol<nrow 时**堆越界写**（内存损坏/崩溃——这正是 git 历史中 "MPI: fix np>1 crash" 系列提交（`65c9c798`、`6ddad714`、`baea4a8d`）未能根治的问题）；ncol>nrow 时部分 hk 未被修正。串行与方形网格恰好安全。
- 修复：按 nrow×ncol 构造，或显式 `WARNING_QUIT` 拒绝 nrow≠ncol。

### C-08 [Critical][PW] "内循环"不重解波函数，λ 线性过冲 inner_nmax 倍
- 证据：`deltap_pw.cpp:191-208`：循环内 `compute_per_atom_gamma_kstring` 用**同一个 psi**（无 re-diagonalize），γ_trial 每次相同 ⇒ 残差不变 ⇒ `λ += step·res` 每次累加同一增量。循环实际是 `λ_final = λ_0 + inner_nmax·step·res`（mixing=1 时）。
- 影响：PW 下设 `deltap_inner_nmax>0` 的用户得到 inner_nmax 倍过冲的 λ，约束体系被严重过推。代码注释自承 "Phase D.1: simple gradient descent (no subspace diag)"——半成品误入可用路径。
- 修复：无重解实现前，内循环分支应 `WARNING_QUIT` 或直接删除。

### C-11 [Critical][数值] 分支平移晶格三种口径互相不一致，可选中物理不可达的 γ
- 证据：γ_I 的实际定义用 **per-band 归一**权重 `w_norm = w_In/w_tot(n)`（`deltap_wannier.cpp:1034-1037`），故能带 n 相位移动 2π 时 γ_I 的真实变化为 `2π·w_In(n,I)/w_tot(n)`。但三处分支搜索用了三种不同的平移量：
  - per-string Step-3：`2π·w_In`（未归一，1138-1142 行）——与 dev log 第 5 条结论一致；
  - 全局 target 搜索（约束矩阵）：`2π·w_In/w_total(I)`（per-atom 归一，1289 行）；
  - 全局搜索（total/per_atom 模式）：同上 per-atom 归一（1396、1457 行）。
  per-atom 归一分母 `w_total(I)=Σ_n w_In(n,I)`（BN 孤对可达 1.6）使搜索晶格系统性缩小 ~1.6 倍。
- 影响：搜索枚举的候选 γ 与 Wilson loop 实际可达的 2π 分支**不对格**——可能把 γ 强制设到物理上不可达的值（`gamma_accum[iat] += delta·n_strings` 直接写入），也可能错过正确分支。这是 07-19 λ sweep 中 "g 在 −5 与 +1 间跳变" 类异常的候选根因之一。分支选择本就是该方法最脆弱的环节（见 T-02），口径不一致使其雪上加霜。
- 修复：统一为 per-band 口径 `2π·w_In/w_tot(n)`（zeta rescale 后还需乘 scale），并加单元测试：人为构造已知分支跳变，验证搜索命中真实分支。

---

## 4. Important — 确认 bug（显著影响特定场景）

| ID | 位置 | 问题 | 影响 |
|---|---|---|---|
| C-07 | `deltap_wannier.cpp:2130-2134` | `compute_resta_z` MPI 分支注释自承 "This is wrong for 2D block-cyclic"（c_nu 用列局部索引按行主序寻址） | MPI 下 Resta-Z 电子中心结果错误（该功能本身也是 Γ 点 Mulliken 近似，2035 行 `if (ik != 0) continue;`） |
| C-09 | `deltap_pw.cpp:158-159` | `gamma_total == 0.0` 时整体跳过 λ 更新 | 中心对称体系（金刚石、未畸变 BTO）γ_total≡0 → **PW 约束永远不生效**；且"计算失败"与"真零"不可区分 |
| C-10 | `deltap_wannier.cpp:1053, 1179, 1539-1540, 1745-1808` | `save_branch/save_match` 及两个 debug 文件（`deltap_branch_enum.dat`、`deltap_zeta_debug.dat`）**无 rank 守卫**，MPI 下所有 rank 同时写同一文件 | 文件竞争损坏 → 下一 run `load_branch/load_match` 读到损坏值 → 跨 run 分支状态污染（T-08 的确定性机制因此被削弱）；zeta_debug 每 SCF 步 append 无界增长 |
| C-12 | `esolver_ks_lcao.cpp:1277, 1234` | `deltap_lambda_set_`/`deltap_inner_loop_done_` 为 esolver 成员，跨离子步不重置 | **relax/MD 中 λ 只在首个离子步更新一次，之后结构变了约束不再跟踪**——弛豫轨迹的约束实际失效 |
| C-13 | `deltap_lcao.cpp:55-58` | `hr_done=false`（hR 重建）时不补加已有 λ 的贡献；对照 DeltaSpin 的处理（`dspin_lcao.cpp:93-97`："HR rebuilt → reset lambda_save, add full lambda"） | 结构更新/restart 后 HR 中的约束项静默丢失（HK 部分仍在）→ 哈密顿不一致 |
| C-14 | `deltap_pw.cpp:429` | `VR[n * m_dim + m]`：zgeev 返回**列主序** VR，正确索引为 `VR[m + n*m_dim]` | PW per-atom 权重用了转置的本征向量矩阵——per-atom γ 分配错误（总 γ 不受影响） |
| C-15 | `deltap_berry.cpp:211-225` vs `deltap_wannier.cpp:388-389`；默认方法 `input_parameter.h:623` = `berry_connection` | berry_connection 路径无 spin 因子（wannier 路径有 `spin_factor=2`），且两路径符号约定相反 | **默认分解方法与 wannier 方法的 P_I 差 2 倍**；nscf 分解默认走的是更少验证的路径 |
| C-16 | `deltap_berry.cpp:130-133, 173-189, 230-232` | 积分对 closure 点（j=0 与 j=nppstr-1 为同一 k）全权重双计；边界处中心差分退化为半权重前/后差分 | berry_connection 路径系统性 O(1/nppstr) 积分误差 |
| C-17 | `deltap_berry.cpp:141-193` | `term2`（D_I 差分）在 alpha 循环内但恒沿 string（gdir）方向 | gdir 以外两个分量的 A_nk 混入了错误方向的导数——berry_connection 路径仅 gdir 分量有意义，其余分量静默错误 |
| C-18 | `esolver_ks_lcao.cpp:1322-1346` | `else` 分支内再判 `use_constraint_matrix`（恒假）的死代码块，内含**无 rank 守卫**的 cout | 死代码 + 误导维护者 |
| C-19 | `esolver_ks_lcao.cpp:1287`、`deltap_pw.cpp:169`、`read_input_item_other.cpp:1263` | `mixing==0.0 → 1.0` 静默改写，用户无法选择"无 mixing"；且参数 availability 标注为 "inner_nmax>0"，实际两阶段模式（inner_nmax=0）也在用 | 参数语义与文档矛盾 |
| C-20 | `deltap_wannier.cpp:473` vs `deltap.cpp:149-164` | 边界 link 的 G 相位 `G_cart_bdy = dk_string·kv_->nmp[gdir-1]` 直接用 `nmp`，而 `setup_kstring` 自己实现了 nmp=[0,0,0] 时的回退推断（symmetry=-1 全网格场景）——推断结果没有回写 nmp | nmp=0 时 G_cart_bdy=0 → 边界 link 相位错误 → 总 Berry 相位错（berry_phase 标准用法 symmetry=-1 恰是该场景） |

---

## 5. Suspected — 疑似 bug / 设计脆弱点（需测试裁决）

| ID | 位置 | 问题 | 触发条件与影响 |
|---|---|---|---|
| S-01 | `deltap_wannier.cpp:1640-1659` | `compute_hk_correction` 的 `w_eff` 用 `kstring_data_` 中残留数据——此时为最后一个 alpha（z 向）最后一条 string 的 D_I，与 `k_index_[0]`（string 0）的 k 点不对应 | 多 string 网格（如 2×2×2 有 4 条 string）时 w_eff 取错 k 点的权重；1×1×N 网格恰好自洽 |
| S-02 | `deltap_wannier.cpp:1102-1104` | zeta rescale 分母保护仅 `1e-15`：`scale = γ_unw_sum/γ_raw_sum` 在 γ_raw_sum 近零（中心对称/抵消体系）时爆炸 | 金刚石类体系 per-atom γ 灾难性放大 |
| S-03 | `esolver_ks_lcao.cpp:1225-1234` | 内循环模式结束后**没有 mix_reset**（两阶段模式在 1400 行有）——内循环末态 DM 由 trial-λ 波函数生成，Broyden 历史跨越约束切换 | 内循环模式（inner_nmax>0）下电荷 sloshing，与 07-20 文档记录的现象同源 |
| S-04 | `esolver_ks_lcao.cpp:1198-1199`、`1124` | `accept_trial` 的 α_opt 修正被丢弃（`lambda_inner = lam_trial`），但 bfgs 内部 dnu_ 已含修正 → λ 轨迹与残差历史错位；`bfgs.init` 每轮重置 α_trial=0.5，跨外层迭代无自适应积累 | 内循环模式 CG 效率/正确性下降（该模式当前默认关闭，影响面小） |
| S-05 | `bfgs.h:204-206, 255-257` | FR-CG 步长 α_trial 恒正（自适应只调幅度不调符号）：若某体系/方向 dγ/dλ<0，`λ += α·(γ−t)` 为正反馈发散 | 算符符号约定翻转的体系无保护 |
| S-06 | 多处（`deltap_wannier.cpp:315-318`、`deltap_berry.cpp:211-214`、`deltap.cpp` 无守卫） | nspin=2（磁性，每自旋 nocc 不同、k 点带自旋块）与 nspin=4（npol=2 spinor，D_I/S_dk 索引不含自旋结构）均无输入守卫 | nspin=2/4 + deltap 静默给出错误结果 |
| S-07 | `deltap_wannier.cpp:384-386`、`deltap_berry.cpp:221-225`、`deltap_pw` 同 | P=γ·|a_α|/(2πΩ) 分量式换算假设**正交晶胞**；非正交时 γ_α 沿倒易方向，需全晶格矩阵变换 | 六方/三方/单斜体系（如菱方 BTO）极化矢量方向错误 |
| S-08 | `deltap.cpp:189-223` | k-string 索引硬编码 `ix + iy·mpx + iz·mpx·mpy` 的 k 点排布假设；1×1×1 网格（nppstr=2、dk=1）无守卫；对称约化 k 点无守卫 | k 点重排/对称开启/Γ 点计算时静默错误 |
| S-09 | `deltap_pw.cpp` 全局 | PW 侧 γ、λ 无任何 MPI 同步（对照 LCAO 的 Bcast：deltap_wannier.cpp:1557-1585、esolver_ks_lcao.cpp:1383-1391）；`deltap_iter_finish` 打印也无 rank 守卫 | PW 多 rank 时 λ/escon 可能不一致 + 重复输出 |
| S-10 | 三处 nocc 计算（wannier:315、berry:211、pw:152） | `nocc = ceil(nelec/DEGSPIN)` 未排除金属/分数占据/展宽；Wilson loop 对金属本无定义 | 金属体系静默给出无意义 γ |
| S-11 | `esolver_ks_lcao.cpp:247` vs `FORCE_STRESS.cpp:438` | `s_stored_lambda` 是**模板实例静态量**：cal_force 存 `<TK,TR>` 实例，FORCE_STRESS 读 `<complex,double>` 实例 | nspin=4（TR=complex）时力修正静默为空 |
| S-12 | `deltap_pw.cpp:369, 428-430` | PW per-atom 权重只用 `get_becp()`（最后一次 onsite 投影的**单个 k 点**）配所有 string 的本征向量 | 权重与本征向量 k 点不对应，per-atom 分配有系统误差 |
| S-13 | `esolver_ks_lcao.cpp:949-972` | `deltap_target_file` 打不开时静默落到"无 target"路径 | 拼错文件名 ⇒ 无约束运行且看起来正常 |
| S-14 | `deltap_wannier.cpp:1299-1365` | 约束矩阵分支选择为顺序贪心（文档 §9.2 自承对耦合约束仅局部最优，有 C=[[1,1],[1,-1]] 反例） | 耦合约束下分支选择错误 |
| S-15 | `deltap_pw.cpp:24, 263`、`esolver_ks_pw.cpp:306` | `s_dp_escon` 为缓存值：drho 门控未通过时不更新但每步都被写入 f_en | 门控通过前的迭代沿用陈旧 escon（量级小但概念错位） |
| S-16 | `deltap_overlap.cpp:382` | SMO 重叠矩阵特征值 <1e-10 时方向静默丢弃（无警告） | 小晶胞大 rm 时 SMO 近线性相关 → 权重空间被静默裁剪 |
| S-17 | 全局 | 全局可变状态：`pw_deltap` 匿名命名空间全局量（deltap_pw.cpp:17-27）、`DeltaPOperator::s_stored_lambda` | 多 esolver 实例/restart/测试夹具间状态污染 |

---

## 6. 理论层面风险（与既往文档的关系）

以下按严重度排序，标注裁决状态。详见各引用文档。

| ID | 风险 | 依据 | 状态 |
|---|---|---|---|
| T-01 | **逐原子极化分解本质不唯一**：γ^I = Σ_n w^I_n·γ_n 依赖 SMO 选择，与 W90 硬归属给出定性不同的分配（液态水 O/H 比例两方法相反） | 07-02 §3.1 | **本质未解**；任何 per-atom 绝对值的物理解读都继承此不确定性。总和与差分是规范不变的（07-22 部分翻案） |
| T-02 | **2π 分支切割破坏跨构型连续性** → 位移差分法 BEC 不可行（三层 γ 给三个互相矛盾的 Z*） | 07-22 `bec-structural-limitation` | **结构性未解**（非实现 bug）；绕行方案 dF/dλ（H2O Z*_O=−1.75 已验证） |
| T-03 | **约束模式测的不是物理极化率**：测的是内场约束刚度 χ=dΣγ/dλ，与 efield 的 α=dμ/dE 差屏蔽因子，量级差 ~250× | 07-26 §4-5 | **机理定论**；"DeltaP 测极化率"的目标应转向 efield+FD |
| T-04 | HK 算符为一阶近似（非 γ 的精确泛函导数），w_eff 预条件器是经验的 | 07-09 §7 | 未解；实测方向可用 |
| T-05 | S_dk 缺位置修正项（O(dk) 还是 O(dk²) 争议**未裁决**）；修正版 `compute_S_dk_link` 已写但**从未被调用**（本轮代码确认其为死代码） | 07-12 C1 / 07-13 rebuttal | 缓解中，粗 k-mesh 有定量误差 |
| T-06 | L0 带交叉检测未实现：近简并相位下 Hungarian 匹配确定但不一定正确 | 07-15 | 未解；大体系外推时升级为 Critical |
| T-07 | 内循环 psi-lambda 不一致（mixing 后 ψ 与 H(λ) 不自洽） | 07-12 C3 / 07-13 | 共识降级 Medium，不修 |
| T-08 | 运行时非确定性：同 SCF 序列内已由 Hungarian+match freeze 修复；**跨独立 run 仅靠 deltap_match.dat**——而该文件在 MPI 下有写竞争（C-10） | 07-13 / 07-19 | 部分缓解，MPI 下机制不可靠 |
| T-09 | SCF 电荷-λ 耦合 limit cycle（曾 30 步不收敛、λ 发散） | 07-12 | 两阶段+mix_reset 缓解；**λ 只有 3-4 个离散能级**（一次性更新，无反馈）——大 λ/铁电体系未验证 |
| T-10 | 伪梯度缺 Jacobian（∇F=−2Jᵀr 中的 J 未计算，J≠−cI 时 CG 共轭性失效） | 07-12 M3 | 未解（内循环被旁路而绕开） |
| T-11 | SMO 权重基组依赖与不完备（Σ_I w^I_n≤1，共价电子在投影球外） | 07-02 §3.5 | 部分缓解（Löwdin）；跨基组可转移性未测 |
| T-12 | branched γ 破坏 Σ_I 守恒；raw/branched 混用已致一次 28% 极化率偏差 | 07-27 R1 | 已认知；**约束残差用 branched γ，分支选择又以 target 为锚——收敛指标 \|γ−t\| 部分反映分支选择而非物理收敛**，可能掩盖 λ 未真正驱动体系的事实（见 C-11） |
| T-13 | **λ→E_eff 换算漏 Ry→Ha 因子 2**：`esolver_ks_lcao.cpp:804` `E_eff_au = −λ_avg·π/a`（λ 为 Ry）应为 `−λ·π/(2a)`——本轮代码确认文档 07-27 R8 的怀疑属实；escon 符号未闭环验证 | 07-27 R8 | **未修**；所有对外 E_eff 表述（含 07-20 的 4.3 GV/m）需带因子 2 存疑 |
| T-14 | 力/应力修正文档闭环缺失 → 本轮代码审查证实确实不自洽（C-02） | 06-24 B.4/B.5 | **证实为 Critical** |
| T-15 | PW 路径为近似方案（on-site 投影约束 ≠ Berry 联络；escon=−Σλ·γ 在 PW 算符下不闭合）；PW 响应弱 ~6× 且不随 ecut 收敛 | 07-23 §5/6、07-27 R5/R6 | 固有限制 + 半成品（C-03/04/08/14 叠加） |
| T-16 | 三份 07-22 文档对 BN raw-γ BEC 结论互相矛盾（+4.76/−3.91 "合理" vs −13.46/+11.06 "错" vs Z*_B≈0.07 "基组限制"） | 07-22 三文档 | **未裁决**，引用 dev log 时需注意其 07-21 条目已被 07-22 推翻（"BN 零刚度"系 dp_escon 缺失假象） |

---

## 7. 代码质量 / 可维护性 / 性能

| ID | 位置 | 问题 |
|---|---|---|
| Q-01 | `deltap_solver.h:11-18` | **文件内含语法错误**（注释块在第 10 行关闭后仍有裸文本与孤立 `*/`）——能存在说明它从未被任何编译单元包含；死文件应删除 |
| Q-02 | `deltap_wannier.cpp:916-941, 959-1003, 1051-1067, 1177-1184`；`deltap_lcao.cpp:302-313`（static 计数器）；`deltap_overlap.cpp:198-214, 330-346` | SCF **每步**执行大量 debug 输出与文件写（Dmat dump、S^{-1/2} 自检 O(m³) 三重循环、branch_enum 写盘、zeta_debug append）——性能、磁盘、MPI 文件竞争（C-10）三重问题 |
| Q-03 | 多处 | 死代码群：`select_branch_set`（wannier:1822）、`compute_S_dk_link`（160，= T-05 的修复从未启用）、`compute_per_atom_gamma_from_becp`（deltap_pw.cpp:284）、`DeltaPOperator::update_lambda()`、`FletcherReevesCG::gradient_decayed`（bfgs.h:283）、cooldown 状态机（`start_cooldown` 有调用，`tick/inner_loop_cooldown` 从未调用）、`phase_corrections_` 只写不读（deltap_gauge.cpp:125）、`deltap_common.h` 除 `compute_dp_escon` 外全部未用（λ 更新逻辑在 LCAO/PW 两处内联重复，已产生 C-19 这样的不一致） |
| Q-04 | `esolver_ks_lcao.cpp:47-53, 927-935` | `dp_scf_/berry_ovl_scf_/r_overlap_scf_` 为 `void*` 裸指针成员，析构函数不释放（违 AGENTS.md RAII 规约；进程生命周期对象，实际泄漏可控但不合规） |
| Q-05 | `deltap.h:104` 注释 vs `input_parameter.h:626` | 文档/注释用 `deltap_nscf`，实际参数名 `deltap_inner_nmax`——多份文档同此漂移 |
| Q-06 | `deltap_wannier.cpp:287-1545` | `compute_wannier_polarization` 单函数 ~1250 行，远超 AGENTS.md 300 行重构线；缩进错乱（404 行起的 string 循环、1442-1506 行） |
| Q-07 | `deltap_wannier.cpp:1329, 1403, 1464` | 分支穷举 K=5 的复杂度 **O(11^nocc) 每原子每 SCF 步**：nocc=4 → 1.5e4（可行）；nocc=10 → 2.6e10（挂死）。**体系稍大即不可用**，需 deltap_nbands 或启发式 |
| Q-08 | `deltap_overlap.cpp:234-247, 312-322` | SMO 矩阵填充对每个矩阵元做 O(m_dim) 线性查找 → 总体 O(m_dim²·nw)；百原子体系 m_dim~900 → ~1e9 次查找 |
| Q-09 | `read_atoms_helper.cpp:458-460` | `[DS-DIAG]` 调试打印留在共享 STRU 解析路径（所有用 sc λ 的运行都输出） |
| Q-10 | 分支卫生 | 本分支捆绑了大量非 DeltaP 改动（deltaspin subspace ~5k 行、dftu PW port ~2.4k 行、cal_dm_psi 混合精度、print_info 格式变化），评审与合并面被人为放大 |

**回归面（deltap_switch=false 时的影响）**：
- R-1：`print_info.cpp` 输出格式变化（所有 LCAO 运行多打一列 NBASE）→ 可能破坏依赖输出基线的集成测试。
- R-2：**最大回归面**：`op_pw_proj.cpp`（408 行）+ `onsite_proj*` + force/stress/onsite kernels（CPU/CUDA/ROCm 全平台）被大改——这是 PW ΔSpin 与 DFT+U 的**共用路径**，必须跑全量 PW DFT+U/ΔSpin 回归才能合入。
- R-3：`charge_mixing` 新增 uom 混合（仅 `mixing_dftu=1` 启用，默认关）；U-Ramping 打印改用 getter（行为等价）。
- R-4：`operator_lcao.cpp` sc_lambda case 重构（删除的本来就是注释代码，行为等价）；新增 dp_lambda case 仅在算符存在时执行。
- R-5：`cal_dm_psi` 混合精度路径仅被 deltaspin subspace 调用；`unk_overlap_lcao` 新参数有默认值（兼容）；`parallel_global` finalize 防护（等价）。
- R-6：`read_input_item_postprocess.cpp:303-305` 放行 `berry_phase`+`calculation=scf`（deltap_corr 时）——改变了既有输入校验行为，需确认无滥用。

**测试覆盖现状**：`module_deltap/test` 仅覆盖 gauge 固定（4 例）、S/dS 数学一致性（3 例）、Wilson loop 平滑性（8 例）；`bfgs_test.cpp` 覆盖 FR-CG。**未覆盖**：hk 算符厄米性、力/应力 FD、MPI 多 rank、分支选择正确性、约束矩阵、esolver 集成、PW 路径全部。

---

## 8. 与既往评审（07-12/07-13）的关系

- 既往 18 项风险中的数学类条目（C2/H3/H4 等）已在代码中确认修复；C1（S_dk 位置修正）**代码证实仍未启用**（compute_S_dk_link 为死代码）；C3 共识不修。
- 既往评审的盲区（"18 项静态风险无法解释冻结电荷跨 run 不同"）由 Hungarian+match freeze 补上，但本报告 C-10 表明该机制的持久化层在 MPI 下不可靠。
- **本轮新发现**（既往未覆盖）：C-01~C-20 中除 C-05 与 C-11 的思想雏形外基本全是新发现，集中在：PW 实现全套（C-03/04/08/09/14、S-09/12/15）、力/应力（C-02）、MPI 能量与文件 I/O（C-01/10）、gdir≠3 与多 string 的错配（C-05/S-01）、relax/MD 失效（C-12/13）、分支晶格口径（C-11）、默认方法路径质量（C-15/16/17）。
- 文档间矛盾新增一例：dev log "Active Bugs" 仍列 Z01（30s/k-pair）为 Open，但 07-20 记录快速路径已上线——dev log 滞后（本次已顺带核实 `berryphase_overlap` 确有 MPI gather+Allreduce，快速路径为生产路径）。

---

## 9. 优先行动清单

**P0（任何 merge/对外结果前必须完成）**
1. C-01：dp_escon 移出 rank 守卫（一行修复）。
2. C-04：hamilt_pw.cpp 传参改 `deltap_switch && deltap_corr`（一行修复）；C-03：λ 与 target 分离（小改）。
3. C-06：hk_correction 越界——加 nrow≠ncol 守卫或修正维度（配合 MPI np>1 复现测试）。
4. C-02：力/应力与 H 的自洽性改造 + FD 验证；**在此之前声明 deltap_corr 不支持 relax/MD**。
5. C-12/C-13：跨离子步状态重置与 hR 重建补加——relax/MD 支持的前置。
6. R-2：PW DFT+U/ΔSpin 全量回归（共用 onsite 路径）。

**P1（结果可信度）**
7. C-05：gdir≠3 方向错配修复 + 三方向各跑一次约束测试。
8. C-11：分支晶格口径统一 + 构造性单测（已知跳变 → 命中正确分支）。
9. C-08：PW 假内循环下线或 WARNING_QUIT；C-14：VR 索引修正。
10. C-09：γ_total==0 与失败解耦（返回 optional）；C-20：G-phase 用推断后的 nmp。
11. T-13：E_eff 因子 2 修正并回溯修订所有对外表述；escon 符号 FD 闭环。
12. C-10/Q-02：debug 输出与文件 I/O 全部加 rank 守卫 + 开关参数；Q-01/Q-03 死代码清除。
13. S-06/S-07/S-08/S-10：输入校验（nspin、正交晶胞、k 网格下界、绝缘体、target 文件存在性、constraint_mode 合法性）。

**P2（能力与效率）**
14. Q-07：穷举搜索替换为启发式/限带（deltap_nbands）；Q-08：SMO 索引预建表。
15. S-01：w_eff 与 string 数据对齐；T-05：启用 compute_S_dk_link 并裁决阶数争议。
16. Q-06：compute_wannier_polarization 拆分（Wilson loop / 匹配 / 分支选择 / IO 四个函数）。
17. T-06：L0 带交叉检测；T-03 后续：极化率工作流切换到 efield+FD。

---

## 10. 本轮规约记录（AGENTS.md 格式）

**测试计划**：对 feat/deltap 分支 DeltaP 理论与实现做系统风险评估——(a) 理论文档与既往评审一致性；(b) LCAO 实现（γ 计算、约束算符、λ 优化、力/应力、esolver 集成）；(c) PW 实现与公共模块；(d) MPI 正确性；(e) deltap 关闭时的回归面。

**测试设置**：静态审查（Read/Grep/git diff merge-base `b9669d37`..HEAD `93a0d782`）；输入：19 份核心设计文档 + deltap-development-log + 30 余个源文件（含 deltap_wannier.cpp 2205 行全量）；未构建未运行。

**结果**：登记风险 61 项——Critical 8（C-01/02/03/04/05/06/08/11）、Important 12（C-07/09/10/12/13/14/15/16/17/18/19/20）、疑似/脆弱 17（S-01~17）、理论 16（T-01~16，含 3 项本质限制）、质量 10（Q-01~10）、回归注意 6（R-1~6）。既往 18 项：确认已修 8、降级关闭 4、仍在 3、新发现不在其列 20+。

**分析**：风险集中于三处：(1) **算符-力-能量三角不自洽**（C-02、T-13/14）——根因是 HK 算符的解析导数从未实现，力是从 DeltaSpin 模板复制后未适配 τ_α 因子；(2) **MPI 状态管理**（C-01/06/10、S-09）——git 历史显示 MPI 修复是打补丁式的（改 Allreduce→Bcast→加守卫），缺乏"哪些状态需要跨 rank 一致"的清单化设计；(3) **分支选择**（C-11、T-02/12）——物理上本就脆弱，实现又有三套不一致的平移口径。PW 路径整体处于演示前状态。

**下一步**：按 §9 P0 清单执行；每个 P0 修复配一个可复现测试（MPI np=2 smoke、FD 力验证、gdir=1 约束、STRU-target PW 启动）；完成后更新本报告状态列并重跑 BN/H2O 基线。
