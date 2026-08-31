# 词汇表：DeltaP/DeltaSpin 与实空间权重约束框架术语·编号索引

> 目的：为无法直接访问仓库全文的协作者提供术语与编号的权威出处。每条注明定义与仓库内出处（文件:行/章节）。
> 维护纪律：新术语/新编号出现时同步本文档（与 AGENTS.md 文档纪律同级）。

## 1. 投影与算符

| 代号 | 含义 | 出处 |
|---|---|---|
| **SMO** | 仓库内唯一的展开式记录是 **Smoothed Maximum Overlap**（2026-07-11-deltap-smo-projection-evaluation.md §1）。指同一套原子投影轨道 {α^I_lm}——第一 zeta 数值原子轨道（NAO），截断到 onsite 半径并平滑。注意：它不是 "Symmetrized Mulliken Orbitals"；《实空间权重约束框架设计方案.md》中"SMO（单 zeta NAO 简单投影）"是操作性描述而非缩写展开 | 见下"两个口径" |
| **SMO 简单投影** | P̂_I = Σ_lm \|α^I_lm⟩⟨α^I_lm\|，逐轨道外积求和、不含交叠度量 S。DeltaSpin 的 pre_hr 即此口径 | dspin_lcao.h:107 |
| **SMO/Löwdin 投影** | 同一套轨道经 Löwdin 正交化：\|α̃⟩ = S^{−1/2}\|α⟩，tilde_proj = S^{−1/2}·proj。DeltaP 逐原子分解用此口径 | 2026-07-11 §1 |
| **raw_sum / tilde_sum** | 对轨道 ψ_n：raw_sum(n) = Σ_{I,lm} \|⟨α^I_lm\|ψ_n⟩\|²（未正交化权重和）；tilde_sum(n) = Σ_{I,lm} \|(S^{−1/2}proj)_{lm,n}\|²（Löwdin 后权重和）。H₂O 实测 raw 5.91、tilde 5.21（n=0）/ 18.35（n=1），**两者都 ≠1**——单位分解在两个口径下均失败；tilde 并非系统性更大 | 2026-07-11 §3 |
| **pre_hr** | LCAO 侧 SMO 简单投影矩阵（HContainer）：pre_hr_{μν}^I(R) = Σ_lm⟨φ_μ\|α^I_lm⟩⟨α^I_lm\|φ_ν(R)⟩ | dspin_lcao.h:107-108 |
| **becp / dbecp** | PW 侧投影系数 ⟨β^I_lm\|ψ_n⟩（OnsiteProjector 的 β 投影器，与 USPP 非局域投影同族）及其位置导数（力链用） | onsite_proj.h:47 |
| **OnsiteProjector** | PW 侧 β 投影器对象（G 空间 tabulate_atomic） | onsite_proj.h:18 |
| **P_I_sub** | LCAO 子空间加速的投影矩阵 C†P_IC | SUBSPACE_USAGE.md |

**"SMO 病态口径"的准确含义**（校正表述用）：病态的不是投影轨道本身，而是"用无度量的外积和作约束观测量"这一**测量口径**——非单位分解（raw/tilde 均失败）、PW/LCAO 口径分裂（SMO vs β 投影器）。仓库 module_deltaspin 至今沿用该口径；npjcm 2026 DeltaSPIN 论文描述的即此实现（仓库内无 "modulated NAO" 字样，论文术语以论文为准）。

## 2. 能量记账

| 代号 | 含义 | 出处 |
|---|---|---|
| **escon** | fp_energy.h:48 注释为 "spin constraint energy"（约束能量记账量；缩写原始语义即 spin 约束能，后泛化）。DeltaSpin：escon = −Σλ·M；家族成员：dp_escon = −Σλ·γ（DeltaP）、cc_escon = Σμ(Q−t)（实空间权重约束） | fp_energy.h:48-50 |
| **E_con** | 实空间权重框架的约束能 Σ_α μ_α(Q_α − t_α)，即 cc_escon 的记账式 | 2026-08-31-m5 |

## 3. DeltaP 专属符号

| 符号 | 含义 |
|---|---|
| **γ** | 逐原子分解的 Wilson loop Berry 相位（带 2π 分支离散、非算符观测量）；约束的驱动观测量 |
| **Γ** | 约束算符期望 ⟨Ô⟩（Route A+ 的记账/驱动观测量，无分支离散问题）；PW 侧 Γ^PW = ⟨P̂^onsite⟩ |
| **t_Γ\*** | 驻点 FD 协议中冻结的代理靶点（自然 Γ 值，位移过程中不跟随分支） |
| **Route A / A+ / A++** | DeltaP 三代算法路线：A=直接约束 γ（力 FAIL 84.8–419 eV/Å）；A+=记账改 Γ（现行生产，力 0.0138 eV/Å）；A++=EFC 场算符修正提案（未实施） |
| **ow 模式** | Ô_w 精确权重算符（T-18 判决 FAIL，挂起） |

## 4. 协议与指标

| 代号 | 含义 | 出处 |
|---|---|---|
| **stationary4** | 第 4 代驻点 FD 力验证协议（仓库仅保留此版，tests/deltap_fd_force/tools/run_stationary4.sh）：冻结 t\*、δ=0.005 Bohr、判据 0.0129 eV/Å、网格前提 ecutwfc=100+ecutrho≥400+scf_thr=1e-8 | dev-guide §4 |
| **⟨η⟩** | SMO 完备性指标 max(0, 1−Σ_I w_In)；2026-08-17 确认不预测力残差（L13） | capability-boundaries L13 |
| **dspin 恒等式三条件** | 力/能量记账自洽的前提：①观测量=H_c 算符期望；②记账恒等相消（E′≡E_KS(ψ\*)）；③约束驻点 | dev-guide §2.3 |

## 5. 实验/发现编号（T-/F-系列，DeltaP 历史）

编号出处：任务卡见 2026-08-04-deltap-execution-todo.md；实测记录见各 dated spec 与 deltap-development-log.md。高频引用项：

| 编号 | 一句话 |
|---|---|
| T-4a′ | secant 外循环失控（λ 跑到 ±1.3 Ry、κ 翻号振荡）——外环护栏必要性的实录 |
| T-5′ | 窗口测绘：λ 预算封顶 0.1 Ry；H 通道双侧死（Δγ_H≤0.001 rad） |
| T-7p / T-7″ | 逐分量（对角 Jacobian）secant 单元层 1 步收敛 / 冻密度响应符号反转、结构性不收敛（F-4 封口） |
| T-17 | 活算符极限环 vs 冻结核全窗收敛——冻结权重设计的实验背书 |
| T-18 | Ô_w 精确算符不提升耦合——窗口由体系物理刚度封顶的判决 |
| F-2 / F-2b | 场模式能量对拍通过 / 场模式力 Maxwell 失配 17.7× |
| F-7 / F-7b | PW escon 恒等式逐点验证 / "结构性失配"假说证伪（测量态伪影） |
| F-8 | H_HK 应力交付（内部 Maxwell 6.8–7.0% = 冻结-C 隙 L12） |

## 6. 实空间权重框架自编号

M0–M8（模块）/ V1–V7（验证）/ R1–R12（风险）/ S1–S6（支撑）/ P1–P4（评审问题）：定义见 plan-architecture.md 与 2026-08-26-realspace-weight-framework-evaluation.md；进度与测试见 2026-08-31-constraint-framework-progress-summary.md。

## 7. 无出处记录（评审 R1 在案）

"μ→−1244 Ry" 与 "H₂O ΣN_I≈11.48 vs 8"：全仓库 grep 无出处，出自设计方案所引的外部"原文档"（未入库）。**在未提供可复现算例前，不得作为论据引用**（2026-08-26 评审 R1）。
