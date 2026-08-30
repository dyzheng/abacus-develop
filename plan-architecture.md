实空间权重约束框架：模块化代码架构与逐模块验证设计
目标：把「统一实空间权重约束框架」落成可独立开发、独立验证的模块体系。 依据：4 份评审文档（框架评估 / 技术清点 / 复杂度 / 创新性）+ DeltaSpin 参考实现（npj Comput. Mater. 12, 52 (2026)）。 组织原则：五层解耦 + 每个模块一个不依赖下游的验证入口。

0. 设计原则（先钉死，所有模块据此判定）
权重与基组解耦——权重定义在实空间网格，PW 与 LCAO 走同一套 w(r)，这是核心卖点（双基组同口径），任何模块不得把基组概念渗入权重层。
「观测量 = 注入算符」硬约束——V_con=Σμ_α·w_α、读数 Q_α=∫w_α·d_α 用同一个 w，天然满足 dspin 恒等式三条件（S2）。这是 λ 成为精确 torque 的前提，架构上必须保证权重对象在「注入侧」和「读数侧」是同一个实例。
每个模块可独立验证——验证不依赖尚未实现的下游模块（权重验证不依赖 SCF，求解器验证不依赖真实 SCF）。
冻结权重——Becke（纯几何）/固定 promolecule Hirshfeld，μ 只在外环更新；禁止密度依赖权重（T-17 活算符极限环教训）。
复用现成骨架，不重造轮子——DeltapScfSolver::Backend 回调抽象、spin_constrain lambda_loop、efield/gatefield veff 注入、module_gint、partition.h Becke 原语全部直接承接。
1. 分层总览与依赖关系
Layer 5  I/O / 审计契约  (M7)          ── 配置、语义守卫、输出证书
Layer 4  导数层          (M6)          ── 力 / 应力 / 力矩
Layer 3  控制层          (M4, M5)      ── 外环 μ 求解器 + 记账恒等式
Layer 2  SCF 耦合层      (M3a, M3b)    ── 约束势注入 (PW veff / LCAO Gint)
Layer 1  网格数据层      (M1, M2)      ── 权重网格 + 约束读数
Layer 0  纯数学层        (M0)          ── 权重数学核 (无 ABACUS 依赖)
依赖规则：上层只依赖下层，同层不互相依赖，跨层只通过接口。

M7 ─┬─> M4,M6 (配置) ─┐
M6 ─┴─> M1,M2         ├─> M3a ─> M1
M5 ────> M2,M4        ├─> M3b ─> M1
M4 ────> M2           └─> M2 ─> M1 ─> M0
2. 核心抽象（C++ 接口，作为模块间唯一契约）
// Layer 0
struct WeightField {            // w_α(r), 逐点求值
  virtual void eval(const Vec3& r, std::vector<double>& w) const = 0;         // Σ_α w_α = 1
  virtual void eval_deriv(const Vec3& r, int atomJ, std::vector<double>& dw) const = 0; // ∂w_α/∂R_J
};

// Layer 1
struct DensityChannel {         // 密度通道读数 (解耦基组)
  virtual double rho_up(const Vec3& r)  const = 0;
  virtual double rho_dn(const Vec3& r)  const = 0;   // nspin=1 时恒 0
};

struct IConstraintObserver {    // Q_α = ∫ w_α d_α
  virtual void observe(const WeightField& w, std::vector<double>& Q) const = 0;
};

// Layer 2
struct IPotentialInjector {     // 注入 V_con = Σ μ_α w_α
  virtual void inject(const std::vector<double>& mu) = 0;
};

// Layer 3
struct IMuSolver {              // 外环：给定 (Q, target, 历史) → 新 μ
  virtual void step(const std::vector<double>& Q,
                    const std::vector<double>& target,
                    std::vector<double>& mu,
                    Status& st) = 0;
};

struct IAccounting {            // 记账与恒等式自检
  virtual void check_identity(const std::vector<double>& Q,
                              const std::vector<double>& mu) const = 0;
};
关键约束：注入侧与读数侧必须共享同一个 WeightField 实例（原则 2）。这由工厂保证：每个约束一个权重对象，分别把只读引用传给 injector 与 observer。

3. 模块逐一分解（职责 / 复用点 / 独立验证）
M0 权重数学核（weight_math）— Layer 0，纯函数，无 ABACUS 依赖
职责：Becke 权重、Hirshfeld promolecule 权重、位置导数。

Becke 3 阶迭代多项式 f_3（已存在：module_grid/partition.h 的 w_becke/s_becke）、Stratmann 屏蔽变体、异核半径比修正 χ_ij（缺口）。
Hirshfeld：promolecule 径向表 + 球平均 + dρ/dr 导数表（数据源复用 charge_init.cpp 的 atomic_rho）。
解析位置导数 ∂w/∂R（缺口，力需要）。
复用点：partition.h（Becke 原语 + 单测 test_partition.cpp）；charge_init.cpp（Hirshfeld 数据源）。

独立验证（单测，无网格、无 SCF）：

解析闭式对拍：Becke f_3 在给定原子坐标下的解析值 vs 手算/高精度数值。
单位分解断言：任意随机点 Σ_I w_I ≡ 1（容差 ~1e-10）。
FD 导数：∂w/∂R 解析 vs 中心差分（δ=1e-5 Bohr，误差 < 1e-6）。
对称性：对称分子（H₂O）两 H 权重逐点相等。
已有 test_partition.cpp 扩充即可，不接任何 SCF。
M1 网格权重构造（weight_grid）— Layer 1，绑定 FFT 网格
职责：把 M0 的逐点权重构造成网格数组 w_I(r_g)，含近邻表缓存（O(N_g×N_at×N_neigh)）、并行域分解（复用现有网格并行）。

复用点：ABACUS 电荷/XC 网格并行；WeightField 接口。

独立验证（有网格、无密度、无 SCF）：

逐点 sum rule 硬断言：max_g |Σ_I w_I(r_g) − 1| < 1e-10，作永久审计行。
对称性：对称分子权重场对称。
并行一致性：1 rank vs 多 rank（含非方进程网格）逐点一致。
内存/复杂度：基准测 O(N_g×N_at×N_neigh) 实际耗时，确认「远小于一步 SCF」。
M2 约束读数（constraint_observe）— Layer 1
职责：Q_α = ∫ w_α(r) d_α(r) dr。电荷通道（ρ）、自旋通道（m，nspin=2 的 ↑−↓）、片段/线性组合（约束矩阵 C 分组求和，复用 DeltaP 约束矩阵设计）。

复用点：module_dipole（偶极权重 w=z 的网格读数先例）；DeltaP 约束矩阵 C。

独立验证（固定已知密度，不跑 SCF）：

原子叠加密度：给一个已知 ρ（原子密度叠加），Q_I 对拍解析期望。
守恒和：电荷通道 Σ_α Q_α = N_el（精确）；这是 sum rule 的读数侧对应。
偶极对拍：w=z−z₀ 时 Q 与 module_dipole 输出一致（对拍先例 <1e-3）。
口径一致性：同一密度、同一 w，PW 读数 ≡ LCAO 读数（双基组同口径基准数据，这是增量 1 的可发表点）。
M3 约束势注入（constraint_inject）— Layer 2
M3a（PW）：V_con(r)=Σ μ_α w_α(r) 加进 veff 网格。复用 efield/gatefield 模块（外部势进 veff 的生产先例，F-2 对拍 <1e-3 eV）。

M3b（LCAO）：W^α_μν = ∫ φ_μ w_α φ_ν dr，走 module_gint；力核借 gint_dvlocal 网格-力基建。这是全新接线（R8），按新建通道估工程量。

独立验证（冻结密度/单步，不需要外环）：

单步注入正确性：固定密度、给定 μ，断言 veff_new(r) = veff_old(r) + μ w(r) 逐点成立（对 PW 平凡可验）。
线性响应（与 M4 集成）：真实 SCF 下 dQ/dμ ≈ −χ（负、量级合理），对照 CDFT 标准行为。
PW≡LCAO 同网格同 Q：同一 μ、同一网格，两基组 Q_α 逐位一致（口径不变量）。
偶极对偶侧（V4，三期）：efield 已有，约束侧交叉验证需细化协议（R5 高危）。
M4 外环 μ 求解器（mu_solver）— Layer 3
职责：逐分量 secant + κ clamp [0.3,20]（补上，R2 缺口）+ 翻号检测 + 单步限幅 + 熔断（μ_max 顶限 + 残差平台联合判据）。一期只用逐分量 secant，Broyden 降二期（R3）。

复用点：spin_constrain 的 lambda_loop（BFGS/PR-CG）外环骨架、sc_scf_thr_mode 门控、escon 记账。

独立验证（用合成 Q(μ) 映射，隔离求解器 bug 与物理 bug）：

Mock 测试：给合成单调映射 Q(μ)=Q₀−χ·μ，验证 secant 在已知根收敛、步数/残差达标。这一步不需要任何真实 SCF，是求解器最干净的独立验证。
翻号/发散护栏：给非单调或双侧不可达映射（模拟 R3「死通道」），断言翻号检测与熔断触发，而非发散。
反假收敛检验：μ=0 自由跑不得收敛到任何非自然靶点（R9 纪律）。
可达靶点实测：真实 SCF 下可达约束 ΔQ→0（约束精度 <2 mrad，H₂O 先例）。
M5 记账与恒等式（constraint_accounting）— Layer 3
职责：E_con=Σ μ_α Q_α、dspin 恒等式三条件自检、每 SCF 永久审计行（Σ_I N_I vs N_el）、残差/线性度/口径元数据哈希。

复用点：DeltaP 的 ⟨η⟩ 完备性指标、T0 双口径 12 位核对方法论。

独立验证（在已收敛约束上）：

E′(λ) 线性项：< 2.5e-5 Ry/Ry（F-7/F-7b 先例阈值）。
记账恒等相消：escon = −λ·Γ 逐点恒等（Route A+ 结构）。
审计行永久输出：每 SCF 打印，机器可读。
M6 力 / 应力 / 力矩导数（constraint_deriv）— Layer 4
职责：F_J=∫ρ ∂w/∂R_J dr（解析，借 gint_dvlocal）；力矩 λ=∂E/∂M（内环优化值，输出为物理观测量）；应力（三期，R6 存疑待证）。

独立验证（FD 交叉，最高危，逐项按历史处方）：

力 FD（stationary4 协议）：冻结 t*、δ=0.005 Bohr、判据 0.0129 eV/Å；网格前提必须写死：ecutwfc=100 + ecutrho≥400 + scf_thr=1e-8（R7，否则假 FAIL）。
力矩 FD（对标 DeltaSpin Fig.4b）：扰动目标磁矩 δM，对比 λ_解析（内环收敛值）vs ∂E/∂M（数值微分），目标误差 < 0.006 eV/μB。这是「能力与 DeltaSpin 一致」的判决实验。
应力 FD（三期）：补推导 + FD；一期若声明支持周期应力则必须补，否则标注「应力未验收」。
M7 I/O / 配置 / 审计契约（constraint_io）— Layer 5
职责：JSON/YAML 解析、delta vs absolute 语义守卫（口径从 ~e 降到 ~0.2 e，absolute 模式打印显式 WARNING，R12）、输出契约（残差、sum rule 审计行、收敛元数据、口径元数据哈希）。

复用点：module_deltaspin 的 JSON 解析。

独立验证（纯，无物理）：

schema 测试 + 往返测试。
语义守卫测试：老输入文件（旧口径）触发 WARNING，不静默产生错误物理。
输出契约字段完备性 + 哈希可复现性。
4. 分阶段实施映射（模块 → 阶段 → 验收）
阶段	交付模块	基组	约束类型	验收（V 项）
一期（判决性）	M0(仅 Becke) + M1 + M2 + M3a + M4 + M5 + M7(最小)	PW	单/双电荷	V1 sum rule + V2 Hirshfeld 基准 vs 文献 + V3 小 q 可达性 + 反假收敛
二期	M3b + M6(力/力矩) + 自旋通道(±μ 复用 DeltaSpin)	PW+LCAO	电荷 + 磁矩	力矩 FD（对标 DeltaSpin） + 力 FD(stationary4) + PW≡LCAO 逐位一致
三期	M6(应力) + Hirshfeld 权重 + Broyden + 偶极(V4) + 输出规范	双	偶极/多极子	应力 FD + 偶极对偶交叉验证（R5 细化协议）
一期止损判据：若 V3 显示物理可达域依然过窄（R4，体系刚度 κ 封顶、换权重只除假饱和），及时止损——这是最大物理不确定性。

5. 逐模块独立验证矩阵（总表）
模块	独立验证方式	关键验收标准	依赖下层	复用点
M0 权重核	单测	Σw≡1 (1e-10)；FD 导数 <1e-6；对称性	无	partition.h, test_partition.cpp
M1 权重网格	网格断言	max|Σw−1|<1e-10；并行一致	M0	网格并行
M2 约束读数	已知密度对拍	ΣQ=N_el；偶极对拍 <1e-3；PW≡LCAO	M1	module_dipole, 约束矩阵 C
M3 注入	单步注入 + 线性响应	veff 逐点 +=μw；dQ/dμ≈−χ；PW≡LCAO	M1	efield/gatefield, module_gint
M4 求解器	合成 Q(μ) mock	已知根收敛；翻号/熔断触发；反假收敛	M2	spin_constrain lambda_loop
M5 记账	恒等式自检	E′(λ)<2.5e-5 Ry/Ry；审计行	M2,M4	⟨η⟩ 完备性
M6 导数	FD 交叉	力 0.0129 eV/Å；力矩 0.006 eV/μB；应力(三期)	M1,M2	gint_dvlocal, stationary4
M7 I/O	schema/往返/守卫	旧输入触发 WARNING；契约完备	—	module_deltaspin JSON
6. 风险与回滚
R8 工程量：M3b（LCAO Gint 新通道）按「新建通道」估，不按「改造」估——这是二期最大工程量项。
R6 应力：M6 应力「无贡献」说法存疑，一期不承诺、三期补 FD，避免过度承诺。
R2/R3 外环：M4 必须补 κ clamp + 翻号检测（历史 secant 翻号振荡、死通道实录），mock 测试先于真实 SCF。
R4 物理上限：换权重只除「假饱和」，物理可达域由体系刚度 κ 封顶，不可跨体系外推 μ 量级表（0.01–0.1 Ry 需逐体系标定）。
回滚策略：每层独立，下层通过验证后冻结接口；上层失败不回滚下层。旧 SMO 路径（pre_hr）保留作对照输出，设下线时间表。
附：一句话落点
这套架构把「能不能做到 DeltaSpin 级别的精确力矩」从口号变成可执行的验收链：M0–M5 先各自独立验证，M6 的力矩 FD 是最终判决——任何一层验证不过，都能定位到具体模块而不牵连全局。 一期成本 ≈ 「一个自包含权重模块（1–2k 行）+ PW 侧小接线 + 一个 FD 测试」，是低投入高信息的判决点。