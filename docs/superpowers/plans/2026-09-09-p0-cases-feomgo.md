# P0 算例逐步实施计划：II-1 FeO 自旋约束 + I-1 MgO 宽电荷扫描

> 依据：`2026-09-09-validation-case-suite-design.md` §1/§2（P0 两项）。
> 基线资源（已核实）：`tests/17_DS_DFTU/11_PW_DFTU_S2_FeO`（FeO，PW/dav_subspace/nspin=2/DFT+U=5 eV）、`12_PW_DS_S2_Z`（Fe 基 DeltaSpin 对照例）；Mg/O 赝势与轨道在库。
> 全程纪律：先冒烟后批量；每步有判据；spec 五段式；结果对干净 commit。**本文档批准前不提交任何计算。**

---

# 算例 II-1：FeO 自旋约束 + DeltaSpin 口径对照

## II-1.0 目的与验证问题

1. TM d 电子局域矩的自旋约束能力（生产场景第一证）；
2. 约束框架与 DFT+U 同开的兼容性（两条势通道：on-site 投影 vs veff 网格，从未同测）；
3. 开放项④结案：同体系同靶点下本框架 μ vs DeltaSpin λ 的**定量**口径归因。

**依赖声明**：问题 3 的干净做法需要 V3b 只读观测口（在同一收敛密度上分别测 Becke 矩与 on-site 投影矩）——**II-1 拆成 II-1a（能力验证，V3b 无关）与 II-1b（口径测量，需 V3b 先交付）**，不阻塞主链。

## II-1.1 设置（继承 FeO 基线，改动最小化）

- 体系：`tests/17_DS_DFTU/11_PW_DFTU_S2_FeO` 的 STRU/KPT 原样（rocksalt FeO，AFM；Fe 子晶格原子下标在搭建时从 STRU 核实登记）；
- INPUT 骨架（相对基线的改动用 `←` 标出）：

```
suffix            autotest
calculation       scf
basis_type        pw
ecutwfc           20
gamma_only        0
nspin             2
scf_thr           1.0e-7        ← 基线 1e-6 收紧一档（约束读数精度需要）
scf_nmax          100
smearing_method   gaussian
smearing_sigma    0.01
mixing_type       broyden
mixing_beta       0.4
ks_solver         dav_subspace
symmetry          0
dft_plus_u        1
orbital_corr      2
hubbard_u         5.0
onsite_radius     3.0
pseudo_dir        ../../PP_ORB
orbital_dir       ../../PP_ORB
pw_seed           1
constraint        true           ← 新增
constraint_type   spin           ← 新增（v1 单通道即可）
constraint_target_file  constraint_target.json
constraint_target_mode  delta
constraint_mu_max 5.0
constraint_thr    1e-4
```

- 靶文件：`{"targets":[<delta>],"atoms":[[<Fe 子晶格下标>]]}`——delta ∈ {+0.1, +0.3, +0.5, −0.1, −0.3, −0.5} μB（Fe 基态矩 ~3.5 μB 量级，±0.5 是温和扰动）；
- 注意：DFT+U 的 `onsite_radius` 与 Becke 权重半径是**两套独立分区**——spec 中明确记录两者不冲突的论证（注入通道不同）。

## II-1.2 执行步骤与判据

| 步 | 内容 | 判据 |
|---|---|---|
| S1 | 冒烟：约束关、基线设置重跑 | 与基线 result.ref 一致（确认复现性）；记录 Fe/O 的 Becke 矩读数（审计行） |
| S2 | 参考态：约束开、delta=+0.1 单点 | CONVERGED（外步 ≤10）；μ* 有限（\|μ\|<1 Ry）；maxdev=2.2e-16；记录 Q_ref |
| S3 | 扫描：delta 六点（热启动续算：上一 δ 的密度作初猜，计划 §3.1） | 全点达标无熔断；μ(δ) 单调；记录 μ–M 表 |
| S4 | 熔断用例：delta=+3.0 μB（超物理可达） | UNREACHABLE + Q(μ) 端点（不发散） |
| S5 | DeltaSpin 对照：同体系同靶点（+0.1/−0.1 两点）跑 `12_PW_DS_S2_Z` 设置 | 两实现收敛矩差 <0.01 μB；**μ/λ 符号一致**（μ=−λ 在真实体系复验）；量级差如实记录（归因留 II-1b） |
| S6 | 汇总：μ–M 响应斜率 dM/dμ、κ_eff、与 H₂O 自旋通道（−1.38 e/Ry）对比 | spec 五段式 |

## II-1.3 成本估计

S1–S5 ≈ 9 次 SCF（小原胞 + dav_subspace + kpar 可用）——预计 **2–4 h np4**。力 FD 不在本算例（留给 V1/V2 链）。

## II-1.4 风险

- DFT+U+约束同开若 SCF 震荡：先关 U 做对照隔离（两通道独立性判定实验）；
- AFM 反平行子晶格需约束**两个 Fe 子晶格**（symmetric pair）时改为两条约束（A6 混合能力顺带复用）；
- nbands 不足（基线注释掉了 nbands=28）：若空带不足致磁矩不可调，放开 nbands=28。

---

# 算例 I-1：MgO 体相宽电荷扫描（R4 判决实验）

## I-1.0 目的与验证问题

**唯一核心问题**：离子体系的电荷约束线性可达域是否显著宽于 H₂O（±0.3 e）？
- 若 ±0.5–±1.0 e 仍线性可达 → 设计方案"DeltaQ 正当适用域=离子/缺陷体系"论断成立（P0 通过）；
- 若同样窄 → 框架适用范围声明必须收紧（同等重要的否定性判决）。

## I-1.1 设置

- 体系：MgO rocksalt，常规 8 原子晶胞（a=4.21 Å；分数坐标 Mg (0,0,0) 等——搭建时按标准结构生成并在 spec 记录）；Γ 点 + 2×2×2 MP 网格标定；
- 基组：**LCAO**（Mg_gga_8au_100Ry_4s2p1d.orb + O_gga_7au_60Ry_2s2p1d.orb，在库）；
- INPUT 骨架：

```
suffix            autotest
calculation       scf
basis_type        lcao
gamma_only        0
nspin             1
ecutwfc           60            ← 标定档（先测噪声，不足升 80/100）
ecutrho           240           ← 同上（网格标定：噪声 ≪ 1e-4/3 即可）
scf_thr           1e-8
scf_nmax          200
smearing_method   gaussian
smearing_sigma    0.002
mixing_type       broyden
mixing_beta       0.7
ks_solver         genelpa       ← LCAO 对角化（小矩阵，便宜）
symmetry          1
pseudo_dir        ../../PP_ORB
orbital_dir       ../../PP_ORB
constraint        true
constraint_type   charge
constraint_target_file  constraint_target.json
constraint_target_mode  delta
constraint_mu_max 5.0
constraint_thr    1e-4
```

- 靶文件（扫描序列）：O 片段 `{"targets":[d],"atoms":[[<O 下标>]]}`，d ∈ {+0.3, +0.5, +0.8, +1.0, −0.3, −0.5, −0.8, −1.0}；**另加 Mg 片段 ±1.0 两点**（Mg⁰ 化 vs Mg³⁺ 化——后者预期熔断，是物理熔断用例）。

## I-1.2 执行步骤与判据

| 步 | 内容 | 判据 |
|---|---|---|
| S1 | 网格标定：ecutwfc=60/ecutrho=240 与 80/320 各 1 次无约束 SCF | Q_O 差 <3e-5（=thr/3）；不足则升档 |
| S2 | 参考态：无约束 SCF | 收敛；记录 Q_ref(O/Mg)（Becke 电荷）；与文献带（MgO Hirshfeld/Becke ~±1.0–1.5 e）带宽核对（口径级，非精确对拍） |
| S3 | O 片段扫描 8 点（热启动续算，\|δ\| 递增序） | 逐点记录 μ–Q；**线性区宽度**（\|dμ/dQ−κ\| 偏离 <20% 的 δ 区间）；超线性区的点如实记非线性不判 FAIL |
| S4 | Mg 片段 ±1.0 两点 | +1.0（Mg⁰ 化）预期可达/部分可达；−1.0（Mg³⁺ 化）预期 UNREACHABLE——验证熔断+端点报告 |
| S5 | 判决：对比 H₂O 的 κ≈1.7/±0.3 e 线性区 | 明确裁定：离子体系线性区是否 ≥2× 于 H₂O；κ_eff(O in MgO) 报告 |

## I-1.3 成本估计

LCAO 8 原子晶胞、scf_thr=1e-8：单次 SCF 预计 1–3 min；S1–S4 ≈ 13 次 SCF + 外环开销 ≈ **1–2 h**。

## I-1.4 风险

- bulk 周期像下 Becke 权重分区的正确性（权重随周期像求和——M1 已支持，本算例是首个 bulk 实证：S2 的 sum rule 审计即判据）；
- MgO 是绝缘体（Eg~4.5 eV @PBE），大 delta 下可能出现占据翻转（如实记录为物理事件，判据只考核 constraint_thr 内行为）；
- smearing 0.002 下的电荷读数稳定性——若 drho 震荡则改 0.001 + scf_nmax 300。

---

# 两算例的关系与顺序

1. **先 I-1 后 II-1**（I-1 资源零缺口、问题最核心；II-1 有 DFT+U 兼容性变量）；
2. 两算例的 spec 各自独立；μ/κ 数据汇总进阶段 B 立项评审；
3. II-1b（口径测量）排期在 V3b 只读观测口交付后；
4. 全部完成后更新验证算例集文档 §5 总表的状态列。
