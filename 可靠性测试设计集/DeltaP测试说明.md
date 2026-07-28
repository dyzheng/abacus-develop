# DeltaP 算法测试说明（P 系列）

> 对应文档: `可靠性测试设计集/` (P01–P18)
> 分支状态: `feat/deltap` (2026-07-27)
> 前置阅读: `docs/superpowers/specs/2026-07-27-h2o-polarizability-all-rounds.md`（R1–R8 全记录）

---

## 一、算法当前状态速查

### 1.1 代码路径

| 路径 | 基组 | 约束模式 | 关键文件 |
|------|------|---------|---------|
| LCAO | 数值原子轨道 | total / per_atom / constraint_matrix | `deltap_wannier.cpp`, `deltap_lcao.cpp`, `esolver_ks_lcao.cpp` |
| PW | 平面波 | per_atom only（total 待实现） | `deltap_pw.cpp`, `op_pw_proj.cpp`, `esolver_ks_pw.cpp` |

### 1.2 已修复/已验证功能

| 功能 | 状态 | 关键提交 |
|------|------|---------|
| LCAO Wilson 循环 per-atom γ（raw, branch） | ✅ | `307ac27` |
| LCAO per-band weight normalization | ✅ | `307ac27` |
| LCAO branch selection no-target gate | ✅ | `ba793be` |
| LCAO constraint matrix (C) 读取 | ✅ | — |
| LCAO total 约束模式 | ✅ | — |
| LCAO dp_escon (约束能修正) | ✅ | — |
| LCAO E-field 诊断打印 (λ→E 换算) | ⚠️ 存在但 `Ry→Ha` 因子 2 未核实 |
| PW lambda_init 正确读入 | ✅ | R5→R6 修复 |
| PW OnsiteProjector SMO 生成 | ✅ | 与 LCAO 共用 `projectors::OnsiteProjector` |
| PW per-atom gamma (Wilson 分解) | ✅ | `compute_per_atom_gamma_kstring` |
| PW 力/应力 (DeltaP 贡献) | ✅ | `forces_onsite.cpp`, `stress_onsite.cpp` |

### 1.3 已知问题

| 问题 | 影响范围 | 详情 |
|------|---------|------|
| Branch γ 的 Σ 不守恒 | 所有 per-atom 输出 | `deltap_wannier.cpp:1147` — 每原子独立分支选择, 总和不等于 raw γ |
| PW SMO 覆盖率不足 (~19%) | PW 响应类测试 | SMO 球半径 6 Bohr vs 15 Bohr 盒子, PW 波函数弥散全空间 |
| PW total/constraint_matrix 模式未实现 | PW | 仅支持 per_atom 模式, 所有原子设同一 λ 可模拟 total |
| 负 λ SCF 不稳定 | LCAO total 模式 | 均匀负 λ 收缩电子密度, 默认 Broyden mixing 不足 |
| esolver LCAO 不打印 escon | LCAO | PW 打印, LCAO 仅内部记录 |
| λ↔E Legendre 变换因子 2 | 所有响应计算 | `E_eff_au = -λ * π / a` 缺少 Ry→Ha 的 `/2` |
| `P_abacus` 永为 0 (verify_sum_rule) | LCAO Sum Rule | 声明但从未赋值, 对照功能失效 |

---

## 二、关键参数与设置技巧

### 2.1 必设参数

```
deltap_switch     true    # 总开关
deltap_corr       1       # 开启约束核 (必须为 1, 否则只测量)
deltap_gdir       1/2/3   # 约束/测量方向 (x/y/z)
onsite_radius     6.0     # SMO 投影轨道半径 (Bohr), 建议 ≥6.0
deltap_rm         6.0     # LCAO 邻居搜索半径, 应与 onsite_radius 一致
```

### 2.2 Lambda 控制

```
deltap_lambda_init    0.0           # 初始约束值 (Ry)
deltap_lambda_step    0.0 or 0.01   # 0.0 = 冻结 λ (只测量); 非零 = 梯度下降更新
deltap_lambda_mixing  0.1           # λ 混合因子 (0-1)
deltap_inner_thr      1e-2          # λ 更新触发阈值 (drho < inner_thr)
deltap_inner_nmax     0             # 内层循环最大步数 (0 = 同步梯度下降)
```

### 2.3 约束模式设置

```
# total 模式: 单个 λ 控制 Σγ
deltap_constraint_mode  total
deltap_target_file      target.dat   # target.dat 第一行: 目标 Σγ 值

# per_atom 模式: 每原子独立 λ
deltap_constraint_mode  per_atom
deltap_target_file      target.dat   # target.dat: 每行一个原子的目标 γ

# 约束矩阵模式 (覆盖 constraint_mode):
deltap_constraint_matrix  cmat.dat   # 文件格式: 第一行 "m n", 然后 m 行每行 n 个 C 值 + 1 个 target
```

约束矩阵文件格式 (`cmat.dat`):
```
m  n
C[0][0]  C[0][1]  ...  C[0][n-1]  target[0]
C[1][0]  C[1][1]  ...  C[1][n-1]  target[1]
...
```

### 2.4 盒子与基组红线

| 测试类型 | 最小盒子 | 推荐基组 | 说明 |
|---------|---------|---------|------|
| 平衡偶极 (λ=0) | 12 Å (23 Bohr) | DZP 可, TZDP 推荐 | 偶极对盒子尺寸相对不敏感 |
| 极化率 (λ≠0, efield) | 15 Å (28 Bohr) | TZDP | α 对盒子和基组均敏感 |
| 绝对值跨方法对比 | 18 Å (34 Bohr) | TZDP | 要求基组/盒子均已收敛 |
| 调试/开发 | 10–12 Bohr | DZP | 快速周转, 不用于定量结论 |

### 2.5 Gamma 取值: raw vs branch（关键）

**raw γ (`gamma_I_raw`):**
- 来源: Wilson 循环本征值分解后**直接累加**, 不经任何修改
- 性质: Σ_raw = Wilson 行列式 = 物理 Berry 相位
- **响应测试必须用此值**

**branch γ (`gamma_I`):**
- 来源: raw 值经 zeta 缩放 + 逐原子独立分支选择后累加
- 缺陷: 每原子独立分支搜索 (`deltap_wannier.cpp:1147`)，Σ 不守恒
- 用途: SCF 迭代间追踪收敛连续性（辅助量），**不作为物理量输出**

输出位置:
- 屏幕: `[DeltaP P1/P3]` 行显示 γ (branch)
- 屏幕: `[rawG]` 行显示 `Σγ_raw` 和 `γ_I_raw` per atom
- `OUT.suffix/deltap_results.dat`: 包含 P_I 和电子中心位移
- `OUT.suffix/deltap_branch_enum.dat`: Wilson 循环特征值/权重调试数据
- `OUT.suffix/deltap_zeta_debug.dat`: k-string zeta 标积调试数据

---

## 三、Lambda 操作协议

### 3.1 Frozen-λ 测量模式 (P02/P04/P09)

```bash
deltap_lambda_init     0.0
deltap_lambda_step     0.0      # λ 冻结
deltap_inner_thr       1e-2
deltap_inner_nmax      0
deltap_target_file     留空或无文件   # 不指定 target = 无约束驱动
```

行为: λ 始终为 init 值, gamma 在 drho < inner_thr 时计算一次并打印。

### 3.2 约束收敛模式 (total/per_atom, 有 target)

```bash
deltap_lambda_init     0.0
deltap_lambda_step     0.01     # 非零, 梯度下降
deltap_target_file     target.dat
```

行为: 每 SCF 迭代, λ 更新: `λ_new = mix·(λ + step·(γ - target)) + (1-mix)·λ`.
收敛后 λ 达到使 γ 等于 target 的值。

### 3.3 ±λ 对称扫描 (P01/P05/P08)

关键陷阱: **LCAO total 模式下负 λ 通常 SCF 发散**（密度收缩导致 mixing 不稳定）。

回避策略:

1. **缩小 |λ| 范围**: 用 ±0.002–0.005 Ry 替代 ±0.01 Ry
2. **降低 mixing_beta**: 0.7→0.3–0.4
3. **热启动**: 负 λ 计算从 λ=0 的收敛电荷密度开始
4. **单侧 FD**: 若负 λ 确认物理性失稳, 改用正 λ 单侧 + 能量 FD 公式:
   ```
   α = 2[E(λ₀) - E(λ₀+h) - (线性项)] / h²
   ```
   但优先度低于双侧对称取点。

### 3.4 内层循环 (deltap_inner_nmax > 0, 实验性)

```
deltap_inner_nmax      3–5
```

在内层循环中, λ 通过 becp 重加权梯度下降（Phase D.1）优化。子空间对角化（Phase D.2）尚未实现。

---

## 四、输出解读

### 4.1 屏幕输出关键行

```
[DeltaP-PW] Initialized with N atoms (lambda_init=X)
   ← PW 初始化确认, 检查 lambda_init 是否正确读入

[DeltaP P1] iter=1   γ=(-6.748) Σγ=-6.748 λ=0.0e+00 |γ-t|=2.386e+00
   ← P1: λ 尚未更新 (首次测量), 显示 branch γ 和 |γ-target|

[DeltaP P3] iter=16  γ=(-6.734) Σγ=-6.734 λ=2.06e-06 |γ-t|=1.607e-03
   ← P3: λ 已更新 (约束激活), 目标接近满足

[rawG] Σγ_raw=-6.734419e+00 γ0=-2.284088e+00 γ1=-2.225081e+00 γ2=-2.225250e+00
   ← raw γ (物理值): 总 Σγ_raw 和各原子分量

[DeltaP-PW] drho=5.8e-03 γ_total=-0.1827 rad  λ_avg=0.000e+00 |res|=4.733
   ← PW 输出: γ_total = 标准 Berry 相位 (mod [-π,π]); per-atom γ/atom 另起一行
```

### 4.2 PW vs LCAO 的 γ 值差异

LCAO 的 `Σγ_raw` (~−6 到 −7 rad) 是 Wilson 循环本征值 unwrapped 求和。
PW 的 `γ_total` (~−0.18 rad) 是标准 `Im log det` 的 [−π,π] 区间的 Berry 相位。
两者通过模 2π 关联: `Σγ_raw(LCAO) ≈ γ_total(PW) + 2πN`。

**对于响应 (dγ/dλ), 两者应相等——因为 2πN 在差分中消去。**
若不等, 说明约束耦合效率不同（PW 的 SMO 覆盖率问题）。

---

## 五、逐测试实施说明

### P01 H₂O 极化率三步裁决 (P0 旗舰)

| 步骤 | 内容 | 设置要点 |
|------|------|---------|
| #0 | 核实单位转换 | 检查 `E_eff_au = -λ·π/a` 的 Ry→Ha 因子 |
| #1 | efield FD 参照 | `efield_amp=±0.0005,±0.001`, `dip_cor=1` |
| #2 | DeltaP total ±λ 扫描 | λ=±0.02, ±0.08 Ry (15 Å 盒), raw γ |
| #3 | α 对比 | |α_DeltaP − α_ref|/α_ref ≤ 10% |

**当前状态**: 10 Bohr 盒子中 R1–R8 已做, 最大障碍是负 λ SCF 发散和盒子太小。
**下一步**: 扩盒到 15 Å (28 Bohr), 仍用 LCAO total 模式, 从 λ=0 热启动负 λ 点。

### P02 H₂O 平衡偶极四方对标 (P0)

四条通道:
1. LCAO DeltaP λ=0: 读 Σγ_raw, μ = (a/π)·γ(需核实 spin factor)
2. PW berry_phase=1: 标准 Berry 相位三方向
3. wannier90: MLWF 中心 + 离子项
4. 实验值: 1.855 D

**当前状态**: LCAO 通道有时序数据 (R1-R3 的 λ=0 点), PW berry_phase 输出在当前 binary 中未触发。
**设置**: 15 Å 盒, TZDP, nbands 充分 (≥4×nelec/2 空带)。

### P03 H₂O Born 有效电荷 (P1)

```
Z*_{I,αβ} = ΔF_{I,α} / ΔE_β  (原子 I 在 β 方向的力 对 α 方向场的响应)
```

DeltaP 途径: 用 total 约束 ±λ 扫描, 读力变化。目前 efield 途径 (通过力输出) 更可靠。

### P04 小分子偶极组 (P0)

5 分子: CH₄, CO, NH₃, HF, H₂S。每个分子 LCAO λ=0, 读 raw γ, 转偶极。
CCSD(T) 参考: CH₄ 0, CO 0.122, NH₃ 1.47, HF 1.83, H₂S 0.97 D.

**关键注意**:
- CO 方向: C⁻O⁺ (偶极向量从 C 指向 O)
- CH₄ 需关 symmetry (-1), 否则高对称代码路径可能不同
- 所有分子主轴对齐 z 轴
- 盒 ≥12 Å

### P05 小分子极化率组 (P1)

同 P04 的五分子, total 约束 ±λ 扫描, 测 α_zz。前置: P01 换算链。
CCSD(T) 参考: CH₄ 17.5, CO 13.1, NH₃ 14.6, HF 5.8, H₂S 24.7 a₀³.

### P06 盒尺寸收敛 (P1)

H₂O, L = 12/15/18/21/24 Å, 逐 L 做 (a) λ=0 偶极 和 (b) 能量 FD α。
绘制收敛曲线, 确认"≥15 Å" 红线有数据支撑。

### P07 基组与截断双收敛 (P1)

DZP/SZP/TZDP/QZDP × ecut20/40/60/80 二维扫描 H₂O。产物: 基组/ecut 误差基线表。

### P08 约束线性与对称性 (P0)

λ = 0, ±0.01, ±0.02, ±0.04, ±0.08 Ry (共九点), total 模式。
拟合全线性 R², 子窗口斜率一致性, 失稳边界定位。

**技巧**: 逐点记录 SCF 迭代数作为"近失稳前兆"信号 (迭代数骤增 → 离开线性区)。

### P09 无缓存确定性 B16 (P0)

清理缓存目录, 同一 INPUT 全新运行 3 次。检查:
- raw γ 三位一致 (≤1e-6 rad)
- 能量一致 (≤1e-8 Ha)
- 变线程数 (/1/4/8 OMP_NUM_THREADS) 复测

**算例**: H₂O (Γ 点) + h-BN (k 点网格), LCAO + PW 双实现。

### P10 E–D 曲线: 约束 vs 外场 (P1)

H₂O, λ 约束扫描 vs efield FD 扫描 (同 λ 对应 E 值), 绘 E_phys vs μ 曲线。
两条线应重叠 (同 κ = 极化率倒数)。

### P11 h-BN 介电常数 (P1)

h-BN 体材料 (AB 堆垛), 四原子元胞, k 点 6×6×2 以上。total 约束 ±λ 扫描 z 方向。
ε_zz = 1 + 4π α/V (需真空层 → 超胞法)。

### P12 NaCl/Si Born 有效电荷 (P1)

NaCl (离子晶体, Z*≈±1.1) 和 Si (共价, Z*≈0) — 两个极端体系。
efield ±0.001 a.u. FD 力响应。验证 DeltaP 力输出的 Born 电荷计算。

### P13 BaTiO₃ 自发极化 (P2)

钛酸钡四方相, λ=0 测量 raw γ, μ = (a/π)·Σγ。与实验 Ps ≈ 0.26 C/m² 对标。
需要 Berry 相位参考值作为对照。

### P14 wannier90 交叉验证 (P1)

H₂O, LCAO/PW 密度导出 → wannier90 → MLWF 中心。对比 MLWF 偶极分解和 DeltaP per-atom γ 分解的原子贡献。
**不要求数值一致** (SMO 划分 vs MLWF 划分本质不同), 只要求趋势可解释。

### P15 极化率张量各向异性 (P2)

H₂O 三方向独立扫描, 得完整 α 张量 (α_xx, α_yy, α_zz 及非对角元)。与 CCSD(T) 张量对照。

### P16 PW 约束核覆盖率整改验证 (P1)

扫描 onsite_radius = 6/10/14/20 Bohr, 测各半径下的 γ 响应。绘制响应(r) 曲线。
收敛时与 LCAO 对比, 确认覆盖率≥99% 时 PW 响应恢复。

### P17 场致几何弛豫对照 (P2)

施加 efield, 弛豫 H₂O 几何, 测几何弛豫对偶极的影响。与 DeltaP 约束下的弛豫力对照。

### P18 实空间密度对照 (P2)

λ 约束 vs efield (等效 E) — 两者的 Δρ(r) = ρ(λ) − ρ(λ=0) 应在实空间一致。
输出 `out_chg=1` 做密度差分析。

---

## 六、常见陷阱与错误排查

### 6.1 "DeltaP-PW Initialized with N atoms (lambda_init=0)" 但 lambda_init 非零

→ Bug: PW 初始化用 `ucell.get_dp_target()` 而非 `deltap_lambda_init`。(已修复, 见 R5/R6)

### 6.2 γ 值所有 λ 相同 (PW)

→ 检查 `deltap_lambda_step` 是否为 0。若为 0, λ 冻结; 确认 `deltap_lambda_init` 非零。
→ 检查 `deltap_inner_thr` 是否 < drho 已满足 (SCF 已部分收敛)。

### 6.3 LCAO 负 λ SCF 发散

→ 减小 `mixing_beta` 到 0.3–0.4
→ 减小 |λ| 到 0.002–0.005 Ry
→ 从 λ=0 的 OUT.suffix/ 中复制电荷密度作为热启动
→ 增加 `scf_nmax` 到 50+
→ 若仍失败, 改用正 λ 单侧 + 能量 FD

### 6.4 raw γ vs branch γ 差太大

→ 检查是否使用 raw γ (`[rawG]` 行或 `gamma_I_raw`) 而非 branch γ (`gamma_I`)
→ 正常差值应 ≤ 0.1 rad (branch selection 引起的 per-atom 重分布)

### 6.5 PW γ_total 显示 −0.18 rad 但 LCAO Σγ_raw 显示 −6.7 rad

→ 正常。两者通过 2π 关联。检查 dγ (响应) 而非绝对值。

### 6.6 SCF 收敛后 DeltaP 不打印

→ 检查 `deltap_corr = 1`
→ 检查 `deltap_inner_thr < drho` (阈值可能太大)
→ 检查 `deltap_inner_nmax` 设置

### 6.7 efield + DeltaP 同时使用时配置

```bash
efield_flag    1               # 外场开关
dip_cor_flag   0 or 1          # 偶极修正
efield_amp     0.001           # 场幅度 (a.u.)
deltap_switch  true            # DeltaP 测量开关
deltap_corr    1               # 约束开关
deltap_lambda_step  0.0        # 冻结 λ (仅测量)
```

efield 产生外部扰动, DeltaP 只测量 γ。两者可共存。

---

## 七、优先级矩阵

| 优先级 | 测试 | 阻塞 | 当前阻碍 |
|--------|------|------|---------| 
| P0 当前可做 | P02, P04, P09 | 无 | 仅需标准运行 |
| P0 需 F1/F2 | P01, P08 | 单位备忘录 | λ↔E 因子, γ↔μ spin factor |
| P1 待 P01 | P03, P05, P10, P11, P12, P17, P18 | P01 裁决 | 不能出定量响应结论 |
| P1 无 P01 依赖 | P06, P07, P14, P16 | 各自独立 | P16 需 code change |
| P2 | P13, P15 | P01/P08 | 放量工作 |

---

*最后更新: 2026-07-27, feat/deltap branch*
