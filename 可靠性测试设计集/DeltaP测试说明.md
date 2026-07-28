# DeltaP P 系列测试辅助说明

> 本文是对 `可靠性测试设计集/P01–P18` 各测试文档的 DeltaP 操作补充——只写**怎么跑 DeltaP、怎么看结果**，不重复测试设计文档中已有的目的、原理、判据。

---

## 一、DeltaP 操作速查

### 1.1 开关与模式

```
deltap_switch  true      # 总开关（必须）
deltap_corr    1         # 约束核开关。1=计算 γ 并允许 λ 更新，0=不计算
deltap_gdir    3         # 方向: 1=x, 2=y, 3=z
```

两种常用配置：

| 用途 | deltap_lambda_step | deltap_target_file | 效果 |
|------|-------------------|-------------------|------|
| 读平衡 γ | 0.0 | 不设 | λ 冻结在 lambda_init，SCF 收敛后输出一次 γ |
| 驱动 γ 到 target | 0.01 | target.dat | λ 自动更新，最终 γ 趋近 target |

### 1.2 约束模式

```
deltap_constraint_mode  total       # 单个 λ 控制全部原子的 Σγ
deltap_constraint_mode  per_atom    # 每个原子独立 λ
```

total 模式搭配 target.dat（第一行一个目标 Σγ 值）。
per_atom 模式搭配 target.dat（每行一个原子的目标 γ）。

约束矩阵模式（`deltap_constraint_matrix cmat.dat`）可覆盖 constraint_mode，文件格式:
```
m  n                          # m 个约束, n 个原子
C[0][0] ... C[0][n-1] t[0]   # m 行, 每行 n 个矩阵元素 + 1 个 target
```

### 1.3 Lambda 控制参数

```
deltap_lambda_init     0.0    # 初始 λ (Ry)
deltap_lambda_step     0.01   # 梯度下降步长。0=冻结
deltap_lambda_mixing   0.1    # λ 更新混合因子 (0–1)
deltap_inner_nmax      0      # 内层循环步数。0=同步单步梯度下降
deltap_inner_thr       1e-2   # λ 更新触发条件: drho < inner_thr
```

### 1.4 投影半径

```
onsite_radius  6.0            # SMO 投影轨道半径 (Bohr)
deltap_rm      6.0            # LCAO 邻居搜索半径，与 onsite_radius 一致
```

---

## 二、输出解读

### 2.1 屏幕输出

运行后在 stdout 中搜索以下关键行：

```
[rawG] Σγ_raw=-6.734419e+00 γ0=-2.284088e+00 γ1=-2.225081e+00 γ2=-2.225250e+00
```
raw γ：Wilson 行列式直接分解，不经任何修改。**取 Σγ_raw 做偶极/极化率换算。**

```
[DeltaP P1] iter=1   γ=(-6.748) Σγ=-6.748 λ=0.0e+00 |γ-t|=2.386e+00
[DeltaP P3] iter=16  γ=(-6.734) Σγ=-6.734 λ=2.06e-06 |γ-t|=1.607e-03
```
branch γ（经分支选择处理，总和与 raw γ 可能有 ~0.1 rad 量级差异）。P1 = λ 尚未更新，P3 = λ 已更新。

PW 路径输出:
```
[DeltaP-PW] drho=5.8e-03 γ_total=-0.1827 rad  λ_avg=0.000e+00 |res|=4.733
```
`γ_total` 是标准 Berry 相位 (mod [−π,π])。和 LCAO 的 Σγ_raw 差 2πN 量级，**但差值（dγ）应一致**。

### 2.2 文件输出

| 文件 | 用途 |
|------|------|
| `OUT.suffix/deltap_results.dat` | 每原子 P_I (极化密度)，电子中心位移 |
| `OUT.suffix/deltap_branch_enum.dat` | Wilson 本征值、SMO 权重（调试分支问题用） |
| `OUT.suffix/deltap_zeta_debug.dat` | k-string 标积（调试 Wilson 循环用） |

---

## 三、常见问题处理

### SCF 不收敛

- 降低 `mixing_beta`（默认 0.7 → 0.3–0.4）
- 增大 `scf_nmax`
- 用已收敛计算的电荷密度热启动（复制 `OUT.suffix/` 中 charge 文件）
- LCAO total 模式下，负 λ 值常比正 λ 更难收敛——从 λ=0 热启动、减小 |λ|

### PW 路径 λ_init 没生效

检查 stdout 中初始化行: `[DeltaP-PW] Initialized with N atoms (lambda_init=X)`。
若 X=0 而 INPUT 设了非零值，说明 STRU 中有 `dp_target` 关键字覆盖了 `deltap_lambda_init`。

### DeltaP 不输出 γ

- 确认 `deltap_corr = 1`
- 确认 `deltap_inner_thr > 0` 且 SCF 已收敛到 drho < inner_thr
- PW: 确认 `deltap_switch true`（需要字符串 true，不是 1）

### PW vs LCAO 的 γ 值差异

LCAO 的 `Σγ_raw` 和 PW 的 `γ_total` 绝对值差 2πN 量级是正常的（前者 unwrapped，后者 wrapped）。比较时用**差值**（dγ/dλ 或 Δγ/ΔE），不用绝对值。

---

## 四、逐测试 DeltaP 操作要点

以下仅补充各测试设计文档中未涵盖的 DeltaP 操作细节。测试的目的、原理、判据见对应设计文档。

### P02/P04 — 平衡偶极

**DeltaP 配置**: `lambda_step 0.0`, 无 target_file, 测量模式。

跑完后取 `[rawG] Σγ_raw`。偶极换算:
```
μ = (a/π) × Σγ_raw      (a = 盒子沿 gdir 方向的边长, Bohr)
```
若分子不沿 gdir 方向对齐，需 gdir=1,2,3 各跑一次合成矢量。

PW 路径用 `[DeltaP-PW] γ_total` 代替 Σγ_raw，公式相同。

---

### P08 — 约束线性与对称性

**DeltaP 配置**: `lambda_step 0.0`（固定 λ）, total 模式, 无 target_file。

对每个 λ 值独立跑一次 SCF，收集 Σγ_raw。注意:
- 从 |λ| 较小开始（如 ±0.01 Ry），若 SCF 发散再缩小。
- 负 λ 发散时用 §三 中热启动方案。
- 每个 λ 值记 SCF 迭代数——迭代数骤增是靠近失稳边界的信号。

---

### P01/P05 — 极化率

**阻塞**: F1 (λ↔E 换算因子) 和 F2 (γ↔μ spin 因子) 两项待核实。

**DeltaP 配置**: `lambda_step 0.0`, total 模式, ±λ 扫描（具体 λ 值由 P08 确定的有效窗口给出）。

**efield 参照侧**: `efield_flag 1, dip_cor_flag 1, efield_amp ±0.0005/±0.001, efield_dir = gdir`。能量取 `OUT.suffix/running_scf.log` 中 `E_KohnSham`，FD 公式:
```
α = [E(+δ) + E(−δ) − 2E(0)] / δ²
```
注意 E_KohnSham 单位为 Ry，efield_amp 单位为 Hartree，换算需除以 2。

---

### P09 — 无缓存确定性

每次运行前 `rm -rf OUT.autotest`。分别用 `OMP_NUM_THREADS=1/4/8` 各测一次。对比 `[rawG] Σγ_raw` 和 `E_KohnSham`。

---

### P06/P07 — 盒尺寸/基组收敛

**偶极部分** (当前可做): 同 P02 的测量配置，逐盒子/基组运行，取 Σγ_raw。

**极化率部分** (阻塞于 P01): 待换算链就绪后，每盒子/基组做 efield FD + DeltaP ±λ 扫描双通道。

---

### P10 — E–D 曲线: 约束 vs 外场

**DeltaP 侧**: 在 P08 确定的线性窗口内取 4–6 个 λ 值。计算 E_phys = E_KS − Σλ·γ（减去约束能）。绘 E_phys vs μ。

**efield 侧**: 相同的 E 场值（由 F1 从 λ 值换算），绘 E_phys vs μ。

两曲线比较曲率（= 1/α）。

---

### P03/P12 — Born 有效电荷

DeltaP total 约束 ±λ 扫描，取各 λ 下的原子力输出。`Z* = ΔF / ΔE`（需 F1 换算 λ→E）。

力在 `OUT.suffix/running_scf.log` 中 `TOTAL-FORCE` 行。

---

### P11 — h-BN 介电常数

**DeltaP 配置**: total 模式, `deltap_gdir 3`（沿 c 轴）, k 点网格 ≥6×6×2。±λ 扫描。`ε = 1 + 4πα/V_cell`。

---

### P13 — BaTiO₃ 自发极化

**DeltaP 配置**: λ=0 测量模式, 三方向各跑一次，取 Σγ_raw。换算同 P02。

---

### P14 — wannier90 交叉验证

跑完 LCAO 或 PW 后，导出电荷密度用于 wannier90 后处理。DeltaP 的 per-atom γ 分解（`[rawG]` 行中各原子分量）与 MLWF 中心分解对比——两者划分方案不同，预期趋势一致但数值不对齐。

---

### P15 — 极化率张量

对 gdir=1,2,3 三个方向分别做 P01 流程，得到 α_xx, α_yy, α_zz。

---

### P16 — PW 约束核整改验证

扫描 `onsite_radius = 6/10/14/20 Bohr`，测各半径下 PW 的 dγ/dλ。与同设置 LCAO 的 dγ/dλ 对比。预期大半径时两者趋近。

---

### P17/P18 — 场致弛豫/实空间密度

P17: efield 或 DeltaP 约束下开启 `cal_force 1` + `relax`。P18: `out_chg 1`, 对比 `λ≠0` 和 `λ=0` 两轮 SCF 的电荷密度差。

---

## 五、命令模板

### 采集 raw γ
```bash
abacus > run.log 2>&1
grep "rawG.*Σγ_raw" run.log
```

### ±λ 批量扫描
```bash
for lam in -0.01 -0.005 0.0 0.005 0.01; do
    d="lam_$(echo $lam | sed 's/-/m/;s/\./p/')"
    mkdir -p $d && cp STRU KPT $d/
    sed "s/LAMBDA_INIT/$lam/" INPUT.tmpl > $d/INPUT
    (cd $d && abacus > run.log 2>&1)
done
grep "rawG.*Σγ_raw" lam_*/run.log
```

### efield 有限差分
```bash
for eamp in -0.001 -0.0005 0.0 0.0005 0.001; do
    d="ef_$(echo $eamp | sed 's/-/m/;s/\./p/')"
    mkdir -p $d && cp STRU KPT $d/
    sed "s/EFIELD_AMP/$eamp/" INPUT.tmpl > $d/INPUT
    (cd $d && abacus > run.log 2>&1)
done
grep "E_KohnSham" ef_*/OUT.autotest/running_scf.log
```

### 收敛巡检
```bash
grep -l "SCF IS NOT CONVERGED" */run.log    # 列出未收敛的目录
```
