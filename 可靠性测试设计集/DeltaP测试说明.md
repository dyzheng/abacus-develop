# DeltaP P 系列测试说明

> 对应: `可靠性测试设计集/P01–P18`
> 本文只写测试执行者需要的内容——每个测试的设置、工作流、判据、陷阱。不包含代码实现细节或 bug 清单。

---

## 一、测试前必读

### 两种运行模式

| 模式 | 用途 | 关键参数 |
|------|------|---------|
| **测量模式** | 读 γ 值（偶极等平衡量） | `deltap_lambda_step 0.0`, 无 target_file |
| **约束模式** | 施加 λ 扰动（极化率等响应量） | `deltap_lambda_step 0.01`, 有 target_file 或 λ 主动更新 |

测量模式只计算一次 γ（在 drho < deltap_inner_thr 时触发），不修改 λ。约束模式每 SCF 步更新 λ 以逼近 target。

### 哪个 γ 值是对的

- **响应用 `raw γ`**: 屏幕 `[rawG]` 行，或 `gamma_I_raw`。其总和 = Wilson 行列式 = 物理 Berry 相位。
- **`branch γ`** (`gamma_I`, 屏幕 `[DeltaP P1/P3]` 行): 每原子独立分支选择后累加，总和不严格守恒。仅用作 SCF 迭代追踪辅助量，**不做定量分析**。

### 当前已知局限（影响测试设计）

1. **负 λ SCF 不稳定** (LCAO total 模式): 均匀负 λ 使密度收缩，混迭参数默认值下常不收敛。回避: 减小 |λ|、降低 mixing_beta、或从 λ=0 热启动。
2. **PW SMO 覆盖率低**: PW 下 SMO 投影球半径（onsite_radius=6 Bohr）在 15 Bohr 盒子中仅覆盖 ~19% 体积。PW 的 per-atom λ 约束对 Berry 相位的耦合效率比 LCAO 差约 6–10×。PW 平衡测量 (λ=0) 不受影响。
3. **标准 Berry 相位输出 (`berry_phase 1`)** 在 SCF 计算中不产出（需 nscf 模式或单独编译）。PW 体系的 Berry 相位对照目前只能从 DeltaP 的 `γ_total` (PW 路径) 获取。
4. **λ↔E 换算因子** 的 Ry↔Ha 转换尚未在代码级核实（esolver 中的 `E_eff_au` 打印可能缺 `/2`）。在核实完成前，跨方法能量对标以 efield 的能量 FD 作为裁判。

### 参数红线

| 测试类型 | 最小盒子 | 推荐基组 |
|---------|---------|---------|
| 平衡偶极 | 12 Å | DZP |
| 极化率/响应 | 15 Å | TZDP |
| 跨方法绝对值对比 | 18 Å | TZDP |

---

## 二、可立即执行的测试

### P02: H₂O 平衡偶极 (LCAO)

**目的**: 验证 λ=0 测量模式下 raw γ 换算的偶极是否正确。

**INPUT 关键参数**:
```
calculation       scf
basis_type        lcao
gamma_only        0
nspin             1
scf_thr           1e-06
smearing_method   gauss
smearing_sigma    0.01
mixing_type       broyden
mixing_beta       0.4
ks_solver         genelpa
symmetry          -1
deltap_switch     true
deltap_corr       1
deltap_gdir       3
deltap_lambda_init 0.0
deltap_lambda_step  0.0
deltap_inner_thr    1e-2
onsite_radius      6.0
deltap_rm          6.0
```

**STRU**: H₂O 实验几何 (rOH=0.9572 Å, ∠=104.52°), C2 轴沿 z。盒 ≥12 Å。

**工作流**: 单次 SCF → 读取 `[rawG] Σγ_raw` → 偶极 μ_z = (a/π)·Σγ_raw (a 为盒子 z 边长 Bohr, Σγ_raw 为 rad)。注: 此公式中的 spin 因子需用已知偶极校准。

**判据**: 所得偶极 ≈ 0.73 e·Bohr ≈ 1.85 D (PBE 下略高是正常系统差)。

**陷阱**: 若分子不沿 z 对齐, 需要三方向 (gdir=1,2,3) 各跑一次取矢量合成。

---

### P04: 小分子偶极组 (LCAO)

**目的**: 校验 P02 在 5 个不同极性分子的推广性。

**体系**: CH₄, CO, NH₃, HF, H₂S——实验几何, 主轴沿 z。盒 ≥12 Å。设置同 P02。

**工作流**: 逐分子单次 SCF → raw γ → 偶极。汇总 MAE, 符号检查 (CO 方向为 C⁻O⁺)。

**判据**: 对 CCSD(T) MAE ≤0.03 D; CH₄ |μ|≤0.01 D; CO 方向正确。

**陷阱**:
- CH₄ 须关对称性 (`symmetry -1`) 避免代码分支差异。
- CO 偶极小 (~0.12 D), 对 SCF 收敛敏感, scf_thr 降至 1e-8。

---

### P08: 约束线性与 ±λ 对称性 (LCAO)

**目的**: 确定 total 约束的线性响应窗口, 找失稳边界。

**设置**: H₂O, 15 Å 盒, TZDP, total 模式。λ = 0, ±0.01, ±0.02, ±0.04, ±0.08 Ry (九点)。step=0 (固定 λ)。

**工作流**:
1. 九点逐点 SCF, 读 raw γ。
2. 绘 Σγ_raw vs λ, 拟合全线性 R²。
3. 检查 ±λ 反对称: |γ(+λ) + γ(−λ) − 2γ(0)| / |γ(+λ) − γ(0)| < 5%。
4. 分窗口 (±0.02 内 vs 全窗口) 斜率对比 ≤3%。

**判据**: R²≥0.98, 反对称 ≤5%, 子窗口一致。

**陷阱**: 负 λ 大概率发散 (见 §一)。策略:
1. 先用 λ=±0.01 测试收敛性。
2. 负 λ 点从 λ=0 的 OUT.autotest/ 复制电荷密度文件热启动。
3. 降低 mixing_beta 到 0.3。
4. 若 ±0.01 仍不收敛, 缩小到 ±0.005。
5. 记录失稳边界 λ_crit, 后续测试只用 |λ| < λ_crit 窗口。

---

### P09: 无缓存确定性

**目的**: 确保同一输入三次独立运行结果逐位一致。

**工作流**:
1. H₂O, 12 Å 盒, LCAO, 同 P02 设置。
2. `rm -rf OUT.autotest`, 运行, 记录 raw γ。重复 3 次。
3. 对比 raw γ 差 ≤1e-6 rad, 总能量差 ≤1e-8 Ha。
4. 改变 `OMP_NUM_THREADS=1/4/8` 各再跑一次。

**陷阱**: 必须清理缓存, 不能用之前 OUT 目录。

---

### P06: 盒尺寸收敛 (偶极部分)

**目的**: 确定偶极的收敛盒尺寸。

**设置**: H₂O, L=12/15/18/21/24 Å 五点, DZP, λ=0 测量模式。

**工作流**: 逐 L 运行 → raw γ → 偶极 → 绘制 μ(L)。取 L=24 Å 为平台参照。

**判据**: 15 Å 处 |Δμ|/μ ≤3% (相对 24 Å)。

---

### P07: 基组收敛 (偶极部分)

**目的**: 确定基组对偶极的系统误差。

**设置**: H₂O, 15 Å 盒, DZP/SZP/TZDP 三点, λ=0。

**工作流**: 逐基组运行 → raw γ → 偶极 → 列表对比。TZDP 为基组收敛参照。

---

## 三、待前置条件满足后方可执行的测试

### P01: H₂O 极化率三步裁决

**阻塞**: 单位备忘录 (F1 λ↔E 因子核实, F2 γ↔μ spin 因子核实)。

**工作流**:
1. efield FD 裁判: `efield_flag 1, dip_cor_flag 1, efield_amp ±0.0005/±0.001` → E_KS → α_ref = −ΔE/δE²。
2. DeltaP total ±λ 扫描 (λ=±0.02, ±0.08 Ry) → raw γ → dγ/dλ。
3. 经 F1/F2 换算得 α_DeltaP, 对比 α_ref。

**判据**: |α_DeltaP − α_ref| / α_ref ≤ 10%。

**陷阱**: 负 λ 使用 §二 P08 的热启动协议。若负 λ 确认物理性不通, 改单侧 FD。

---

### P03/P05/P10–P12/P17/P18

**阻塞**: P01 裁决通过。在此之前, DeltaP 不能输出定量响应结论。

---

### P13–P15

**阻塞**: P01 + P08。需要较大的计算资源（体材料 k 点、大超胞）。P16 待 PW 代码整改。

---

## 四、常用命令模板

### LCAO 单次测量 (P02/P04)

```bash
# 准备 INPUT (见 §二 P02), STRU (几何对齐 z), KPT (Γ 1×1×1)
abacus > out.log 2>&1
grep "rawG.*Σγ_raw" out.log   # 取 raw γ
```

### LCAO ±λ 扫描 (P08)

```bash
for lam in -0.01 -0.005 0.0 0.005 0.01; do
    dir="lam_${lam}"
    mkdir -p $dir && cp STRU KPT $dir/
    sed "s/LAMBDA_INIT/$lam/" INPUT.template > $dir/INPUT
    (cd $dir && abacus > run.log 2>&1)
done
# 汇总: grep "rawG.*Σγ_raw" lam_*/run.log
```

### efield 有限差分 (P01 裁判侧)

```bash
for eamp in -0.001 -0.0005 0.0 0.0005 0.001; do
    dir="ef_${eamp}"
    mkdir -p $dir && cp STRU KPT $dir/
    sed "s/EFIELD_AMP/$eamp/" INPUT.template > $dir/INPUT
    (cd $dir && abacus > run.log 2>&1)
done
# 取能量: grep "E_KohnSham" ef_*/OUT.autotest/running_scf.log
# α = [E(+δ) + E(−δ) − 2E(0)] / δ²  (注意 Ry↔Ha 转换)
```

### SCF 收敛检查

```bash
grep "SCF IS NOT CONVERGED\|achieved" */run.log
grep "ETOT/eV" */run.log | tail -1   # 最终迭代总能量
grep "ITER.*DRHO" */run.log | tail -3 # 最后几轮密度残差
```

### 确定性复测 (P09)

```bash
for i in 1 2 3; do
    rm -rf OUT.autotest
    abacus > run_${i}.log 2>&1
    grep "rawG.*Σγ_raw" run_${i}.log
done
```

---

## 五、输出文件快速参考

| 文件 | 内容 |
|------|------|
| `OUT.suffix/running_scf.log` | SCF 迭代详情 + E_KohnSham |
| 屏幕 stdout | `[rawG]` raw γ, `[DeltaP P1/P3]` branch γ |
| `OUT.suffix/deltap_results.dat` | P_I (极化密度), r_elec (电子中心) |
| `OUT.suffix/deltap_branch_enum.dat` | Wilson 特征值 + SMO 权重 (调试用) |
| `OUT.suffix/deltap_zeta_debug.dat` | k-string zeta 标积 (调试用) |
