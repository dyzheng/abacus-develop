# P04　小分子偶极组四方对标（CH₄/CO/NH₃/HF/H₂S）

> 系列：DeltaP 可靠性测试 ｜ 前置：**无阻塞** ｜ 设计文档：`../P04-小分子偶极组.md`

## 1. 目的

把单分子 H₂O 的对标扩展到 5 个覆盖不同极性的小分子（含 CH₄=0 零点检验、
CO=0.122 D 小偶极符号检验），验证极化模块在多点化学空间的平衡测量可靠性，
对标 GSCDB138/Dip146 的 CCSD(T) 参考。

## 2. 算例清单

| 分子 | 几何 | 盒 | CCSD(T) μ (D) |
|---|---|---|---|
| CH₄ | r_CH=1.087 Å 正四面体 | 12 Å | 0 |
| CO | r=1.128 Å，O 在 +z | 12 Å | 0.122 |
| NH₃ | r=1.012 Å，C3 轴沿 z | 12 Å | 1.47 |
| HF | r=0.917 Å，F 在 +z | 12 Å | 1.83 |
| H₂S | r=1.336 Å，C2 轴沿 z | 12 Å | 0.97 |

每分子两通道：LCAO λ=0 测量（`[rawG] Σγ_raw`，gdir=3）与
PW berry_phase（`[DeltaP-PW] γ_total`，gdir=3）。
换算：μ(D) = (a/π)·γ×2.541746，a = 12 Å = 22.6767 Bohr。

## 3. 目录结构与用法

```
cases/<mol>/{STRU,KPT}   五分子实验几何（主轴沿 z，盒中心）
cases/INPUT_lcao.tmpl    LCAO DeltaP 测量模板（λ=0 冻结）
cases/INPUT_pw.tmpl      PW berry_phase 模板（berry_phase 1 + DeltaP-PW 测量）
run.sh                   主工作流；结果汇总 runs/results.txt
```

```bash
bash run.sh                 # 或 NPROC=4 bash run.sh
```

## 4. 判据

| 项 | 判据 |
|---|---|
| 逐分子 LCAO−PW 互差 | ≤ 0.02 D（5 项） |
| 对 CCSD(T) 组 MAE（LCAO 通道，\|μ\|） | ≤ 0.03 D |
| CH₄ 零点 | \|μ\| ≤ 0.01 D（两通道各自） |
| CO 符号 | μ_z 符合"O 端为负"约定，方向相反打印 FAIL |

末尾 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

## 5. CO 符号约定与判定逻辑

- 几何：C 在盒中心，O 在 +z 方向（z_O = z_C + 1.128 Å）。
- 偶极矢量取物理约定 μ = Σ q_i·r_i（负端 → 正端），总偶极含离子项与电子项。
- 本测试判定逻辑：O 端为负 ⇒ 期望 μ_z < 0（脚本常量 `CO_EXPECT_SIGN=-1`）。
- 注意：文献中 CO 基态极小偶极方向为 C⁻O⁺（负端在 C），与本约定相反。
  本检查的目的是验证**代码输出符号约定的一致性**而非裁决物理方向；
  若备忘录/物理认定采用 C⁻O⁺，仅需将 `CO_EXPECT_SIGN` 翻转为 +1。

## 6. 风险与注意（抄自设计文档 §7）

- CH₄ 需关闭对称性（symmetry 0），避免代码对高对称分子的分支处理差异；
- CO 小偶极对 SCF 阈值敏感（scf_thr 1e-8）；
- PW 的 γ_total 与 LCAO 的 Σγ_raw 绝对值差 2πN 量级属正常（wrapped vs unwrapped），
  本组分子偶极均远小于极化量子，直接比较有效。
