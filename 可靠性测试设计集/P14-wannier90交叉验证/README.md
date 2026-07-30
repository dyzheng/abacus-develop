# P14　wannier90 交叉验证（可执行算例）

> 系列：DeltaP ｜ 前置：无（**无阻塞**）｜ 设计文档：`../P14-wannier90交叉验证.md`

## 1. 目的

wannier90 的 MLWF 中心（WC）是偶极/极化的独立实空间表示。DeltaP 的 per-atom γ_I 分解
与 WC 分解是两种"局域化"语言——不可能逐项相同（划分方案不同），但**总偶极必须一致**。
本测试同时验证测量层（λ=0）与分解层的物理合理性。

## 2. 目录结构

```
P14-wannier90交叉验证/
  README.md
  run.sh
  cases/
    STRU_H2O          # 15.873 Å 盒，O(c,c,c) H(c±0.7571,c,c+0.5858)，c=7.9365
    STRU_NH3          # 12 Å 盒，N(c,c,c) H 三枚（z=c−0.3818），c=6.0
    STRU_CH4          # 12 Å 盒，正四面体 H=C+0.6276·(±1,±1,±1)
    STRU_Si           # 金刚石 fcc 初基胞（a=5.431 Å）
    KPT_GAMMA         # 分子 Γ 点
    KPT_SI            # Si 4×4×4
    INPUT_lcao.tmpl   # LCAO λ=0 测量模式
    INPUT_pw.tmpl     # PW SCF（w90 密度源，out_chg 1）
    INPUT_nscf.tmpl   # nscf + towannier90 1 + nnkpfile（w90 接口）
  runs/               # 运行产物（含 results.txt）
```

## 3. 用法

```bash
bash run.sh                  # 或 NPROC=4 bash run.sh
```

wannier90 通道自动检测：`command -v wannier90`（或 `wannier90.x`）无则整体 SKIP，
不影响 LCAO 部分判定。

## 4. 工作流

**LCAO 部分（四体系）**：λ=0 测量模式（`lambda_step 0.0`，无 target），读 `[rawG]` 行
总 Σγ_raw 与各原子 γ 分量。分子总偶极 μ_z=(a/π)·Σγ_raw（a 为盒 z 边长，Bohr），
×2.541746 得 Debye；Si 打印总极化（模 2π 量子）。

**w90 部分（分子三体系，PW 密度源）**：
1. PW SCF（`out_chg 1`）出电荷密度；
2. 生成 `seed.win`（num_wann=4，sp3 投影，Γ 点无需 disentanglement 窗口，
   仅占用带）→ `wannier90 -pp seed` 出 `seed.nnkp`；
3. ABACUS `calculation nscf` + `towannier90 1` + `nnkpfile seed.nnkp`
   + `init_chg file` 出 seed.mmn/amn/eig；
4. `wannier90 seed` 收敛 MLWF → `seed_centres.xyz`；
5. WC 偶极 μ_z = −2·Σz_WC + Σ_I Z_I·z_I（每 MLWF 双占据 −2e；Z_I 取赝势价电子数
   O6/H1/N5/C4，常量区）。

## 5. 判据表

| 编号 | 判据 | 状态 |
|---|---|---|
| J1 | 四体系 LCAO SCF 收敛且 [rawG] 齐全 | 有效（硬判） |
| J_H2O/J_NH3/J_CH4 | 总偶极 LCAO vs w90 差 ≤0.02 D | 硬判，**仅 w90 有产出时计入**，否则 SKIP |
| — | H2O μ∈[1.5,2.2] D、NH3 μ∈[1.2,1.9] D、CH4 \|μ\|≤0.05 D | 软判（参考区间，越界 WARNING） |
| — | per-atom γ 分解 vs WC 划分趋势对照 | 仅打印对照表，**不做硬判**（划分方案不同） |
| — | WC 位置 vs 化学预期（H₂O 两孤对+两键，差 ≤0.05 Å） | 打印 seed_centres.xyz 人工核对 |

末尾 `SUMMARY: n/n PASS`，退出码 0/1；结果存 `runs/results.txt`。

## 6. 关键约定

- **不逐值对齐**：DeltaP per-atom γ_I 按 SMO 投影轨道划分，WC 按 MLWF 中心划分，
  二者只在总量上可比；脚本只打印对照表（设计文档 §1/§5）。
- **w90 窗口设置**：分子 Γ 点、价带 4 条 MLWF，sp3 初始投影固定（保证可复现，
  设计文档 §7）；不启用 disentanglement（无半占据带）。首次实跑若 MLWF 不收敛，
  在 run.sh 的 seed.win 生成段加 dis_win/dis_froz。
- **Si 的 w90 通道**：固体 k 点 wannier 接口的 mmn 收集与窗口设置需专项确认，
  本轮 Si 只做 LCAO 部分（w90 循环仅含三个分子）。
- **μ 符号约定**：γ→μ 的符号依赖 F2 备忘录；若 LCAO 与 w90 偶极系统性反号，
  先核查符号约定再改判据。

## 7. 风险与注意（抄自设计文档 §7）

- MLWF 对初始投影敏感，固定初始猜值（sp3）保证可复现；
- 分子体系用 Γ 点 wannier90 设置（已落实）。

补充（工程层面）：

- w90 管线为首次实跑前的脚手架：`towannier90` 产物（mmn/amn/eig）的输出目录
  随版本可能不同，脚本用 `find` 通配收集；若收集失败该体系 SKIP 并保留日志；
- `seed_centres.xyz` 解析取第 4 列为 z（格式 `X x y z`），首次实跑请人工核对；
- 离子项 Z_I 取赝势价电子数（O6 N5 C4 H1），若赝势含半芯态需改常量区 `IONZD_*`。
