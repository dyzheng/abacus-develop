# 2026-07-30 P 系列算例与工作流构建（A 组：P01/P02/P08/P09）

> 依据：`docs/superpowers/specs/2026-07-30-pseries-test-cases-design.md` ｜ 分工：A 组（P0 旗舰）

## 1. 本轮目标

为 P01/P02/P08/P09 四个测试构建自包含的可执行算例与工作流脚本（`README.md + run.sh + cases/`），不实际运行 ABACUS，仅做 `bash -n` 语法验证与静态审查。

## 2. 测试计划

1. 四个 run.sh 全部 `bash -n` 通过；
2. INPUT 关键字与当前分支 deltap_* 参数表一致；
3. 提取键与既有测试输出样例比对（rawG、E_KohnSham、TOTAL-FORCE、DeltaP-PW 格式）。

## 3. 测试设置

- 目录：`可靠性测试设计集/P0X-*/`，每目录 `README.md / run.sh / cases/`
- H₂O：实验几何（O 盒心 7.9365 Å，H 在 x=7.9365±0.7571、z=+0.5858），30 Bohr 立方盒（LATTICE_CONSTANT 1.8897261254578284 + LATTICE_VECTORS 15.873 Å），O.upf/H.upf + 6au 2s2p1d/2s1p 轨道，Γ 点
- h-BN：六方晶胞 a=2.512/c=6.692 Å，Direct 坐标，B.PD04.PBE.UPF/N_ONCV_PBE-1.0.upf，KPT Gamma 4 4 2

## 4. 结果

| 检查 | 结果 |
|---|---|
| `bash -n` 四个 run.sh | 4/4 PASS |
| rawG 提取格式 vs `tests/deltap_h2o_polarizability/lam_0p000/run.log` | 一致（`Σγ_raw=` 后取值，tail -1） |
| E_KohnSham 提取 vs `running_scf.log` 样例 | 一致（`awk '{print $2}'`，tail -1） |
| TOTAL-FORCE 格式 vs 仓库样例 | 一致（`#TOTAL-FORCE (eV/Angstrom)#` 后 `O1` 行取末列） |
| DeltaP-PW γ_total 格式 vs `pw_total/` 样例 | 一致 |

未运行 ABACUS（按任务要求）。

## 5. 实现中做的假设/偏差

1. **P01 α_ref 符号**：任务给定公式 `2·[E(+δ)+E(−δ)−2E(0)]/δ²`，其符号依赖 dip_cor 能量记账约定（物理上 E(δ)=E0−½αδ² 给出负值）。脚本按字面实现后对 α_ref、α_DeltaP 取绝对值进入判据，README 已注明——属 F1/F2 阻塞的一部分。
2. **P02 PW 通道 ks_solver**：PW 基组下用 `ks_solver cg`（genelpa 仅 LCAO），偏离"其余同"的字面约定，属必要修正。
3. **P02 wannier90 通道**：检测到 wannier90 也只打印提示并 SKIP（后处理需手动对接），不计入 PASS/FAIL 项数。
4. **P02 离子项**：假设代码输出 γ 已含离子点电荷项（README 标注待 P02 实跑确认）。
5. **P08 SCF 迭代数**：以 `running_scf.log` 中 `E_KohnSham` 行数计数（每迭代一行，已由样例确认）。
6. **P09 仅 LCAO 通道**：原设计要求 LCAO 与 PW 双实现；PW 的 γ_total 是 wrapped 量、比较口径需另行约定，本轮只做 LCAO，README 记为已知偏差。
7. **P09 ΔE 判据**：E_KohnSham 单位 Ry，脚本换算 Ha（÷2）后与 1e-8 Ha 比较。

## 6. 文件清单

```
可靠性测试设计集/P01-H2O极化率三步裁决/  README.md run.sh cases/h2o/{STRU,KPT}
可靠性测试设计集/P02-H2O平衡偶极四方对标/ README.md run.sh cases/h2o/{STRU,KPT}
可靠性测试设计集/P08-约束线性与对称性/    README.md run.sh cases/h2o/{STRU,KPT}
可靠性测试设计集/P09-无缓存确定性B16复测/ README.md run.sh cases/h2o/{STRU,KPT} cases/hbn/{STRU,KPT}
```

## 7. 下一步

1. 冒烟运行 P02（LCAO 通道）与 P09（H₂O 单次），确认端到端可跑（设计文档 §6 验证计划第 3 条）；
2. B/C/D 组其余 14 个测试按同一模板构建；
3. P01 需等 F1/F2 备忘录定稿后回头校正换算常量区。
