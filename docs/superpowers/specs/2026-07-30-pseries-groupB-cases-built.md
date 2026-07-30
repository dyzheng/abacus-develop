# P 系列算例构建 B 组（P03/P04/P05/P15）

> 日期：2026-07-30 ｜ 分支：feat/deltap ｜ 依据：`2026-07-30-pseries-test-cases-design.md`（B 组分工）

## 1. 测试计划

- 为 P03（H₂O Born 有效电荷）、P04（小分子偶极组）、P05（小分子极化率组）、
  P15（极化率张量各向异性）构建自包含算例目录：README.md / run.sh / cases/。
- 验证：`bash -n` 4/4 通过；awk 拟合/提取/位移生成等核心逻辑用样例数据冒烟；
  提取键与既有测试输出样例比对；不运行 ABACUS。

## 2. 测试设置

- 公共约定按设计文档 §3：ABACUS/PSEUDO_DIR/ORBITAL_DIR/NPROC 环境变量、
  `rm -rf OUT.*`、run.log、提取键（`[rawG] Σγ_raw`、`E_KohnSham`、`TOTAL-FORCE`、
  `[DeltaP-PW] γ_total`）、awk 判据、`SUMMARY: n/n PASS` + exit 0/1。
- 分子几何按任务书实验值（H₂O 30 Bohr 盒，其余 12 Å 盒，盒中心，主轴沿 z）。
- 赝势/轨道已盘点存在：O/H/C/F/S.upf、N_ONCV_PBE-1.0.upf 及对应 2s2p1d（H 2s1p）轨道。
  DZP 层级轨道不存在 → P05 dzp 层级运行时自动 SKIP。

## 3. 结果

- `bash -n`：P03/P04/P05/P15 全部通过。
- awk 冒烟：线性拟合（4 点，slope/R² 正确）、二次拟合 c₂（y=10+2x² → c₂=2, α=−2 正确）、
  TOTAL-FORCE 末块提取（按原子标签取 Fz 正确）、O 原子位移 STRU 生成
  （±0.005 Å，OFMT/CONVFMT=%.7f 保精度）正确。
- 提取键比对：`rawG`/`E_KohnSham`(第 2 列 Ry)/`TOTAL-FORCE`(eV/Å)/`γ_total` 均与
  tests/deltap_h2o_polarizability 真实输出样例匹配。
- dzp 层级轨道检测逻辑验证：占位文件名在 ORBITAL_DIR 中缺失 → 正确判 MISSING。

## 4. 假设与偏差

1. **H₂O 盒边长取 30 Bohr = 15.8753163 Å**（任务书 15.873 Å 为约数；换算系数 a=30 Bohr 需精确）。
2. **P15 DeltaP 窗口加 λ=0 点**（任务书写 ±0.02 两点，但 R²≥0.99 判据需 ≥3 点才有意义）。
3. **PW 通道同时开 `berry_phase 1` + `deltap_switch true`**（满足 PW 基线约定与
   `[DeltaP-PW] γ_total` 提取键；PW 下 deltap_switch 用字符串 true，见 DeltaP测试说明 §三）。
4. **P04 CO 符号判据按任务书"O 端为负"约定实现**（`CO_EXPECT_SIGN=-1`，O 在 +z 期望 μ_z<0）；
   与文献 C⁻O⁺ 方向的冲突在 README §5 说明，翻常量即可切换。
5. **P05/P15 α 换算链**：α_DeltaP = (2a²/π²)·dγ/dλ（由 E=−πλ/(2a)、α=−(a/π)dγ/dE 复合）；
   efield FD α = −c₂（E(Ry) 对 δ(Ha) 二次系数）。系数集中常量区，待 F1/F2 备忘录定稿。
6. **P03 力单位换算** `EVA_TO_RYBOHR=0.0388938`（eV/Å→Ry/Bohr）放常量区并注明待备忘录确认。
7. **P05 组 MAE 用两通道均值**（设计文档未指明通道）；CH₄ 三方向平均后按单分子计 5/5。
8. **PW 模板未设 ks_solver**（genelpa 仅 LCAO；与 A 组 P02 用 cg 不同，取 PW 默认求解器）。

## 5. 下一步

- 冒烟运行 P04（无阻塞，LCAO+PW 五分子）验证端到端；
- F1/F2 备忘录定稿后回填 P03/P05/P15 常量区并解除 WARNING；
- 构建 C/D 组其余测试。
