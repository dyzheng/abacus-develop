# 2026-08-31 M5：记账 E_con + 审计行

## 1. Test plan
- `EsconKnown`：单约束 μ=0.5、Q=1.9、t=2.0 → E_con=−0.05、max_residual=0.1、
  total_charge=1.9、nelec 透传、maxdev（M1 审计）< 1e-10。
- `EsconMultiComponent`：双约束 μ={0.5,−0.25}、Q={1.9,3.2}、t={2.0,3.0}
  → E_con=−0.10、逐分量残差 {−0.1, +0.2}。
- `ZeroWhenConverged`：Q==t → E_con=0、max_residual=0、total_charge=ΣQ。
- `AuditLineMachineReadable`：审计行 key=value 令牌齐备
  （nconstraint/e_con/max_residual/total_charge/nelec/maxdev + 逐约束
  c[i] q/t/mu/res 行）。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest，`MODULE_ESTATE_constraint_accounting`。
- 输入：合成 mu/Q/target（无 SCF），WeightGrid 仅用于 maxdev 审计值
  （H2O 20 Bohr 盒 20³ 网格）。

## 3. Results
- 4/4 PASS。首跑 2 项失败为 FP 表示差异（0.5×(−0.1)=−0.050000000000000044
  vs 字面量 −0.05），改 EXPECT_NEAR 1e-12 后通过。
- 审计行格式：
  `CONSTRAINT_AUDIT nconstraint=1 e_con=-0.05 max_residual=0.1 total_charge=1.9 nelec=10 maxdev=...`
  + 每约束一行 `CONSTRAINT_AUDIT c[0] q=... t=... mu=... res=...`。

## 4. Analysis
- E_con=Σ_α μ_α(Q_α−t_α) 沿用 deltaspin escon 形式，由外环（Task 8）汇入
  电子总能量（仿 f_en.dp_escon 先例）。
- total_charge=Σ_α Q_α 仅在约束片段划分全部原子时等于 Σ_I N_I（默认
  每原子一约束满足）；V1 sum rule 用默认映射即可直接对拍 nelec。
- 审计行纯 key=value 文本，V1/V2/V3 验证脚本可直接 grep 解析。

## 5. Next steps
- Task 8：外环编排 constraint_loop（建 M1 → 参考态 SCF → 约束 SCF 注入/
  读数 → M4.step → 收敛/熔断）+ esolver_ks_pw 钩子 + E_con 汇入总能量 +
  H2O 冒烟。
