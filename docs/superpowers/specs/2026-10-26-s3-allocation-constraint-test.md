# S3 统一分配约束：阶段记录

## 1. 测试计划

- 验证稠密分配算符
  `K_i = W_i - P S^{-1} W_i - W_i S^{-1} P + P S^{-1} W_i S^{-1} P + Q_i`
  的低秩实现与显式 `S^{-1}` 参考逐元素一致。
- 验证权重分割 `sum_i W_i = S` 时 `sum_i K_i = S`，以及零乘子不产生修正。
- 验证 S2 报告与 S3 约束共享同一套 `C'=C S_T^{-1/2}` 目标空间和 DMR 投影。
- 本阶段先确认代数核、共享读数接口和 LCAO 适配器可独立编译；随后才接入
  Hamiltonian 算子、ALM 和冻结力。
- 明确观测口径：S3 约束量是完整分配读数
  `Tr[(D_up ± D_dn) K_i]`，不是只读阶段的 `n_T/m_T`。

## 2. 测试设置

- 系统：Linux，单进程 `abacus_basic_para` 构建。
- 构建：`cmake --build build -j 32 --target abacus_basic_para`。
- 单元测试：
  - `MODULE_ESTATE_constraint_alloc_constraint`
  - `MODULE_ESTATE_constraint_alloc_lcao`
- 稠密代数测试使用固定种子的随机对称正定 `S`、随机对称 `W`，并与显式
  `S^{-1}` 公式逐元素比较；判据 `1e-10`。

## 3. 结果

- `abacus_basic_para` 构建与链接：PASS。
- `MODULE_ESTATE_constraint_alloc_constraint`：1/1 PASS（内部 3 个 gtest）。
  - 低秩核与显式 `S^{-1}` 公式一致：PASS。
  - `sum_i K_i = S`：PASS（`1e-10`）。
  - 零乘子修正为零：PASS。
- `MODULE_ESTATE_constraint_alloc_lcao`：1/1 PASS。
- `git diff --check`：PASS。

## 4. 分析

- 当前实现利用 `P S^{-1} = S B B^T`，只构造 `S B` 与 `B^T W` 的
  `norb x R` 小矩阵，不显式形成 `S^{-1}`，满足设计中的低秩组装要求。
- S2 报告已改为调用共享的生产目标空间与 DMR 投影函数，避免 S2 读数与
  S3 约束出现两套口径。
- LCAO 适配器已能由 Gamma-only `S(R)`、权重网格和 Hamiltonian 布局预计算
  `K_i`。Hamiltonian operator、`ConstraintLoop` 观测和冻结力的后续集成见
  `2026-10-26-s3-allocation-integration-test.md`；本记录只覆盖基础层。

## 5. 后续集成结果

- `alloc_constraint_lcao_operator` 已加入 Gamma-only Hamiltonian 主链，
  `constraint_alloc_switch` 已接入配置期守卫和 ALM。
- `ConstraintLoop::observe_allocation()` 读取完整分配量
  `Tr(DMR K_i)`；旧的 `n_T/m_T` 只保留为诊断量。
- 冻结力使用互补密度 `Q_c D Q_c`；位置 FD、目标 FD、两帧 ALM 和
  `Σm=M_tot` 均已通过，原始数据见
  `2026-10-26-s3-allocation-integration-test.md`。
