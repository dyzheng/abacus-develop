# S3 统一分配约束：阶段记录

## 1. 测试计划

- 验证稠密分配算符
  `K_i = W_i - P S^{-1} W_i - W_i S^{-1} P + P S^{-1} W_i S^{-1} P + Q_i`
  的低秩实现与显式 `S^{-1}` 参考逐元素一致。
- 验证权重分割 `sum_i W_i = S` 时 `sum_i K_i = S`，以及零乘子不产生修正。
- 验证 S2 报告与 S3 约束共享同一套 `C'=C S_T^{-1/2}` 目标空间和 DMR 投影。
- 本阶段先确认代数核、共享读数接口和 LCAO 适配器可独立编译；随后才接入
  Hamiltonian 算子、ALM 和冻结力。

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
  `K_i`；尚未注册到 Hamiltonian operator，也尚未接入 `ConstraintLoop`。
  因此本记录只证明 S3 基础层可编译，不构成 S3 闸门验收。

## 5. 下一步

- 新增 `alloc_constraint_lcao_operator`，把 `sum_i mu_i K_i` 注入 LCAO
  Hamiltonian。
- 新增 `constraint_alloc_switch` INPUT，并加入 Gamma-only、共线、
  `basis_type=lcao` 的配置期守卫。
- 在 `ConstraintLoop` 中切换观测到 `m_T`、同步 `mu_eff`，禁止旧权重势
  与 `K_i` 重复注入。
- 完成冻结互补密度力、位置/目标有限差分与 100 步 ALM 刚性验收后，
  再更新本记录的 S3 闸门结果。
