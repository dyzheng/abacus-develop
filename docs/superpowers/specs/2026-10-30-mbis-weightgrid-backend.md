# 2026-10-30 阶段 2：`WeightGrid` 新增 MBIS 后端（编译集成）

## 0. 结论

在现有 `WeightGrid` 中新增 `WeightType::MBIS` 与 `build_mbis(rho_total)`，
**复用已验证的 `mbis_alm::compute_mbis`**（不经 SCF 改动）；
主库 `abacus_basic_para` **编译/链接成功**。现有 Becke/Hirshfeld/DeltaSpin 路径
完全不变（新增枚举 + 新方法，仅在显式调用 `build_mbis` 时生效）。

## 1. 改动

- `weight_grid.h`：`WeightType` 增 `MBIS`；新增 `void build_mbis(const std::vector<double>& rho_total);`
  与 `bool is_mbis() const`。
- `weight_grid.cpp`：实现 `build_mbis`：
  1. 串行守卫（`rho_basis_->poolnproc==1`；MPI 需先 gather）；
  2. `Grid`：`a_i=latvec_{i,:}*lat0`、`dv_i=a_i/n_i`（复用 `pw_grid_bridge::make_grid`）；
  3. 原子：`R=tau*lat0`、`Zval=ncpp.zv`（`make_atoms`）；
  4. `compute_mbis(rho_total,...)` → 填 `w_[iat][ir]`、`cw_[alpha][ir]`、`maxdev_`。
- `source_estate/CMakeLists.txt`：把 `module_mbis_alm/mbis_alm.cpp` 加入主库 objects
  （模块仍保留独立测试目标）。
- 密度布局：`ir=(ix*ny+iy)*nz+iz`（z 最快），与 cube、`Grid.size()` 一致。

## 2. 验证

- **编译/链接**：`cmake --build build_rel` 成功，`abacus_basic_para` 生成；
  仅既有 `if constexpr` C++17 警告（与本次无关）。
- 组成件均已单独验证：网格转换（`GridBridgeMatchesLattice`，9/9 核心测试；
  LiCl/Fe2 约定核对）、MBIS 数值（cube C++↔Python 逐位一致）。
- **运行期一致性**（LiCl Li 2.0447；Fe2 16/16、±2.65）需阶段 3 把
  `constraint_weight_type mbis` 接进 `constraint_loop` 后跑真实用例。

## 3. 已知限制

- **MPI**：当前要求串行 pool；真实并行需 gather→rank0→scatter，或把
  `compute_mbis` 改为 MPI-aware（pro-density 本地、KL 求和 all-reduce）。
- **小/常规胞对称性**：`cfg.symmetrize=false`（Fe2 bcc 需 true；留待输入开关）。
- **力/应力**：尚未接（`build_derivatives` 对 MBIS 未实现）；后续用模块的
  `constraint_force` 或扩展 `constraint_deriv`。

## 4. 下一步（阶段 3）

- INPUT 增 `constraint_weight_type mbis`；
- `constraint_loop` 在每个外层 μ 步用当前 `chr` 调 `wg.build_mbis(rho_total)`；
- 跑 LiCl/Fe2 自由密度，核对 N/M 与 cube 结果；再做约束目标 + ALM/k 标定。
