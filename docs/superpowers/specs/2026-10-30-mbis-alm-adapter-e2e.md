# 2026-10-30 MBIS+ALM 薄 adapter 与端到端测试

## 0. 结论

新增**薄、解耦**的宿主接口 `SolverAdapter` + `run_alm_mbis` 驱动；用 mock 求解器
（响应矩阵由真实 MBIS 权重重叠构造）跑通**电荷+自旋同时约束的端到端 ALM**，
2/2 gtest 通过；核心模块（`mbis_alm.cpp`）不改动、不耦合。

## 1. 接口设计

```cpp
class SolverAdapter {
  // 宿主收敛"内层固定乘子问题"：按 mu_eff = mu + k(Q-t) 更新约束势，
  // 返回自洽的每原子 Q=(N,M)。alm_c/alm_s 提供 mu,k,target。
  virtual Observables solve(const MbisState& st, const Alm& alm_c, const Alm& alm_s) = 0;
};
DriverResult run_alm_mbis(grid, atoms, rho0, spin0, alm_c, alm_s, solver, cfg);
```
- **内层**（宿主 SCF）：`V_σ=Σ_A(mu_eff^c_A ± mu_eff^s_A)w_A`，`mu_eff=mu+k(Q−t)`，
  反复到自洽；返回 Q。这是真实 ESolver 的 SCF 该做的事。
- **外层**（驱动）：`mu += k(Q−t)`，检查 `max|Q−t|<tol`。
- 权重 `w_A` 冻结自初始密度（frozen-weight，实用形态）；宿主可在外层重建。

## 2. mock 求解器与结果

- 响应矩阵由**真实 MBIS 权重重叠** `O_AB=∫w_Aw_Bρ` 构造：
  `Q=Q0+J·coef`，`J=Chi·O`（Chi=−gO）。
- 内层**精确自洽解**：`(I−KJ)coef = mu + k(Q0−t)`，再 `Q=Q0+J·coef`。
- 端到端测试（`mbis_alm_adapter_test.cpp`）：
  - `EndToEndAlmConverges`：同时约束 N 与 M，ALM 收敛到目标（<1e-6）。
  - `ChargeAndSpinBothConstrained`：给**非 AFM 自旋目标**，电荷仍精确到自己的目标，
    证明两通道独立可控。
- 全部 gtest：核心 8 + adapter 2 = **10/10 PASS**。

## 3. 两个关键教训（实现层面）

1. **"外层驱动只发一次势"会发散**：旧设计用上一轮 Q 设 `coef`、求解一次，
   等价于未收敛的势迭代。正确做法是**宿主内层把 `mu_eff` 迭代到自洽**
   （真实 SCF 天然如此）；驱动只做外层 `mu` 更新。
2. **`k` 必须满足 `k|J|∈(0,2)`**：外层收缩率 `1/(1+k|J|)`（或 `|1−k|J||`）。
   本 mock 的 `|J|~10³`，而 `Alm::arm` 的**默认 `kmin=0.1`** 会把 `k=1/|J|≈1e-3`
   抬到 0.1，使 `k|J|≫2` → 外层发散。**调用方必须把 `kmin` 设到低于 `1/|J|`**
   （测试用 `arm(chi, 1.0, 1e-10, 1e9)`）。
   → 真实体系（Fe2/LiCl）标定 k 时同样要注意：先测 `χ`，确保 `k|χ|~1`，
   不要把 k 的下限设得过大。

## 4. 产物

- `source/source_estate/module_mbis_alm/mbis_alm_adapter.h/.cpp`
- `source/source_estate/module_mbis_alm/test/mbis_alm_adapter_test.cpp`
- `test/CMakeLists.txt`（`AddTest MODULE_ESTATE_mbis_alm_adapter`）
- 独立编译（无 ABACUS 依赖）：
  ```bash
  g++ -std=c++14 -O2 -I source \
    source/source_estate/module_mbis_alm/mbis_alm.cpp \
    source/source_estate/module_mbis_alm/mbis_alm_adapter.cpp \
    source/source_estate/module_mbis_alm/test/mbis_alm_adapter_test.cpp \
    -lgtest -lgtest_main -pthread -o /tmp/mbis_alm_adapter_test
  ```

## 5. 下一步

- 把 mock 换成**真实 KS 求解**（PW/LCAO）的最小 adapter：先用现成 SCF 接口包一层，
  只让 adapter 负责 `V_σ` 注入与 `observe`；
- 在 Fe2/LiCl 上验证 `kmin` 标定与 ALM 收敛；
- 保持 `mbis_alm` 核心不变，adapter 作为唯一耦合点。
