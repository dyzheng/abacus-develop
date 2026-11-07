# 2026-10-30 探针式 k 标定（`calibrate_alm`）

## 0. 结论

新增 `mbis_alm::calibrate_alm(...)`：用**有限差分探针**测量 `chi_a = dQ_a/dmu_a`
并 arm ALM。它是真实 KS adapter 在进入外层循环前要调的组件，只需宿主在
**固定乘子（k=0）** 下做一次普通 SCF。

## 1. 公式与实现

对每个分量 a（电荷、自旋各自）：
```
mu_a = +delta, 其余 mu=0, k=0  ->  mu_eff = mu
Q_a^probe = solver.solve(st, k=0, mu=delta e_a)
chi_a = (Q_a^probe - Q_a^0) / delta
k_a   = clamp(kappa / |chi_a|, kmin, kmax)
```
然后 `alm_c.arm(chi_c,...)`、`alm_s.arm(chi_s,...)`，进入外层 `mu += k(Q-t)`。

接口：
```cpp
void calibrate_alm(SolverAdapter& solver, const MbisState& st,
                   Alm& alm_c, Alm& alm_s, const Observables& Q0,
                   double delta=1e-3, double kappa=1.0,
                   double kmin=1e-8, double kmax=1e9);
```
**对宿主要求**：只需实现 `solve(st, alm_c, alm_s)`（在 k=0 时即"固定乘子 SCF"）；
这正是真实 ESolver 已有的能力。

## 2. 验证

- **mock（线性响应）**：探针 `chi` 恢复解析 `J_aa`（gtest 断言）；
  `EndToEndAlmConverges` / `ChargeAndSpinBothConstrained` 均通过（2/2）。
- **真实 cube（LiCl/Fe2）**：探针标定的 k 与解析一致
  （LiCl：Li 0.161、Cl 2.57；Fe2：0.0487/0.0487），ALM 仍 25/21 步，
  与之前 `chi=J_AA` 的结果逐位相同。
- 核心 8 + adapter 2 = **10/10 PASS**。

## 3. 意义

- 真实 adapter 无需"知道"解析响应；它只需调用 `calibrate_alm` 并在 k=0 下
  跑一次固定乘子 SCF。
- 结合上一轮"响应异质性"结论：探针测出的 `chi` 谱直接告诉我们要不要用 ALM、
  `k` 取多少、`kmin` 是否安全。

## 4. 复现

```bash
g++ -std=c++14 -O2 -I source tools/mbis_alm_e2e.cpp \
    source/source_estate/module_mbis_alm/mbis_alm.cpp \
    source/source_estate/module_mbis_alm/mbis_alm_adapter.cpp -o /tmp/mbis_alm_e2e
/tmp/mbis_alm_e2e OUT.*/chg.cube --dc 0.2
```

## 5. 下一步

接真实 KS 最小 adapter：实现 `SolverAdapter::solve` = "用 `V_σ=Σmu_eff w_A`
跑固定乘子 SCF，返回 N/M"；`calibrate_alm` 直接可用。`mbis_alm` 核心不变。
