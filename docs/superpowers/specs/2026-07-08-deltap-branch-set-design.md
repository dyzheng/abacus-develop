# 2π 分支集合方法设计与实现

> **日期**: 2026-07-08
> **分支**: `feat/deltap-wilson-per-atom`
> **目标**: 用分支集合 + 最近邻选择替代失败的 zeta 缩放（Algorithm C），解决 Wilson loop 特征值的 2π 跳变

## 问题

当前 `compute_wannier_polarization` 在 `deltap_wannier.cpp:851-869` 用 zeta 缩放（trace/det 比例）做分支修正，即 Algorithm C。批判性评估已证明此方法失败：trace/det 比例随结构变号（-4.77→+2.90→+4.29），逐原子分配被等比例扭曲。

## 方案

### 数学

Wilson loop 第 n 个特征值 λ_n 的 Berry phase：γ_n = arg(λ_n) + 2πk_n, k_n ∈ Z

逐原子主值：γ^I_0 = Σ_n w^I_n · arg(λ_n)

全部可能解的集合：Γ^I = {γ^I_0 + 2π · w^I · k : k ∈ Z^N_occ}

选择准则：P^I_eff = argmin_{P∈Γ^I} |P - P^I_prev|

### 实现

**新增方法** `select_branch_set()`：
- 输入：主值 γ^I_0、权重 w^I_n、上一步值 P^I_prev
- 搜索策略：先检查 |γ^I_0 - prev| < π（无跳变）；若跳变，搜索单能带位移 (k_n = ±1)；若仍不收敛，搜索双能带
- 复杂度：O(N_occ) 单能带，O(N_occ²) 双能带

**集成**：替换 `deltap_wannier.cpp:851-869` 的 zeta 缩放，改为：
1. 每个 k-string 计算主值 γ^I_0(string) = Σ_n w^I_n(string) · arg(λ_n(string))
2. 构造集合，选最近邻到 prev（第一个 string 用 W_prev_，后续 string 用前一个 string 的选择值）
3. 累积选择值，最终平均

**r_elec 不受影响**：保留 gamma_unwrapped 用于位移计算，不改动。

### 测试

| 体系 | 预期 | 判据 |
|------|------|------|
| H₂O (nocc=4) | 无跳变，主值=选择值 | |P_eff - P_0| < 1e-10 |
| BaTiO₃ ref | 无跳变（参考结构） | ratio ~0.97 |
| BaTiO₃ Ba+0.01 | 已知符号反转，应恢复正确符号 | ratio ~0.97（非 -0.53） |
| BaTiO₃ Ti+0.01 | 可能跳变 | ratio ~0.97 |

用 ref 的 P^I 作为 W_prev_（写入 deltap_branch.dat），运行 disp 结构时加载。
