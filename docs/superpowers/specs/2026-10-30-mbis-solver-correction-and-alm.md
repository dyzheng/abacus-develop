# 2026-10-30 纠正与关键发现：外层已是 DIIS 式；DIIS 式快路径是 in-SCF ALM

## 0. 纠正（撤回"饱和/不可达"结论）

- 之前 `2026-10-30-lcao-mbis-saturation.md` 的"LCAO 下 MBIS 电荷饱和、目标 2.0 不可达"
  **不成立**：它基于**未收敛的 SCF 状态**（所有 LCAO 运行都以 `SCF IS NOT CONVERGED`
  结束，q 值是半收敛值），不是真实的 `F(μ)`。
- **2.0 距自然值仅 Δq=0.045（2%）**，是**近目标**，不可能因饱和而不可达。
- 结论：LCAO 的失败是**内层 SCF 在约束势下未收敛**（预算内），不是可达性。

## 1. 外层其实已经是 DIIS 式（每步降一个量级）

PW ecutwfc 50、目标 2.0 的外层残差（实测）：
```
q−t: 0.032 → 0.0086 → 0.00054 → 0.000029
```
每外层步降 **~10–16×**，与 DIIS 的"每步降一个量级"一致。所以**外层收敛本身没问题**。

## 2. 84 步的真实来源：结构开销，不是外层慢

84 = **参考 SCF（1 次）** + **FD 探针 SCF（1 次）** + **3–4 次约束 SCF**，每次 ~10–15 步。
即"每次 SCF 求解 ~10–15 步 × 5–6 次求解"。外层次数不多，**多是"多解了几次 SCF"**。

## 3. 真正的 DIIS 式快路径：`constraint_mu_schedule alm`（in-SCF ALM）

代码事实（`constraint_loop.cpp:315,690`）：
```
alm_mode_ = (cfg_.mu_schedule == "alm");
```
- `alm` 调度启用 **in-SCF ALM**：
  1. 一次 probe SCF 测 χ → `alm_arm(k≈1/χ)`；
  2. `alm_penalty_live_=true` → 罚项 `(k/2)(Q−t)²` 与 `mu_eff=mu+k(Q−t)`
     **在每个 SCF 迭代内更新**（`finish_observation`）。
- 于是**约束与密度在同一个 SCF 内一起收敛**（类似把约束残差并入 DIIS 向量），
  预期 **~1 次 SCF（~15–30 步）**，而不是"参考+外层多解"。
- **我们之前的 84 步是因为用了默认调度（非 `alm`）**，走了旧的 secant 外层。

## 4. 为什么之前没走 ALM

- `alm_armed_` 为 false（无 `ALM armed` 行）→ 旧的 `outer_step`/secant 路径。
- 原因是 `constraint_mu_schedule` 未设为 `alm`（默认非 alm）。
- 设 `constraint_mu_schedule alm` 即进入 in-SCF ALM。

## 5. 下一步（关键验证）

```
INPUT: constraint_mu_schedule alm ; constraint_weight_type mbis ;
       constraint true ; target (absolute) 2.0 ; scf_nmax 100 ; -np 1
```
预期：**probe(≈一次 SCF) + in-SCF ALM(≈一次 SCF)** → 总 ~30–50 步、DIIS 式，
远少于 84。若确实如此，则"85 步太多"的真因是**没启用 ALM 调度**，而非算法本身。

## 6. 附：若 in-SCF ALM 仍慢

- 检查 `alm_probe_pending_` 是否触发（`[constraint] ALM armed` / probe 行）；
- 确认 `constraint_alm_k`（k override）未误设；
- 内层仍可用 `scf_thr 1e-6`（inexact）进一步压步数；
- `constraint_trust_region` 对 in-SCF ALM 非必需（可关）。

## 7. 结论一句话

**外层已 DIIS 式；84 步来自"多解了几次 SCF"。代码里已有 in-SCF ALM
（`constraint_mu_schedule alm`）把约束并入 SCF 迭代、一次收敛——这才是对
"能否 DIIS 式每步降一个量级"的答案。下一步就是启用它并计数。**
