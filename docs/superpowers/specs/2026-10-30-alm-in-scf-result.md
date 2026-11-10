# 2026-10-30 in-SCF ALM（constraint_mu_schedule alm）实测：罚项过刚 → dampen 塌缩

## 0. 结果

PW ecutwfc 50、目标 2.0、`constraint_mu_schedule alm`、scf_nmax 100：
- `ALM armed: k = 1.66667 Ry/e`（已 arm）；
- 但反复出现 `ALM dampen: drho growing three iterations (penalty too stiff) -> k halved`，
  **k 被连砍 7 次**（1.67 → ~0.013）；
- 100 步未收敛，q≈2.021，μ 每步仅 +0.002（几乎不动）。

## 1. 根因

- **罚项太硬**：`alm_penalty_live_` 的 `(k/2)(Q−t)²` 在 SCF 内注入，k=1.67 使电荷
  每次迭代振荡（drho 连续三步增长）；
- **dampen 守卫**（设计用于防罚项过刚）把 k 反复减半 → k 塌缩 → 约束作用消失 → 慢。
- 这与之前的"ALM 曲率 k 需 `k|χ|~1`"一致：本体系 SCF 对 k 很敏感，大的罚项曲率
  直接引起 charge sloshing。

## 2. 与 `outer` 调度对比

| 调度 | 结果 |
|---|---|
| `outer`（默认，Lagrange+secant/Newton） | **84 步收敛**（q−t: 0.032→0.0086→0.0005→3e-5，外层 DIIS 式） |
| `alm`（in-SCF penalty） | 100 步未收敛（罚项过刚 → dampen 塌缩） |

→ **对本 MBIS 电荷约束，in-SCF ALM（罚项）不如 outer**：罚项曲率使 SCF 振荡。

## 3. 结论与建议

- **不要用 in-SCF ALM（penalty）做该电荷约束**：罚项过刚、dampen 塌缩。
- **用 outer Newton + 非精确内层（`scf_thr 1e-6`）**：外层每步降一个量级（已证），
  内层放宽后每次 SCF ~10–15 步 → 总 = 参考 + 探针 + 2–3 次约束 ≈ **40–60 步**。
- 若要 **≤2 次求解**：去掉小 h 探针，用"大 h 试探（=一次约束 SCF）测 χ + Newton"；
  或在 outer 上把首次步长设为 `r0/χ_guess`（χ_guess≈0.05）。
- 真正的"DIIS 式总成本"上限受制于**每次 SCF ~10–15 步**；2 次求解 ≈ 20–30 步是理论下界。

## 4. 文件/状态

- 试验（`alm` 调度、FD 探针）均未改代码（仅 INPUT 实验）；分支保持已提交状态。
- 本文记录 `alm` 负结果，供下一步选型。
