# 2026-10-30 FeO BRANCH_FLIP 深入分析：images 语义、守卫定义、7 eV 异常

## 1. `constraint_mbis_images` 的 0/1 到底指什么

代码（`mbis_alm.cpp::image_translations`）：
```cpp
const int R = std::max(0, images);
for lx,ly,lz in [-R,R]^3:  T.push_back(lx*c0 + ly*c1 + lz*c2);
```
- `images = k` ⇒ 取 **晶格平移壳层半径 R=k**，共 **`(2k+1)^3`** 个平移：
  - **`images = 0`** → `{0}`（**仅 home cell**），即 **direct** pro `e^{−ζ|x−R_A|}`，**不含原子自身的周期像**。
  - **`images = 1`** → `[-1,1]^3` = **27** 个平移（含最近邻周期像）→ 等价于最小镜像。
  - `images = 2` → 125，依此类推。
- **小胞必须 ≥1**：小胞里原子离边界近，direct pro **漏掉自身周期像**，corner 处严重低估
  → 分区错误（FeO：O 1.4/8.7、Fe 0.4/33.5）。`images=1` 后 O≈7.65、Fe≈14.3 才合理。
- **大盒**（原子远离边界、pro 衰减长度 << L/2）用 `0` 即可且更快（LiCl）。

## 2. `BRANCH_FLIP` 具体指什么

代码（`constraint_loop.cpp:1874-1945`，`on_scf_converged` 内，仅在 `constraint_branch_tol>0` 时启用）：
- 参考相（mu=0）记录 `e_ref_ = scf_energy_`。
- 约束相每次收敛后：`e_guard_ = scf_energy_ + output_energy_correction()`。
- 若 `e_guard_ - e_ref_ < -constraint_branch_tol`（默认 **0.001 Ry**）⇒ 判定
  **"约束相能量比 mu=0 参考低超过容差"**，即 SCF **切到了另一自洽分支/磁态**，
  于是 **fused（BRANCH_FLIP）**，**拒绝把目标当作已验证**。
- 依据代码注释：约束解应是参考态的**连续形变**；能量**骤降**说明换了电子分支（或磁态），
  此时"恰好满足加权观测量"的 Q **无意义**，故熔断。

## 3. 实测数据（FeO，PBE，无 U，LCAO，nspin 2）

| 计算 | 收敛 | FINAL_ETOT (eV) | MBIS Fe0 M | Fe1 M | Fe N | O N |
|---|---|---|---|---|---|---|
| 纯 SCF, mag 2（AFM） | 是 | −7637.04 | +2.30 | −2.81 | 14.3 | 7.67 |
| 纯 SCF, mag 4（AFM） | 是 | −7637.04 | +2.89 | −2.89 | 14.25 | 7.75 |
| 纯 SCF, mag 0（→FM） | 是 | −7636.46 | +2.42 | +2.43 | 14.37 | 7.63 |
| **约束（charge+spin, 小步）** | — | **−7643.93** | **−3.61** | **−3.38** | 14.2 | 7.7 |

- 参考（未约束）Fe0 **charge 14.27、spin 2.85**，与独立 MBIS **14.29/2.85** 一致
  ⇒ **MBIS 权重与观测量正确**（charge+spin 同时生效）。
- **AFM(−7637.04) 与 FM(−7636.46) 仅差 ~0.6 eV**（都收敛）。

## 4. 异常：同 FM 序却差 7 eV

- 约束解是 **FM**（Fe 皆负，−3.6/−3.4），而纯 FM（mag0→FM）是 **+2.4**（同为 FM，整体自旋翻转等价）。
- 但 **约束 FINAL_ETOT = −7643.93** vs **纯 FM = −7636.46**，**相差 ~7.5 eV**。
- 对同一磁性序、相近电荷（Fe 14.2 vs 14.37），**7 eV 的能量差物理上不可能**
  ⇒ **不能用"参考态亚稳"解释**（AFM↔FM 才 0.6 eV）。

**指向：`FINAL_ETOT` 含人为约束势 `∫V_con ρ` 的大项**
- 约束势 `V_con = μ_c w_charge (+ μ_s w_spin, ±)`。Fe 的通道布居 `Q≈14`，
  `μ_c·Q_c ≈ (−0.0447 Ry)·14.3 ≈ −0.64 Ry ≈ −8.7 eV`；LiCl 的 `Q≈2` 时仅 ~0.01 Ry。
- 若 esolver 的 `scf_energy_`（E_band）**已含** `∫V_con ρ`，而 legacy 修正
  `output_energy_correction() = μ·(Q−t)`（`constraint_loop.cpp:2177` 的 legacy 分支）
  **只减去 `μ(Q−t)`、未减 `μ·Q`**，则守卫用的 `e_guard` 被 `≈ μ·Q`（eV 量级）污染。
- 数值上 `−7643.93 + 8.7 ≈ −7635`，接近纯 FM(−7636.5) ⇒ 与"约束密度≈纯 FM"自洽。

## 5. 结论（回答"参考非全局极小" vs "算法问题"）

- **不是"参考态非全局极小"**：无法用物理亚稳解释 7 eV。
- **不是 MBIS 权重/观测量 bug**：参考 Q 与独立 MBIS 一致，权重归一。
- **更像"算法/能量记账问题"（尤其大 Q 约束）**：约束相总能量/守卫含人为 `∫V_con ρ`，
  legacy 的 `E_con=μ(Q−t)` 对**大布居 Q（Fe~14）**修正不足 → **假性 BRANCH_FLIP**；
  对**小 Q（LiCl~2）**可忽略，故此前 LiCl 正常。
- **尚不能 100% 定量分离**：需 (i) **DFT+U FeO**（gapped、唯一基态）验证小约束步只产生
  **meV** 能量变化；(ii) **审计** esolver 的 `FINAL_ETOT` 是否含 `∫V_con ρ`，并让守卫
  使用**物理** `E_KS[ρ_con]`（减掉 `Σ μ_alpha · Q_alpha^{channel}`），而非 `E_band+μ(Q−t)`。

## 6. 建议修复方向（待 DFT+U 验证后实施）
- 分支守卫/报告的约束能量改为 **物理总能量**：`E_phys = E_esolver − Σ_alpha μ_alpha·Q_alpha`
  （charge 用 `Q`，collinear spin 用 `M`；分配模式已有 `−Σμ·t` 约定需统一）。
- 或让 esolver 的 `FINAL_ETOT` 直接输出 `E_KS[ρ_con]`（去掉约束势贡献），再加 `μ(Q−t)`。

## 7. 文件
本日志；验证输入在 `/tmp/opencode/feo_mbis_cs`、`feo_plain_mag{4,0}`（无代码改动）。
