# 2026-10-30 半真实端到端：真实 cube 上的 MBIS+ALM（自动 k 标定）

## 0. 结论

在**真实 ABACUS 密度 cube**（LiCl、Fe2）上跑通"MBIS 权重 → 由真实权重重叠
构造约束响应 → 自动标定 k → ALM vs Lagrange vs penalty"的端到端流程。
**关键发现：ALM 的优势取决于各约束响应的异质性。**

## 1. 流程

```
cube → MBIS 权重 w_A（真实密度）
     → 响应用真实权重重叠 O_AB=∫w_A w_B ρ，J=-g O·O  （Q=Q0+J·coef）
     → 自动标定 chi_A=J_AA → k=1/|chi_A|（宽 clamp）
     → ALM（内层自洽 (I-KJ)coef=mu+k(Q0-t)，外层 mu+=k(Q-t)）
     → 与全局步 Lagrange、penalty-only 对比
```
工具：`tools/mbis_alm_e2e.cpp`（复用 `tools/mbis_cube_io.h` + `mbis_alm_adapter`）。

## 2. 结果

| 体系 | chi（各约束） | ALM 步数/残差 | Lagrange(全局)/残差 | penalty 偏差 |
|---|---|---|---|---|
| **LiCl**（电荷） | Li **−0.389**，Cl **−6.21**（差 16×） | **25 / 7.7e-9** | **295 / 9.9e-9** | 0.0995 |
| **Fe2**（电荷+自旋） | 两约束均 **≈−20.5**（几乎相同） | 21 / 8.4e-9 | **12 / 6.1e-9** | 0.089 |

- LiCl：Li 响应很软、Cl 很硬；全局步受最硬模限制，软模收缩率≈1 → Lagrange 295 步；
  ALM 按模给 k（Li 0.161、Cl 2.57）→ **25 步（~12×）**。
- Fe2：两约束响应几乎相同，**没有软/硬混杂**；全局步已合适 → Lagrange 12 步，
  ALM（k≈0.049）21 步，反而略慢。
- penalty-only 两种体系都留下 ~0.09–0.10 的偏差（有限 k），不能用。

## 3. 分析（修正之前的过度概括）

"ALM 总是更快"**不成立**。ALM 的收益来自**按模态自适应 k**，因此：
- **响应异质**（软模 + 硬模、近零模）→ ALM 优势巨大（LiCl 12×）；
- **响应同质**（各约束 χ 接近）→ 全局步 Uzawa 已足够，ALM 略慢（Fe2）；
- penalty-only 永远有偏，只能作预条件。
这与之前近零模分析一致：ALM 的价值在**调节约束方向的条件数**，而非绝对加速。

## 4. 自动标定与 kmin 提醒

- 标定用 `chi_A=J_AA`（真实权重重叠导出）→ `k=1/|chi_A|`；本工具用宽 clamp
  `[1e-10,1e9]`。
- **`kmin` 不能设过大**：若 `kmin > 1/|chi|`，`k|J|` 可能 >2 使外层振荡/发散
  （adapter 文档已记录；真实体系标定同理）。
- Fe2/LiCl 的 `chi` 从 -0.39 到 -20.5，标定出 k 从 0.049 到 2.57——**必须逐体系标定**。

## 5. 局限

- 响应是**模型**（线性，耦合来自真实 MBIS 权重重叠），不是完整 KS 响应；
  步数反映真实空间重叠/异质性，但非真实 SCF 成本。
- Fe2 的电荷与自旋用了同一 `J`（简化）；真实中自旋响应另需标定。
- 下一步：把模型 solver 换成**真实 KS 最小 adapter**，在 Fe2/LiCl 上验证上述
  异质性结论与 k 标定。

## 6. 复现

```bash
g++ -std=c++14 -O2 -I source tools/mbis_alm_e2e.cpp \
    source/source_estate/module_mbis_alm/mbis_alm.cpp \
    source/source_estate/module_mbis_alm/mbis_alm_adapter.cpp -o /tmp/mbis_alm_e2e
/tmp/mbis_alm_e2e OUT.*/chg.cube --dc 0.2
/tmp/mbis_alm_e2e OUT.*/chgs1.cube OUT.*/chgs2.cube --images 2 --symmetrize --dc 0.2 --ds 0.2
```
