# DeltaP 极化矢量计算 — gdir 问题与三方向方案

> 日期: 2026-07-13  
> 问题: 当前实现只计算了单方向极化。是否可以同时计算 P_x, P_y, P_z？

---

## 一、现状：数据有，但只用了一个方向

### 1.1 Berry Connection A_nk — 已计算三方向

`compute_berry_connection` (`deltap_berry.cpp:141`)：

```cpp
for (int alpha = 0; alpha < 3; alpha++)    // x, y, z 全部计算
{
    A_nk_[iat][ik][n][alpha] = term1 + term2;
}
```

**每个 k 点的 Berry connection 矢量 A_nk(alpha=0,1,2) 都已就绪。**

### 1.2 但下游只取了一个分量

| 函数 | 使用方式 | 位置 |
|------|---------|------|
| `compute_wannier_polarization` | `gamma_I[iat][alpha_idx]` 只填 `alpha_idx = gdir_-1` | `deltap_wannier.cpp:1158` |
| `integrate_polarization` | `A_nk[alpha_idx]` 只积分 `gdir_-1` | `deltap_berry.cpp:237` |
| `compute_resta_z` | `G_cart[alpha_idx]` 只在 gdir 方向非零 | `deltap_wannier.cpp:1502` |

`alpha_idx = gdir_ - 1` 单方向 → P_x, P_y 永远是 0。

---

## 二、为什么不能直接扩展 Wilson Loop 到三方向？

Wilson Loop 方法计算的是**沿 k-string 方向的 Berry phase 积分**。每条 string 给出一个标量 γ，全部 string 平均后得到该方向的极化：

```
P_alpha ∝ Σ_strings γ_string
```

γ 只沿 string 方向 (gdir)。要得到 P_x, P_y, P_z，需要分别在 x, y, z 方向各跑一次 Wilson Loop。**不能从 z-string 的 γ 推出 P_x 或 P_y。**

---

## 三、可行方案

### 方案 A：三次独立 Wilson Loop（当前可行，需改代码）

同一组 k 点 + 波函数，切换 `gdir_` 三次调用：

```
gdir=1 → setup_kstring(沿x) → compute_wannier_polarization → P_x
gdir=2 → setup_kstring(沿y) → compute_wannier_polarization → P_y
gdir=3 → setup_kstring(沿z) → compute_wannier_polarization → P_z
```

**改动**：在 `compute_wannier_polarization` 外层加 `for alpha=0..2`，每次重设 gdir_、重调 setup_kstring、但复用同一组 psi_k 和 S_k/D_I。

**注意**：Berry connection 的 HK 修正也依赖于 gamma，需要同时改为三方向。

### 方案 B：A_nk 直接积分（数据已有，需改代码）

既然 `A_nk_[iat][ik][n][alpha]` 已存储三方向分量，直接对所有 k 点求和得到极化矢量：

```
for alpha = 0..2:
    P_alpha ∝ Σ_ik Σ_n f_n × A_nk[alpha]
```

**改动**：修改 `integrate_polarization`，将 `alpha_idx` 替换为 `for alpha=0..2` 循环。

**优点**：不需要 3 次 Wilson Loop，一次积分出三方向。
**局限**：A_nk 只在 k-string 的 k 点上计算（非全文 BZ），且沿 string 方向的是 ∂/∂k，垂直方向用的是 ∂S/∂k_alpha 有限差分，精度可能不如 Wilson Loop。

### 方案 C：Wilson Loop + A_nk 混合

用 Wilson Loop 计算 string 方向（高精度），用 A_nk 积分计算垂直方向（已有数据）。两种方法在 string 方向上应等价。

---

## 四、当前 gdir 的硬编码改点

如果选择方案 A，需要修改以下位置（`alpha_idx = gdir_ - 1` 出现处）：

| 文件 | 行号 | 用途 |
|------|------|------|
| `deltap_wannier.cpp` | 351 | Wilson loop prefactor 方向 |
| `deltap_wannier.cpp` | 1158 | results_.gamma_I 填入 |
| `deltap_wannier.cpp` | 1494 | Resta-Z 方向 |
| `deltap_berry.cpp` | 207 | integrate_polarization 方向 |
| `deltap.cpp` | 144-170 | setup_kstring 方向 |

---

## 五、建议

1. **优先方案 A**：Wilson Loop 三次，最可靠。先实现并测试 P_x=P_y=P_z 的旋转等价性，验证算法在三方向的正确性。

2. **验证后方案 B**：用 A_nk 积分替代重复 Wilson Loop，提升效率。需要与方案 A 交叉验证一致性。

3. **gdir 参数最终应废弃**：目标是无须指定方向，自动输出 (P_x, P_y, P_z) 矢量。

下一步：实施方案 A，验证 → 实施方案 B，对比 → 废弃 gdir 参数。
