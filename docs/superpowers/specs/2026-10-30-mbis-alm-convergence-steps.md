# 2026-10-30 收敛 SCF 步数：MBIS+ALM vs DeltaSpin

## 0. 结果一览

| 用例 | 约束量 | 目标 | SCF 步数 | 结果 |
|---|---|---|---:|---|
| **LiCl（PW, dav_subspace, ecutwfc 50）** | MBIS **电荷** `N_Li` | 2.0 | **85** | CONVERGED（res=6.5e-6，q=2.0000065，μ=0.826，信任半径 1，零拒绝） |
| **BCC-Fe（PW, ecutwfc 20, 2×2×2）** | MBIS **自旋** `M_Fe` | ±3.8 | **>200（未收敛）** | 信任半径塌到 2e-7、20 次拒绝；q[0]=0.075、q[1]=7.92 |
| **BCC-Fe（PW，DeltaSpin 参考）** | onsite **投影** `m_T` | ±3.8 | **13** | CONVERGED（每原子 3.8001，来自 `fe2_bcc_ds`） |

## 1. 结论

- **LiCl 电荷约束**：MBIS+ALM 需 **85 SCF 步**完全收敛（开启信任域）。
- **BCC-Fe 自旋**：MBIS 在 ±3.8 下 **不收敛**，原因有两层：
  1. **bcc 常规胞 MBIS 分区未对称化**：`build_mbis` 目前 `images=0/symmetrize=false`，
     常规胞（原子在角/体心）的分区**破缺**，导致两 Fe 的观测极不对称
     （q=0.075 vs 7.92），约束无法同时满足；
  2. **观测量不同**：DeltaSpin 约束 **onsite 投影** `m_T`（±3.8 可达，13 步）；
     MBIS 是 **空间 AIM 自旋**（Fe2 自然值 ~2.65），±3.8 是投影口径目标，
     对 MBIS 而言可能**不可达**。
- 因此 **MBIS 自旋 85? vs DeltaSpin 13 不是同类对比**：目标/观测量不同，
  且 bcc 需对称化。**LiCl 电荷 85 步**是 MBIS+ALM 的真实收敛成本；
  DeltaSpin 的 13 步是 onsite 投影、可行目标下的成本。

## 2. 公平对比所需（下一步）

1. **bcc 对称化**：给 `build_mbis` 暴露 `images`/`symmetrize`（bcc 用 images=2+symmetrize）；
2. **MBIS-native 目标**：用 MBIS 空间自旋的自然值（~2.65）或其小幅偏移作为目标，
   而非投影口径 ±3.8；
3. 在**同一观测量**下比较步数（MBIS 电荷/自旋 vs 对应口径）。

## 3. 可信的数字

- **MBIS 电荷（LiCl）**：**85 SCF 步**，res 6.5e-6。
- **DeltaSpin 投影（Fe2）**：**13 SCF 步**，每原子 3.8001。
- MBIS 自旋（Fe2）在 ±3.8：**当前不可行**（对称化 + 目标口径未对齐）。

## 4. 复现

```bash
# LiCl 电荷（85 步）：constraint_weight_type mbis, target 2.0,
#   constraint_trust_region 1, alpha_trial 1.0, scf_nmax 200, -np 1
# Fe2 MBIS 自旋（未收敛）：同上但 constraint_type spin, target ±3.8
# DeltaSpin Fe2（13 步）：tests/constraint_s5_parity/fe2_bcc_ds
```
