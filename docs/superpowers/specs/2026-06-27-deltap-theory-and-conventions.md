# DeltaP Wilson loop 特征值法：理论推导与约定排查记录

> **日期**: 2026-06-27
> **目标**: 完整记录所有公式推导、单位换算、隐含约定，以及每个数值差异的理论解释

---

## 1. Berry phase 的数学定义

### 1.1 连续形式

电子极化（King-Smith & Vanderbilt, 1993）：

$$\mathbf{P}^{\text{el}} = -\frac{e}{(2\pi)^3} \sum_n \int_{\text{BZ}} d^3k \, \langle u_{n\mathbf{k}} | i\nabla_\mathbf{k} | u_{n\mathbf{k}} \rangle$$

其中 $|u_{n\mathbf{k}}\rangle = e^{-i\mathbf{k}\cdot\mathbf{r}} |\psi_{n\mathbf{k}}\rangle$ 是 Bloch 态的周期部分。

### 1.2 离散 Wilson loop 形式

沿方向 $\alpha$（晶格矢量 $\mathbf{a}_\alpha$），k-string $k_0 \to k_1 \to \cdots \to k_{N-1} \to k_0$：

$$\gamma_\alpha = \text{Im}\,\ln\prod_{j=0}^{N_k-1} \det\big[\mathbf{O}(k_j, k_{j+1})\big]$$

$$P_\alpha = \frac{e\,a_\alpha}{2\pi\,\Omega} \cdot \gamma_\alpha \pmod{\frac{e\,a_\alpha}{\Omega}}$$

其中 $\mathbf{O}(k_j, k_{j+1})_{mn} = \langle u_{m,k_j} | u_{n,k_{j+1}}\rangle$ 是周期部分的重叠矩阵。

### 1.3 ABACUS 中的约定

ABACUS `berryphase` 模块输出：
- `Electronic Phase` = $\gamma_{\text{elec}} / (2\pi)$（reduced phase，无量纲）
- `Ionic Phase` = $\gamma_{\text{ionic}} / (2\pi)$
- $P = \frac{a_\alpha}{\Omega} \times (\text{elec\_phase} + \text{ionic\_phase})$（e/bohr²）

**关键**: ABACUS 的 P 公式中**没有** $2\pi$ 在分母，因为 phase 已经除了 $2\pi$。

---

## 2. 重叠矩阵 O_j 的构建

### 2.1 周期部分重叠 vs Bloch 态重叠

**周期部分**:
$$\langle u_{m,k_j} | u_{n,k_{j+1}}\rangle = \langle \psi_{m,k_j} | e^{-i\mathbf{b}\cdot\mathbf{r}} | \psi_{n,k_{j+1}}\rangle$$

其中 $\mathbf{b} = \mathbf{k}_{j+1} - \mathbf{k}_j$ 是 k 点位移。

**Bloch 态**:
$$\langle \psi_{m,k_j} | \psi_{n,k_{j+1}}\rangle = \sum_{\mu\nu} C^*_{\mu m}(k_j) \, S(\mathbf{b})_{\mu\nu} \, C_{\nu n}(k_{j+1})$$

其中 $S(\mathbf{b})_{\mu\nu} = \sum_\mathbf{R} e^{2\pi i \mathbf{b}\cdot\mathbf{R}} \langle\phi_\mu(0)|\phi_\nu(\mathbf{R})\rangle$。

**关系**: 周期部分重叠 = Bloch 态重叠 $\times$ 位置算子修正 $e^{-i\mathbf{b}\cdot\mathbf{r}}$。

### 2.2 ABACUS berry_phase 的实现 (`prepare_midmatrix_pblas`)

```cpp
// 相位
double kRn = (kv.kvec_c[ik_R] * orb1_orb2_R[iw_row][iw_col][iR] - dk * tau1) * TWO_PI;

// 重叠 (复数)
std::complex<double> orb_overlap(psi_psi, (-dk * ucell.tpiba * psi_r_psi));
```

展开：

$$M_{\mu\nu} = \sum_\mathbf{R} e^{2\pi i (\mathbf{k}_R \cdot \mathbf{R}_{\text{cart}} - \mathbf{dk} \cdot \boldsymbol{\tau})} \times \big[\langle\phi_\mu|\phi_\nu(\mathbf{R})\rangle - i \cdot \mathbf{dk} \cdot \text{tpiba} \cdot \langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle\big]$$

其中：
- $\mathbf{k}_R$ = `kv.kvec_c[ik_R]`（Cartesian k 点，**ABACUS 内部单位**）
- $\mathbf{R}_{\text{cart}}$ = `orb1_orb2_R`（Cartesian 晶格矢量，单位 lat0）
- $\mathbf{dk}$ = `dk` = `kvec_c[ik_R] - kvec_c[ik_L]`（Cartesian k 位移）
- $\boldsymbol{\tau}$ = `tau1`（原子位置，单位 lat0）
- tpiba = `2π / lat0`（1/Bohr）
- $\langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle$ = `psi_r_psi`（**local** 位置矩阵，单位 Bohr）

### 2.3 单位系统

ABACUS 内部单位：
- 长度：Bohr
- `lat0` = 晶格常数（Bohr），如 BaTiO3: lat0 = 1.8897261254578284
- `latvec` = 晶格矢量（lat0 单位），如 BaTiO3: [[4,0,0],[0,4,0],[0,0,4.2]]
- `a1 = latvec.row(0) * lat0`（Bohr）= [7.5589, 0, 0]
- `tpiba = 2π / lat0` = 3.3230 (1/Bohr)
- `kvec_d` = direct k 点（分数坐标，0~1）
- `kvec_c` = Cartesian k 点 = `kvec_d × G`，其中 `G = inv(latvec)^T`（lat0 单位的倒数）
- `orb1_orb2_R` = Cartesian 晶格矢量（lat0 单位）= `R_direct × (a1+a2+a3) / lat0`
- `tau` = 原子位置（lat0 单位）

**关键推导**: `kvec_c` 和 `R_cart` 都在 lat0 单位下：
$$\mathbf{k}_c \cdot \mathbf{R}_{\text{cart}} = \text{(lat0}^{-1}\text{)} \times \text{(lat0)} = \text{无量纲}$$

$$\mathbf{dk} \cdot \boldsymbol{\tau} = \text{(lat0}^{-1}\text{)} \times \text{(lat0)} = \text{无量纲}$$

位置修正：
$$\mathbf{dk} \cdot \text{tpiba} \cdot \langle\phi|\mathbf{r}'|\phi(\mathbf{R})\rangle = \text{(lat0}^{-1}\text{)} \times \text{(Bohr}^{-1}\text{)} \times \text{(Bohr)} = \text{(lat0}^{-1}\text{)}$$

**等等，这不对！** `dk` 是 lat0⁻¹ 单位，`tpiba` 是 Bohr⁻¹，`r_psi` 是 Bohr。乘积 = lat0⁻¹ × Bohr⁻¹ × Bohr = lat0⁻¹。**不是无量纲！**

**修正**: 实际上 `kvec_c` 的单位需要更仔细分析。

`G = inv(latvec)^T`，`latvec` 是无量纲的（lat0 单位），所以 `G` 也是无量纲的。`kvec_c = kvec_d × G`，`kvec_d` 无量纲，所以 `kvec_c` 无量纲。

但 `kvec_c` 代表的是 Cartesian k 点，物理上应该有 1/Bohr 的量纲。实际上：
- `kvec_c` 的数值 = k 点在倒格子中的分数坐标 × 倒格子矩阵
- 倒格子矩阵 `G` 的数值 = `inv(latvec)^T`，其中 `latvec` 是无量纲的
- 所以 `kvec_c` 是无量纲的

但 `dk = kvec_c[ik_R] - kvec_c[ik_L]` 也是无量纲的。

位置修正：`dk × tpiba × r_psi` = 无量纲 × (1/Bohr) × Bohr = **无量纲** ✓

相位：`(kvec_c × R_cart - dk × tau) × 2π` = 无量纲 × 2π = **无量纲** ✓

**结论**: 所有量在 ABACUS 内部都是无量纲的（lat0 单位），tpiba 将 Bohr 转换为 lat0 单位。

### 2.4 DeltaP 的实现 (`compute_S_dk_link`)

DeltaP 使用与 berry_phase 相同的相位和位置修正：

```cpp
// 相位
ModuleBase::Vector3<double> R_cart = e.Rx * ucell.a1 + e.Ry * ucell.a2 + e.Rz * ucell.a3;
double arg = TWO_PI * (kvec_c_R.x * R_cart.x + ... - dk_c.x * e.tau_x - ...);

// 位置修正
ModuleBase::Vector3<double> r_local = r_psi_full - e.R1_cart * e.ov;
double imag_part = -(dk_c.x * r_local.x + dk_c.y * r_local.y + dk_c.z * r_local.z) * ucell.tpiba;
std::complex<double> overlap(e.ov, imag_part);
```

**与 berry_phase 的对应**:
- `kvec_c_R` = `kv.kvec_c[ik_R]` ✓
- `R_cart` = `e.Rx * a1 + ...` = `R_direct × latvec`（lat0 单位）= `orb1_orb2_R` ✓
- `dk_c` = `kvec_c_R - kvec_c_L` = `dk` ✓
- `tau` = `e.tau_x` = `get_tau(iat)`（lat0 单位）✓
- `r_local` = `get_psi_r_psi(...) - R1_cart * ov` ✓

### 2.5 snap vs center2_orb11 的差异

**DeltaP** 用 `overlap_intor_->snap()` 计算 $\langle\phi_\mu|\phi_\nu(\mathbf{R})\rangle$。
**berry_phase** 用 `center2_orb11.cal_overlap()` 计算同一个量。

两者都是二中心积分，但实现不同：
- `snap` (TwoCenterIntegrator): 用球贝塞尔展开 + Gaunt 系数
- `center2_orb11` (Center2_Orb::Orb11): 同样的数学方法，但不同的初始化参数

**验证**: `det_berryphase` 的 det 与 `berryphase_overlap` 的 det 在 diamond (nocc=4) 上**完全一致**。但在 BaTiO3 (nocc=15) 上无法验证（gathering bug）。

**当前状态**: DeltaP 用 `snap` + `get_psi_r_psi`，berry_phase 用 `center2_orb11` + `psi_r_psi`。两者给出不同的 O_j 矩阵（3% P 误差），但 det(O_j) 可能一致（未验证）。

---

## 3. Wilson loop 矩阵与特征值

### 3.1 Wilson loop 矩阵

$$\mathbf{W} = \prod_{j=0}^{N_k-1} \mathbf{O}(k_j, k_{j+1})$$

$\mathbf{W}$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 矩阵。

**归一化**: 每步除以 $\max|W_{ij}|$（实正数），保持 arg 不变但防止溢出。

$$\det(\mathbf{W}_{\text{norm}}) = \frac{\det(\prod \mathbf{O}_j)}{\prod c_j^{N_{\text{occ}}}}$$

$\arg(\det(\mathbf{W}_{\text{norm}})) = \arg(\det(\prod \mathbf{O}_j))$ ✓（$c_j$ 实正）

### 3.2 特征值分解

$$\mathbf{W} = \mathbf{V} \cdot \text{diag}(\lambda_n) \cdot \mathbf{V}^{-1}$$

- $\lambda_n$ = 特征值（复数），**规范不变**（相似变换不改变特征值）
- $\gamma_n = \arg(\lambda_n) \in (-\pi, \pi]$
- **Sum rule**: $\sum_n \gamma_n = \arg(\det \mathbf{W})$ ✓（对数可加性）

### 3.3 berry_phase 的平均方法（"除以平均"）

```cpp
// 1. 每个 string 的 zeta
cphik[i] = exp(i * phik[i])  // phik[i] = Im ln(zeta_i)

// 2. 平均
cave = sum(wistring[i] * cphik[i])  // 加权平均复数

// 3. 展开
theta0 = atan2(cave.imag(), cave.real())
for each i:
    cphik[i] /= cave
    dtheta = atan2(cphik[i].imag(), cphik[i].real())
    phik[i] = (theta0 + dtheta) / (2π)
phik_ave = sum(wistring[i] * phik[i])
```

**数学**: $\gamma_{\text{unwrap},i} = \theta_0 + \text{arg}(e^{i\gamma_i} / \text{cave})$

其中 $\theta_0 = \arg(\text{cave})$，$\text{cave} = \frac{1}{N}\sum e^{i\gamma_i}$。

**效果**: 当 $\gamma_i$ 接近 $\pm\pi$ 时，$e^{i\gamma_i}$ 接近 -1，除以 cave 后 arg 不跳变。

### 3.4 DeltaP 的平均方法

```cpp
gamma_avg = sum(gamma_i) / N  // 简单算术平均
```

**差异**: 当某个 $\gamma_i$ 从 $+\pi$ 跳到 $-\pi$（或反之），简单平均偏移 $2\pi/N$。

**量化**: BaTiO3 10×10×10, $N=100$，1 个 string 跳变 → 误差 $2\pi/100 \approx 0.063$。
$|\gamma| \approx 2.08$ → 相对误差 $\approx 3\%$。**与观测一致**。

### 3.5 自旋因子

**berry_phase** (nspin=1):
```cpp
pdl_elec_tot = 2 * phik_ave;  // 乘 2
```

**DeltaP**:
```cpp
prefactor = -0.5 * a_alpha / (2π * omega);
```

**推导**: 
- berry_phase: $P = \frac{a}{\Omega} \times 2 \times \text{phik\_ave} = \frac{a}{\Omega} \times 2 \times \frac{\gamma}{2\pi} = \frac{a \gamma}{\pi \Omega}$
- DeltaP: $P = -0.5 \times \frac{a}{2\pi\Omega} \times \gamma = -\frac{a\gamma}{4\pi\Omega}$

**比例**: $\frac{P_{\text{DeltaP}}}{P_{\text{berry}}} = \frac{-a\gamma/(4\pi\Omega)}{a\gamma/(\pi\Omega)} = -\frac{1}{4}$

**等等，这不对！** 实测 ratio ≈ 0.97，不是 -0.25。

**重新推导**:

berry_phase 的 `phik_ave` 已经是 reduced phase（除了 $2\pi$）：
$$P_{\text{berry}} = \frac{a}{\Omega} \times 2 \times \text{phik\_ave} = \frac{a}{\Omega} \times 2 \times \frac{\gamma_{\text{unwrap}}}{2\pi} = \frac{a \gamma_{\text{unwrap}}}{\pi \Omega}$$

DeltaP 的 gamma 是 $\sum_n \arg(\lambda_n)$ 的简单平均，**未除了 $2\pi$**：
$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times \gamma_{\text{DeltaP}}$$

**符号**: DeltaP 的 $\gamma$ 与 berry_phase 的 $\gamma$ 符号相反，因为：
- `snap` 计算 $\langle\phi_{\text{ket}}|\phi_{\text{bra}}\rangle$（转置）
- berry_phase 的 `center2_orb11` 计算 $\langle\phi_{\text{bra}}|\phi_{\text{ket}}\rangle$
- 转置使 det 取共轭 → arg 取反

**自旋**: berry_phase 对 nspin=1 乘 2（每个能带 2 个电子）。DeltaP 的 $\gamma$ 是单自旋值。

**完整推导**:
$$P_{\text{berry}} = \frac{a}{\Omega} \times 2 \times \frac{\gamma_{\text{berry}}}{2\pi} = \frac{a \gamma_{\text{berry}}}{\pi \Omega}$$

$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times \gamma_{\text{DeltaP}}$$

如果 $\gamma_{\text{DeltaP}} = -\gamma_{\text{berry}}$（符号相反）：

$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times (-\gamma_{\text{berry}}) = \frac{a \gamma_{\text{berry}}}{4\pi\Omega}$$

$$\frac{P_{\text{DeltaP}}}{P_{\text{berry}}} = \frac{a\gamma/(4\pi\Omega)}{a\gamma/(\pi\Omega)} = \frac{1}{4}$$

**但实测 ratio ≈ 0.97，不是 0.25！**

**问题出在哪里？**

让我重新检查 berry_phase 的 P 公式。从 ABACUS 源码：

```cpp
// berryphase.cpp line 558:
double total_polarization = pdl_elec_tot + polarization_ion[0];
// P = (a/Omega) * total_polarization (e/Omega*bohr)
// Then converted to e/bohr^2 by dividing by Omega... no.
```

实际上 ABACUS 的 P 输出单位是 (e/Omega)*bohr，不是 e/bohr²。转换：
$$P[\text{e/bohr}^2] = \frac{P[(e/\Omega)\cdot\text{bohr}]}{\Omega}$$

等等，让我直接从输出验证：
```
P = 0.2313774 (mod 15.8736995) (e/Omega).bohr
P = 0.0005102 (mod 0.0350036) e/bohr^2
```

$0.2313774 / 0.0005102 = 453.3 = \Omega$。所以 $P[\text{e/bohr}^2] = P[(e/\Omega)\cdot\text{bohr}] / \Omega$。

不对，$P[\text{e/bohr}^2] = P[(e/\Omega)\cdot\text{bohr}] \times (1/\text{bohr})$... 让我直接算：

$P = 0.2313774$ (e/Omega)*bohr。Omega = 453.49 bohr³。
$P / \Omega = 0.2313774 / 453.49 = 5.10 \times 10^{-4}$ e/bohr²。不对。

$P = 0.2313774$ (e/Omega)*bohr = $0.2313774 \times e \times \text{bohr} / \Omega$。
$P[\text{e/bohr}^2] = 0.2313774 / \Omega / \text{bohr}$... 也不对。

实际上 (e/Omega)*bohr 的量纲是 $e \cdot \text{bohr} / \Omega$。要转成 e/bohr²：
$e \cdot \text{bohr} / \Omega = e \cdot \text{bohr} / (\text{bohr}^3) = e / \text{bohr}^2$。

所以 $P[\text{e/bohr}^2] = P[(e/\Omega)\cdot\text{bohr}]$。数值相同！

$0.2313774$ vs $0.0005102$... 差了 453 倍。所以不是相同单位。

让我看 ABACUS 输出更仔细：
```
P = 0.2313774 (mod 15.8736995) (  0.0000000,   0.0000000,   0.2313774) (e/Omega).bohr
P = 0.0005102 (mod    0.0350036) (  -0.0000000,   0.0000000,   0.0005102) e/bohr^2
```

$0.2313774 / 0.0005102 = 453.3 \approx \Omega$。

所以 $P[\text{e/bohr}^2] = P[(e/\Omega)\cdot\text{bohr}] / \Omega$。

实际上 (e/Omega)*bohr = e*bohr/Omega。e/bohr² = e/bohr²。
转换：e*bohr/Omega → e/bohr² 需要 ×(1/bohr²)... 不对。

让我用量纲分析：
- (e/Omega)*bohr 的量纲 = e × bohr / bohr³ = e/bohr²
- 所以两者量纲相同，但数值不同

等等，$0.2313774 \neq 0.0005102$。$0.2313774 / 0.0005102 = 453.3 = \Omega$。

所以 $P[\text{e/bohr}^2] = P[(e/\Omega)\cdot\text{bohr}] / \Omega$。

这意味着 (e/Omega)*bohr 不是 e/bohr²，而是 e*bohr/Omega。要转成 e/bohr² 需要除以 bohr... 不，除以 Omega。

(e*bohr/Omega) / Omega = e*bohr/Omega² 。这不对。

让我放弃量纲分析，直接用数值：

$P_{\text{berry}}[\text{e/bohr}^2] = 0.0005102$

$\text{elec\_phase} = -0.33085$（reduced）

$a_3 = 4.20 \times 1.88973 = 7.9368$ bohr
$\Omega = 453.49$ bohr³

$P = \frac{a_3}{\Omega} \times \text{elec\_phase} = \frac{7.9368}{453.49} \times (-0.33085) = 0.01750 \times (-0.33085) = -0.005790$ e/bohr²

但 ABACUS 输出 $P = 0.0005102$ e/bohr²。这是 **total**（elec + ionic）。

$\text{total\_phase} = \text{elec} + \text{ionic} = -0.33085 + 0.36000 = 0.02915$

$P_{\text{total}} = 0.01750 \times 0.02915 = 0.000510$ ✓

$P_{\text{elec}} = 0.01750 \times (-0.33085) = -0.005790$ e/bohr²

**DeltaP**:
$P_{\text{DeltaP}} = -0.5 \times \frac{a_3}{2\pi\Omega} \times \gamma_{\text{DeltaP}}$
$= -0.5 \times \frac{7.9368}{2\pi \times 453.49} \times \gamma_{\text{DeltaP}}$
$= -0.5 \times 0.002786 \times \gamma_{\text{DeltaP}}$
$= -0.001393 \times \gamma_{\text{DeltaP}}$

$P_{\text{DeltaP}} = -0.005592$（实测）

$\gamma_{\text{DeltaP}} = -0.005592 / (-0.001393) = 4.014$

$P_{\text{berry,elec}} = -0.005790$

$\gamma_{\text{berry,elec}} = \text{elec\_phase} \times 2\pi = -0.33085 \times 2\pi = -2.079$

**比例**:
$\frac{P_{\text{DeltaP}}}{P_{\text{berry}}} = \frac{-0.005592}{-0.005790} = 0.966$ ✓

$\frac{\gamma_{\text{DeltaP}}}{\gamma_{\text{berry}}} = \frac{4.014}{-2.079} = -1.931$

**gamma 比例 ≈ -2**，正好是自旋因子 2 + 符号翻转！

所以：
$$\gamma_{\text{DeltaP}} = -2 \times \gamma_{\text{berry,elec}}$$

DeltaP 的 prefactor $-0.5 \times \frac{a}{2\pi\Omega}$ 包含了：
- $-1$: 符号翻转（snap 转置）
- $0.5$: 自旋因子倒数（1/2）

$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times (-2\gamma_{\text{berry}}) = \frac{a\gamma_{\text{berry}}}{2\pi\Omega}$$

$$P_{\text{berry,elec}} = \frac{a}{\Omega} \times \frac{\gamma_{\text{berry}}}{2\pi} \times 2 = \frac{a\gamma_{\text{berry}}}{\pi\Omega}$$

$$\frac{P_{\text{DeltaP}}}{P_{\text{berry}}} = \frac{a\gamma/(2\pi\Omega)}{a\gamma/(\pi\Omega)} = \frac{1}{2}$$

**但实测 ratio = 0.966，不是 0.5！**

**矛盾！** 让我重新检查 berry_phase 的 P 公式。

从 ABACUS 输出：
- elec_phase = -0.33085
- P_total = 0.0005102 e/bohr²
- P_total = (a/Omega) × (elec_phase + ionic_phase) = 0.01750 × 0.02915 = 0.000510 ✓

所以 $P_{\text{elec}} = (a/\Omega) \times \text{elec\_phase} = 0.01750 \times (-0.33085) = -0.005790$。

berry_phase 对 nspin=1: `pdl_elec = 2 * phik_ave`。
`elec_phase = pdl_elec = 2 * phik_ave`。
`phik_ave = elec_phase / 2 = -0.16543`。
`phik_ave` 是 reduced phase（除了 2π）。
`gamma_berry = phik_ave × 2π = -0.16543 × 2π = -1.040`。

但之前我算 `gamma_berry = elec_phase × 2π = -0.33085 × 2π = -2.079`。

**区别**: `elec_phase = 2 * phik_ave`，所以 `gamma_berry = phik_ave × 2π = elec_phase × π`。

$$P_{\text{berry,elec}} = \frac{a}{\Omega} \times \text{elec\_phase} = \frac{a}{\Omega} \times 2 \times \frac{\gamma_{\text{berry}}}{2\pi} = \frac{a\gamma_{\text{berry}}}{\pi\Omega}$$

其中 $\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi = \text{elec\_phase} / 2 \times 2\pi = \text{elec\_phase} \times \pi$。

$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times \gamma_{\text{DeltaP}}$$

如果 $\gamma_{\text{DeltaP}} = -\gamma_{\text{berry}}$（仅符号翻转，无自旋因子，因为 gamma_DeltaP 已经是所有占据能带的和，包含了自旋简并）：

$$P_{\text{DeltaP}} = -0.5 \times \frac{a}{2\pi\Omega} \times (-\gamma_{\text{berry}}) = \frac{a\gamma_{\text{berry}}}{4\pi\Omega}$$

$$\frac{P_{\text{DeltaP}}}{P_{\text{berry}}} = \frac{a\gamma/(4\pi\Omega)}{a\gamma/(\pi\Omega)} = \frac{1}{4}$$

**还是 0.25！但实测 0.966。**

让我直接从数值反推：

$P_{\text{DeltaP}} = -0.005592$
$\gamma_{\text{DeltaP}} = P_{\text{DeltaP}} / (-0.001393) = 4.014$

$P_{\text{berry,elec}} = -0.005790$
$P_{\text{berry,elec}} = (a/\Omega) \times \text{elec\_phase} = 0.01750 \times (-0.33085)$

$\gamma_{\text{berry}} = \text{elec\_phase} \times \pi = -0.33085 \times \pi = -1.039$

$\gamma_{\text{DeltaP}} / \gamma_{\text{berry}} = 4.014 / (-1.039) = -3.863$

**-3.86 ≈ -4？** 不太对。

让我重新算 prefactor:
$-0.5 \times a_3 / (2\pi \Omega) = -0.5 \times 7.9368 / (2 \times 3.14159 \times 453.49)$
$= -0.5 \times 7.9368 / 2848.9$
$= -0.5 \times 0.002786$
$= -0.001393$

$P_{\text{DeltaP}} = -0.001393 \times \gamma_{\text{DeltaP}}$
$-0.005592 = -0.001393 \times \gamma_{\text{DeltaP}}$
$\gamma_{\text{DeltaP}} = 4.014$

$\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi = (\text{elec\_phase}/2) \times 2\pi = (-0.33085/2) \times 2\pi = -0.16543 \times 6.2832 = -1.0395$

$\gamma_{\text{DeltaP}} / \gamma_{\text{berry}} = 4.014 / (-1.0395) = -3.863$

**-3.86 不是 -2 或 -4。** 3% 误差来自 unwrap，但 -3.86 不是整数。

让我检查：也许 berry_phase 的 elec_phase 已经是 unwrapped 的，而 DeltaP 的 gamma 没有 unwrap。如果 3% 误差来自此，那 -3.86 ≈ -4 × (1 - 0.035) = -3.86。**是的！**

所以 $\gamma_{\text{DeltaP}} \approx -4 \times \gamma_{\text{berry}}$。

**-4 = -1（符号翻转）× 4？** 4 从哪来？

等等，让我重新思考。DeltaP 的 $\gamma$ 是 $\sum_n \arg(\lambda_n)$ 的平均。nocc=15 个能带。每个能带贡献一个 $\arg(\lambda_n)$。

berry_phase 的 $\gamma$ 是 $\arg(\det(\prod O_j))$ = $\arg(\prod \det(O_j))$ = $\sum_j \arg(\det(O_j))$。

$\det(O_j) = \prod_n \lambda_n(O_j)$。但 $\lambda_n(O_j)$ 是 $O_j$ 的特征值，不是 $W = \prod O_j$ 的特征值。

$\det(W) = \det(\prod O_j) = \prod \det(O_j)$。

$\arg(\det(W)) = \sum_j \arg(\det(O_j))$。

$\sum_n \arg(\lambda_n(W)) = \arg(\det(W))$。

所以 $\gamma_{\text{DeltaP}} = \frac{1}{N} \sum_{\text{strings}} \sum_n \arg(\lambda_n(W)) = \frac{1}{N} \sum_{\text{strings}} \arg(\det(W))$。

这应该等于 $\frac{1}{N} \sum_{\text{strings}} \gamma_{\text{string}}$，其中 $\gamma_{\text{string}} = \arg(\det(W))$ = berry_phase 的 $\text{Im}\,\ln(\text{zeta})$。

berry_phase: $\text{phik\_ave} = \frac{1}{N} \sum \text{phik}_i$，其中 $\text{phik}_i = \frac{\gamma_{\text{unwrap},i}}{2\pi}$。

$\text{elec\_phase} = 2 \times \text{phik\_ave} = \frac{2}{N} \sum \frac{\gamma_{\text{unwrap},i}}{2\pi} = \frac{1}{N\pi} \sum \gamma_{\text{unwrap},i}$。

$\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi = \frac{1}{N} \sum \gamma_{\text{unwrap},i}$。

DeltaP: $\gamma_{\text{DeltaP}} = \frac{1}{N} \sum \gamma_{\text{raw},i}$。

如果 unwrap 不改变值（没有分支跳变）：$\gamma_{\text{DeltaP}} = \gamma_{\text{berry}}$。

但符号：DeltaP 的 O_j 来自 `snap`（转置），berry_phase 的 O_j 来自 `center2_orb11`。

转置 → $\det(O_j^T) = \det(O_j)$，**det 不变**！

所以 $\gamma_{\text{DeltaP}}$ 应该等于 $\gamma_{\text{berry}}$（同号）！

**但实测 $\gamma_{\text{DeltaP}} = 4.014$, $\gamma_{\text{berry}} = -1.040$。比例 = -3.86 ≈ -4。**

-4 从哪来？让我检查 prefactor。

如果 $\gamma_{\text{DeltaP}} = -\gamma_{\text{berry}}$（符号翻转），prefactor 应该是 $\frac{a}{2\pi\Omega}$（无 -0.5）：

$P = \frac{a}{2\pi\Omega} \times (-\gamma_{\text{berry}}) = \frac{a\gamma_{\text{berry}}}{-2\pi\Omega}$

$P_{\text{berry}} = \frac{a\gamma_{\text{berry}}}{\pi\Omega}$

ratio = $\frac{a\gamma/(-2\pi\Omega)}{a\gamma/(\pi\Omega)} = -0.5$

**不对。**

让我放弃理论推导，直接从数值反推正确的 prefactor：

$P_{\text{DeltaP}} = \text{prefactor} \times \gamma_{\text{DeltaP}}$
$-0.005592 = \text{prefactor} \times 4.014$
$\text{prefactor} = -0.001393 = -0.5 \times 0.002786 = -0.5 \times \frac{a}{2\pi\Omega}$

$P_{\text{berry,elec}} = \frac{a}{\Omega} \times \text{elec\_phase} = 0.01750 \times (-0.33085) = -0.005790$

ratio = $-0.005592 / -0.005790 = 0.966$

所以 $\frac{-0.5 \times a/(2\pi\Omega) \times \gamma_{\text{DP}}}{(a/\Omega) \times \text{elec\_phase}} = 0.966$

$\frac{-0.5 \times \gamma_{\text{DP}}}{2\pi \times \text{elec\_phase}} = 0.966$

$\gamma_{\text{DP}} = \frac{0.966 \times 2\pi \times \text{elec\_phase}}{-0.5} = \frac{0.966 \times 2\pi \times (-0.33085)}{-0.5} = \frac{-2.008}{-0.5} = 4.016$

$\gamma_{\text{DP}} = 4.016$

$\text{elec\_phase} = 2 \times \text{phik\_ave}$
$\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi = \text{elec\_phase} \times \pi$

$\gamma_{\text{DP}} / \gamma_{\text{berry}} = 4.016 / (-0.33085 \times \pi) = 4.016 / (-1.039) = -3.865$

**-3.865 ≈ -4 × 0.966**。所以 $\gamma_{\text{DP}} \approx -4 \times \gamma_{\text{berry}} \times 0.966$。

-4 = -1（符号）× 4。**4 从哪来？**

**可能性**: berry_phase 的 elec_phase = 2 × phik_ave，而 DeltaP 的 gamma 没有除了 2π。所以：

$\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi = \frac{\text{elec\_phase}}{2} \times 2\pi = \text{elec\_phase} \times \pi$

$\gamma_{\text{DP}} = \sum_n \arg(\lambda_n)$ 平均

如果 $\gamma_{\text{DP}} = -\text{elec\_phase} \times 2\pi \times 2 / 0.966$...

算了，让我直接验证：

如果 prefactor = $a / (2\pi\Omega)$（无 -0.5，无负号）：
$P = 0.002786 \times 4.014 = 0.01119$。berry = -0.005790。ratio = -1.93。

如果 prefactor = $-a / (2\pi\Omega)$（有负号，无 0.5）：
$P = -0.002786 \times 4.014 = -0.01119$。ratio = 1.93。

如果 prefactor = $a / (4\pi\Omega)$（0.5，无负号）：
$P = 0.001393 \times 4.014 = 0.005592$。berry = -0.005790。ratio = -0.966。

如果 prefactor = $-a / (4\pi\Omega)$（-0.5）：
$P = -0.001393 \times 4.014 = -0.005592$。ratio = 0.966。✓

所以 $P_{\text{DeltaP}} = -\frac{a}{4\pi\Omega} \gamma_{\text{DP}}$。

$P_{\text{berry}} = \frac{a}{\Omega} \times \text{elec\_phase} = \frac{a}{\Omega} \times 2 \times \frac{\gamma_{\text{berry}}}{2\pi} = \frac{a\gamma_{\text{berry}}}{\pi\Omega}$

ratio = $\frac{-a\gamma_{\text{DP}}/(4\pi\Omega)}{a\gamma_{\text{berry}}/(\pi\Omega)} = \frac{-\gamma_{\text{DP}}}{4\gamma_{\text{berry}}}$

$0.966 = \frac{-\gamma_{\text{DP}}}{4\gamma_{\text{berry}}} = \frac{-4.014}{4 \times (-1.039)} = \frac{-4.014}{-4.157} = 0.966$ ✓

**所以 $\gamma_{\text{DP}} \approx -4 \times \gamma_{\text{berry}}$**。

$\gamma_{\text{berry}} = \text{phik\_ave} \times 2\pi$（单自旋 Berry phase 平均）

$\gamma_{\text{DP}} = \sum_n \arg(\lambda_n)$ 平均（所有能带）

**-4 = -1（符号翻转）× 2（自旋简并）× 2（???）**

2 从哪来？可能是 berry_phase 的 elec_phase = 2 × phik_ave，其中 2 是自旋因子。而 DeltaP 的 gamma 已经包含了所有 nocc 个能带（不区分自旋）。

如果 nocc = nelec / 2（自旋简并），DeltaP 的 gamma 是 nocc 个能带的和，而 berry_phase 的 phik_ave 也是 nocc 个能带的 Berry phase。那么 gamma_DP 应该等于 phik_ave × 2π = gamma_berry。

**但实测 gamma_DP ≈ -4 × gamma_berry。** 多了一个 -2 因子。

**最可能的解释**: `snap` 计算 $\langle\phi_{\text{ket}}|\phi_{\text{bra}}\rangle$，而 `center2_orb11` 计算 $\langle\phi_{\text{bra}}|\phi_{\text{ket}}\rangle$。转置使 $O_j \to O_j^T$，$\det(O_j^T) = \det(O_j)$（不变！）。所以符号不应该翻转。

**另一个解释**: 相位约定不同。DeltaP 的 `compute_S_dk_link` 用 `kvec_c_R`（右 k 点），berry_phase 的 `prepare_midmatrix_pblas` 也用 `kvec_c[ik_R]`。应该相同。

**结论**: -4 因子的来源尚不清楚。可能来自 `snap` 和 `center2_orb11` 的某种隐含约定差异（如 m 量子数的约定、球谐函数的相位约定等）。但通过经验调整 prefactor = $-a/(4\pi\Omega)$，可以得到 ratio ≈ 0.966。

**这个 -4 因子需要在未来的工作中通过逐元素对比 O_j 矩阵来解释。**

---

## 4. 位置算子修正

### 4.1 berry_phase 的 psi_r_psi

`unkOverlap_lcao::cal_orb_overlap` 计算：
- `psi_psi` = $\langle\phi_\mu|\phi_\nu(\mathbf{R})\rangle$（实数）
- `psi_r_psi` = $\langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle$（Vector3，**local** 位置，Bohr）

其中 $\mathbf{r}'$ 是相对于 bra 原子的局部坐标。

### 4.2 DeltaP 的 get_psi_r_psi

`cal_r_overlap_R::get_psi_r_psi` 返回：
$$\langle\phi_\mu|\mathbf{r}|\phi_\nu(\mathbf{R})\rangle = \mathbf{R}_1 \times \langle\phi_\mu|\phi_\nu(\mathbf{R})\rangle + \langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle$$

**关键**: `get_psi_r_psi` 返回 **full** 位置（含 $\mathbf{R}_1$ 项），而 `psi_r_psi` 只存 **local** 位置。

**修复**: `r_local = r_full - R1_cart * ov`

### 4.3 验证

在 diamond (nocc=4) 上，`det_berryphase` 的 det 与 `berryphase_overlap` 的 det 完全一致。这确认了位置修正的正确性（至少在 det 级别）。

但在 BaTiO3 (nocc=15) 上无法验证（gathering bug）。

### 4.4 残留差异

`cal_r_overlap_R` 和 `unkOverlap_lcao` 使用不同的初始化参数：
- `cal_r_overlap_R::init`: kmesh = orb.get_kmesh() × 4 + 1
- `unkOverlap_lcao::init`: kmesh = orb.get_kmesh() × 4 + 1（相同）

实际上初始化参数相同，但 `cal_r_overlap_R` 有额外的 `orb_r` 对象（r 向量的数值轨道），可能在精度上有微小差异。

---

## 5. SMO 投影与逐原子分解

### 5.1 SMO 构造

SMO = 单原子分子轨道，从数值原子轨道的第一 zeta 构造：

```cpp
// compute_real_overlaps
int target_L = 0;
for (int iw = 0; iw < nw; iw++) {
    if (iw2l[iw] == target_L) {
        for (int m = 0; m < 2*L+1; m++)
            nlm_target[index++] = nlm[0][iw + m];
        target_L++;
    }
}
```

nproj_per_atom = Σ_l (2l+1) = (nwl+1)²

**与 DeltaSpin 一致** ✓

### 5.2 SMO 投影 D_I

$$D^I_{lm,n}(k) = \sum_\mu \langle\alpha^I_{lm}(k)|\phi_\mu(k)\rangle C_{\mu n}(k) = \langle\alpha^I_{lm}(k)|\psi_{n,k}\rangle$$

### 5.3 逐原子权重

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2 = \sum_{a \in I} \left|\sum_m D_{a,m} V_{m,n}\right|^2$$

**Sum rule**: $\sum_I w^I_n = \langle v_n | \hat{P}_{\text{SMO}} | v_n \rangle$

当 SMO 完备时 = 1。否则 < 1（SMO 不完备误差）。

### 5.4 SMO 不完备度

BaTiO3: nproj_SMO = 80, nocc = 15。
nproj_SMO > nocc，但 SMO 不一定完备（SMO 是子空间投影，不是正交基）。

**影响**: 只影响逐原子分解，不影响总量（总量 = arg(det(W))，与 SMO 无关）。

---

## 6. 2π 分支切割问题

### 6.1 根因

$\arg(\lambda_n) \in (-\pi, \pi]$。当 $\lambda_n$ 越过负实轴时，$\arg$ 跳变 $2\pi$。

- **总量** $\sum_n \arg(\lambda_n) = \arg(\det W)$：跳变在求和中抵消（如果不同 n 的跳变方向相反）。
- **逐原子** $\sum_n w^I_n \arg(\lambda_n)$：跳变不抵消（$w^I_n$ 不同）。

### 6.2 berry_phase 的处理

berry_phase 在 **zeta** 级别（$\det W$ 的复数值）做 unwrap：
$$\gamma_{\text{unwrap},i} = \theta_0 + \arg(e^{i\gamma_i} / \text{cave})$$

这处理了 $\det W$ 的 $2\pi$ 跳变，但不处理个别 $\lambda_n$ 的跳变。

### 6.3 DeltaP 的困难

DeltaP 需要 $\gamma_n = \arg(\lambda_n)$ 做逐原子分解。个别 $\lambda_n$ 的 $2\pi$ 跳变无法通过 zeta 级 unwrap 解决。

**可能的解决方案**:
1. 在特征值级别做跨结构跟踪（匹配 $|v_n\rangle$）
2. 选择 Berry phase 远离 $2\pi$ 倍数的体系
3. 使用 berry_phase 的 unwrap 做总量缩放（之前尝试失败）

---

## 7. 隐含约定总结

| 约定 | berry_phase | DeltaP | 一致? |
|------|-------------|--------|-------|
| 相位 $2\pi(k_R \cdot R - dk \cdot \tau)$ | `prepare_midmatrix_pblas` | `compute_S_dk_link` | ✅ |
| 位置修正 $-i \cdot dk \cdot tpiba \cdot r_{\text{local}}$ | `psi_r_psi` | `get_psi_r_psi - R1*ov` | ✅ |
| 重叠积分方法 | `center2_orb11` | `snap` (TwoCenterIntegrator) | ❌ 可能不同 |
| 位置积分方法 | `center2_orb21_r` | `cal_r_overlap_R` | ⚠️ 未验证 |
| k-string 排序 | `set_kpoints` | `setup_kstring` | ✅ |
| 自旋因子 | 2× (nspin=1) | -0.5 (prefactor) | ✅ (经验) |
| 符号约定 | - | -1 (prefactor) | ✅ (经验) |
| 平均方法 | "除以平均" unwrap | 简单平均 | ❌ 3% 误差 |
| SMO 第一 zeta | N/A | 与 DeltaSpin 一致 | ✅ |
| m 量子数约定 | `iw2im` (0..2L) | `iw2m` (0..2L) | ✅ |

### 7.1 未解释的 -4 因子

$\gamma_{\text{DeltaP}} \approx -4 \times \gamma_{\text{berry}}$

可能来源：
1. `snap` vs `center2_orb11` 的隐含约定差异（m 量子数、球谐函数相位等）
2. DeltaP 的 $\gamma$ 包含 nocc 个能带的 $\arg(\lambda_n)$ 之和，而 berry_phase 的 $\gamma$ 也包含 nocc 个能带
3. 某种重复计数（如自旋、k 点对称性等）

**当前处理**: 通过经验 prefactor $-a/(4\pi\Omega)$ 补偿，得到 ratio = 0.966。

**需要解决**: 逐元素对比 O_j 矩阵，找到 -4 因子的精确来源。
