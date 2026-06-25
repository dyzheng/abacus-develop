# 跨SCF平滑性保障机制：Wilson loop + 分支跟踪

> **日期**: 2026-06-25  
> **问题**: 策略2（子空间∂_k C）仅保障内循环内平滑性，跨独立SCF如何保障P^I连续？

---

## 1. 两种平滑性需求

### 1.1 内循环内平滑性

场景：一次SCF中，λ变化→子空间对角化更新C→重算P^I

保障机制：**策略2（子空间响应）**
- ∂_k C通过子空间微扰估计（解析，平滑）
- 子空间本征矢可对角化固定规范（无相位任意性）
- H^λ = ∂_k Fourier(H^pre)·λ（预存储，平滑）

### 1.2 跨独立SCF平滑性

场景：不同结构的独立SCF运行（Z*有限差分、E(P)扫描、结构弛豫）

保障机制：**Wilson loop + 分支跟踪**
- 独立SCF的波函数有任意相位 → Berry connection积分不连续
- Wilson loop的det()消除相位 → W^I复数随结构连续变化
- arg(W^I)的分支切割通过跨结构跟踪展开

---

## 2. 跨SCF不连续的根源

独立SCF对角化：
```
结构A: |ψ_nk^A⟩ = e^{iφ_A(n,k)} |ψ̃_nk⟩
结构B: |ψ_nk^B⟩ = e^{iφ_B(n,k)} |ψ̃_nk⟩
```

Berry connection积分依赖∂_k C，而C的相位φ_A ≠ φ_B：
```
P^I(A) = f(C^A, ∂_k C^A)  ← 依赖φ_A
P^I(B) = f(C^B, ∂_k C^B)  ← 依赖φ_B
```

P^I(A)和P^I(B)的差值包含相位差∂_k(φ_A - φ_B)的虚假贡献，与结构变化无关。

---

## 3. Wilson loop的规范不变性证明

### 3.1 Wilson loop乘积

$$W^I = \prod_j \det\left(\mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}^I(k_{j+1})\right)$$

### 3.2 规范变换下的不变性

波函数规范变换：$C(k) \to C(k) \cdot e^{i\varphi(k)}$（对角相位矩阵）

**D_I的变换**：
$$D_I(k) = S^*(k) \cdot C(k) \to S^*(k) \cdot C(k) \cdot e^{i\varphi(k)} = D_I(k) \cdot e^{i\varphi(k)}$$

**SVD极分解的变换**：
$$D_I = W\Sigma V^\dagger \to (W \cdot e^{i\varphi}) \Sigma V^\dagger$$

极分解 $U = WV^\dagger \to U \cdot e^{i\varphi(k)}$

（SVD的W吸收相位，V不变——因为相位是右乘的）

**重叠矩阵O的变换**：
$$O = C^\dagger(k_j) \cdot S(\Delta k) \cdot C(k_{j+1}) \to C^\dagger(k_j) \cdot e^{-i\varphi(k_j)} \cdot S(\Delta k) \cdot C(k_{j+1}) \cdot e^{i\varphi(k_{j+1})}$$
$$= e^{-i\varphi(k_j)} \cdot O \cdot e^{i\varphi(k_{j+1})}$$

**M^I的变换**：
$$M^I = U^{I\dagger}(k_j) \cdot O \cdot U^I(k_{j+1})$$
$$\to (U^I \cdot e^{i\varphi_j})^\dagger \cdot e^{-i\varphi_j} O e^{i\varphi_{j+1}} \cdot (U^I \cdot e^{i\varphi_{j+1}})$$
$$= e^{-i\varphi_j} U^{I\dagger} \cdot e^{-i\varphi_j} O e^{i\varphi_{j+1}} \cdot U^I e^{i\varphi_{j+1}}$$

Wait, this doesn't simplify to M^I. Let me redo:

$$M^I \to (e^{-i\varphi_j} U^{I\dagger}) \cdot (e^{-i\varphi_j} O e^{i\varphi_{j+1}}) \cdot (U^I e^{i\varphi_{j+1}})$$
$$= e^{-2i\varphi_j} U^{I\dagger} O U^I e^{2i\varphi_{j+1}}$$
$$= e^{-2i\varphi_j} M^I e^{2i\varphi_{j+1}}$$

Hmm, this gives $\det(M^I) \to e^{-2i\varphi_j \cdot n_{proj}} \det(M^I) e^{2i\varphi_{j+1} \cdot n_{proj}}$

For a single SMO channel (n_proj = 1):
$\det(M^I) \to e^{-2i\varphi_j} \det(M^I) e^{2i\varphi_{j+1}} = e^{2i(\varphi_{j+1} - \varphi_j)} \det(M^I)$

The Wilson loop product:
$W^I = \prod_j \det(M^I_j) \to \prod_j e^{2i(\varphi_{j+1} - \varphi_j)} \det(M^I_j) = e^{2i\sum_j(\varphi_{j+1}-\varphi_j)} \prod_j \det(M^I_j)$

The phase sum telescopes: $\sum_j (\varphi_{j+1} - \varphi_j) = \varphi_{N} - \varphi_0 = 0$ (periodic boundary)

**Therefore $W^I$ is gauge-invariant!** ✓

(The phase factors cancel in the telescoping product around the k-string loop.)

### 3.3 跨SCF连续性

$W^I$ as a complex number depends on:
- $D_I(k)$ → changes continuously with structure (S and C both continuous)
- $O(k_j, k_{j+1})$ → changes continuously with structure
- SVD $U = WV^\dagger$ → continuous when singular values don't cross

Therefore $W^I$ changes continuously with structure (as long as no singular value crossing).

### 3.4 arg()分支切割

$P^I = -(a_\alpha/2\pi\Omega) \cdot \arg(W^I)$

$\arg(\cdot) \in (-\pi, \pi]$ has a branch cut at $W^I \in \mathbb{R}^-$.

When $W^I$ crosses the negative real axis, $\arg$ jumps by $\pm 2\pi$.

**分支跟踪**：比较相邻结构的$\arg$值，如果跳变$> \pi$，加$\pm 2\pi$校正：

```
arg_continuous(a) = arg(W^I_a)
if |arg_continuous(a) - arg_continuous(a-1)| > π:
    arg_continuous(a) += 2π × sign(arg_continuous(a-1) - arg_continuous(a))
```

这给出连续的$P^I$，直到跨过极化量子$eR/\Omega$（需要物理判断选择分支）。

---

## 4. 与H^λ的分工

| 计算 | 机制 | 平滑性范围 |
|------|------|-----------|
| **P^I评估** | Wilson loop + 分支跟踪 | 跨SCF + 内循环（规范不变） |
| **H^λ构建** | 策略2子空间响应 | 仅内循环（解析响应） |
| **∂P/∂λ** | 子空间微扰 | 仅内循环（解析） |

P^I始终用Wilson loop（无论场景）。H^λ仅在λ内循环中需要。

---

## 5. 实现路径

### 5.1 当前状态

`deltap_wannier.cpp`已实现：
- SVD极分解（zgesvd）✓
- Wilson loop乘积（zgetrf det）✓
- 但使用O≈I近似（需替换为精确O）

### 5.2 需要新增

1. **精确重叠矩阵O(k_j, k_{j+1})**：
   - $S(\Delta k) = \sum_R e^{2\pi i \Delta k \cdot R} S(R)$ — 预计算一次
   - $O = C^\dagger(k_j) \cdot S(\Delta k) \cdot C(k_{j+1})$ — zgemm
   - S(R) = ⟨φ_μ(0)|φ_ν(R)⟩ — 可用overlap_orb二中心积分

2. **分支跟踪**：
   - 存储上一个结构的W^I（复数）
   - 比较arg，展开跳变
   - ~20行代码

3. **替换deltap_wannier.cpp中的O≈I为精确O**：
   - 修改M^I计算：M^I = U^I† · O · U^I（当前是U^I† · U^I）

### 5.3 代码量估计

| 组件 | 新增代码 | 复用 |
|------|---------|------|
| S(Δk)预计算 | ~30行 | S(R)从overlap_orb |
| O = C†·S(dk)·C | ~20行 | zgemm |
| 精确M^I = U†·O·U | ~15行 | 修改deltap_wannier.cpp |
| 分支跟踪 | ~20行 | 新增 |
| **总计** | **~85行** | 大部分复用现有 |

### 5.4 验证方案

1. **规范不变性测试**：对同一结构，用不同随机相位运行SCF，验证W^I不变
2. **跨结构平滑性测试**：±0.01 Bohr位移，验证P^I变化线性
3. **Z*对比**：与berry_phase Z*（Ti=6.80, Ba=2.67）对比
4. **分支跟踪测试**：扫描大位移，验证跨极化量子的连续性
