# DeltaP SMO 投影算法评估 (2026-07-11)

## 1. 问题提出

当前 DeltaP 的**每原子极化分解**使用 SMO (Smoothed Maximum Overlap) 投影：
```
gamma_I = Σ_n w_In × arg(λ_n)
w_In = Σ_{lm∈I} |tilde_proj[lm,n]|²
```
其中 `tilde_proj = S^{-1/2} · proj`（Löwdin 正交化）。

## 2. SMO 投影是否等价于 Mulliken population？

**不。** D_I = <α_iat_lm | ψ_n> 是 SMO 基函数与 KS 轨道的内积:

```cpp
D_I[iat][lm][n] = Σ_μ conj(S_k[iat][lm][μ]) × ψ_n(μ)
```

这是直接重叠积分，**不是** Mulliken 分析。Mulliken population 涉及密度矩阵和 AO 重叠矩阵。

## 3. 数值检证: SMO 正交化是错误的

从 H₂O 测试输出:

| 指标 | 期望值 | 实际值 | 评估 |
|------|--------|--------|------|
| S 矩阵迹 | 17 | 17 | 正确 |
| S 本征值范围 | [>0, ~1-2] | [0.10, 2.18] | 条件数 ~21，可逆 |
| **S·Sinv 最大误差** | **≈ 0** | **0.237** | **严重** |
| **S·Sinv·S - S** | **≈ 0** | **0.282** | **严重** |
| raw_sum (n=0) | ~O(1) | 5.91 | 远超 1 |
| tilde_sum (n=0) | ~O(1) | 5.21 | 远超 1 |
| tilde_sum (n=1) | ~O(1) | 18.35 | 远超 1 |

**S^{-1} 的误差 ~24%**。Löwdin 正交化 `S^{-1/2}` 在此误差下产生**不可靠的权重**。

## 4. 为什么 orthogonalization 失效

SMO 基函数是**原子中心、非正交**的。相邻原子间的 SMO 重叠产生大的非对角元。

`S^{-1}` 通过数值求逆计算（可能 LAPACK `dsyev` 或 `zgetrf`），对于条件数 ~21 的矩阵应精确。但 `S·Sinv max_err = 0.24` 说明逆矩阵计算本身有数值问题（可能是精度不够或算法选择不当）。

## 5. 后果: 每原子 Gamma 分配不可靠

- `w_In` **不满足 partition of unity**: Σ_I w_In ≠ 1
- 不同原子的权重**互相污染**（non-orthogonal SMO basis）
- Per-atom gamma 是**非物理的**，依赖 chosen SMO basis size 和 orthogonalization 精度
- **gamma_I 约束的物理意义不明确**: 约束一个非物理的原子分解量，系统可能走向错误状态

## 6. 但是: 总极化 P_total 是正确的

```
P_total = Σ_I P_I = prefactor × Σ_I gamma_I
        = prefactor × (1/n_strings) × Σ_string arg(det(W_string))
```

Wilson loop 的幺正不变性: 本征值 λ_n 在占据子空间旋转下不变 → det(W) 不变 → P_total 不变。P_total **与原子投影无关**，是可靠的物理量。

## 7. 评估结论

| 量 | 物理可靠? | 说明 |
|----|---------|------|
| P_total | 是 | Wilson loop 行列式，幺正不变 |
| gamma_I (per-atom) | **否** | SMO 投影不满足 partition of unity |
| DeltaP constraint on P_total | 可能 | 需要修改为约束总和 |
| DeltaP constraint on per-atom | **不可靠** | 约束非物理量 |

## 8. 建议

### 短期（当前框架可用）
1. **约束 P_total 而非 gamma_I** — P_total 是物理的
2. 修复 S^{-1} 的数值精度（当前 24% 误差不可接受）
3. 验证修复后 γ_I 是否满足 Σ_I w_In = 1

### 中期
4. 用 Wannier90 MLWF 替代 SMO 投影做原子分解
5. 或用 Bader-like 电荷分区方案

### 风险评估
若继续用不可靠的 per-atom γ_I 做约束 SCF:
- Lambda 可能 push 系统到非物理状态
- Gamma 振荡可能部分来源于 SMO 投影的不稳定性
- 正确性无法验证（没有 ground truth 基准）
