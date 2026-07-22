# DeltaP λ→γ 响应测试——分支选择非确定性分析

> 日期: 2026-07-19

---

## 一、λ sweep 测试数据

deltap_nscf=0 (sync 模式, 梯度下降), mixing_beta=0, 每次独立运行删除 match 文件

| λ_init | g0 | g1 | 分支状态 |
|--------|-----|------|----------|
| 0.0 | -4.2718 | -4.9825 | Branch A |
| 0.1 | +0.9448 | +0.9832 | Branch B (对干支) |
| 0.5 | -4.2793 | -4.9750 | Branch A |
| 1.0 | +0.8647 | +1.0174 | Branch B |
| 2.0 | +14.578 | +17.379 | Branch C (跳至远支) |
| 5.0 | -4.2585 | -4.9958 | Branch A |

模式: g 在约 -5 和约 +1 之间交替跳变 (差约 6 rad ≈ 2π·w_sum), λ=2.0 处跳到 +15。

---

## 二、根因分析

### 2.1 sign alternation 的数学解释

无 L1 match freeze 时，匈牙利匹配每次独立计算。zgeev 的特征值返回顺序不确定 → first_diag 按 arg 排序列 → 可能两带的 arg 相近时排序易位 → 不同 λ 下排列不同。

排列不同 → `gamma_unwrapped[n]` 跨带重排 → `γ_I = Σ_n w_In × γ_unwrapped[n]` 的带权重分配不同 → per-atom γ 跳变。

### 2.2 为什么 sign flip (~5 ↔ ~+1) 是系统性模式

BN 的 per-atom γ 量级约 1.7 rad (B, 带 2 贡献 93%)。在分支 A: g≈-5 (相当于 -1.7 - 2π)。分支 B: g≈+1 (相当于 +1.7 - 2π·0)。两者的差 ≈ 6.28 = 2π。

实际上，2π·w_sum(B) ≈ 2π×0.8 ≈ 5.02, 2π·w_sum(N) ≈ 2π×0.95 ≈ 5.95。sign flip 发生在 w_sum 差异足够大，一个 2π 跳变影响 per-atom γ 约 5-6 rad。

### 2.3 λ=2.0 处的 +15 跳变

per-atom γ = +14.6/+17.4 是更远的 branch (约 +3×2π·w_sum)。这表明较大的 λ 改变了波函数结构，使得 Hungarian 匹配跳到了完全不同的本征值配对。

---

## 三、对 accept_trial 插值公式的影响

### 3.1 插值逻辑

```
α_opt = α_trial × Σ(r_cur × Δγ) / Σ(Δγ)²
```

其中:
- r_cur = γ_cur - target  (内循环前的残差)
- Δγ = γ_trial - γ_cur  (trial λ 后的 γ 变化)

### 3.2 分支跳变下的灾难性后果

若 γ_cur 在 Branch A (g≈-5) 而 γ_trial 在 Branch B (g≈+1):

```
Δγ ≈ (+1) - (-5) = +6
r_cur ≈ -5 - 0 = -5
Σ(r_cur × Δγ) ≈ (-5) × (+6) = -30
Σ(Δγ)² ≈ 36
α_opt / α_trial = -30/36 ≈ -0.833
α_opt ≈ -0.83 × 0.5 ≈ -0.42
```

优化器认为 Δγ 是物理响应(实际是分支跳变)，计算出的 α_opt 完全错误。λ 被往反方向推。

若 γ_cur 和 γ_trial 在同一 branch:

```
Δγ ≈ 0.02 (dγ/dλ × λ_trial)
r_cur ≈ -5
Σ(r_cur × Δγ) ≈ (-5) × (0.02) = -0.1
Σ(Δγ)² ≈ 0.0004
α_opt / α_trial = -0.1/0.0004 = -250
α_opt ≈ -125
```

α_opt 被推至极值 (check_restriction 限制为 max_step=0.005)。

两种情况都产生错误的 λ 更新:
- 跨 branch 跳变: sign flip → λ 走错方向
- 同 branch 微扰: Δγ 太小 → α_opt 爆炸 → λ 步长被 max_step 卡住

---

## 四、为什么之前的测试得出 dγ/dλ≈0.02 但实际不可靠

**关键**: dγ/dλ≈0.02 的测量依赖跨 λ 的 branch 一致性。若 λ=0 和 λ=0.05 落在不同 branch，则所谓"响应"是 branch artifact，不是物理信号。

从 λ sweep 数据看:
- λ=0.0→0.1: g 从 -5 跳到 +1 → Δg=6, dγ/dλ=60 (artifact!)
- λ=0.5→1.0: g 从 -5 跳到 +1 → 同样的 artifact

07-15 的 0.02 rad/λ 是在 L1 match freeze 生效下测得的 (同一次计算内，match 在 iter=1 保存后复用)。这在同一个 SCF 迭代序列内是有效的，但跨 λ_init 值不适用。

**结论: 跨 λ 的独立响应测试在不冻结匹配的条件下没有意义。**

---

## 五、正确的测试方法

### 5.1 同一次 SCF 序列内的内循环测试

使用 deltap_nscf>0 (内循环激活):

```
iter=1:
  - solve(λ=0) → psi_0
  - iter_finish: 创建 dp_scf_, compute_gamma_scf, save_match
  - 梯度下降: lambda += step × γ (若 nscf=0)

iter=2:
  - iter_init: H = H_KS + H_HK(λ_from_iter1)
  - hamilt2rho_single:
    - compute_gamma_scf → γ_cur  [第1次测量]
    - bfgs.start_outer(lambda)
    - inner=0:
      - bfgs.step → lam_trial = lambda + α_trial × r_cur
      - set_lambda(lam_trial), compute_hk_correction
      - solve(H, psi, ...) → psi_trial  [重新对角化]
      - compute_gamma_scf → γ_trial  [第2次测量]
      - Δγ = γ_trial - γ_cur
      - bfgs.accept_trial(residual)
```

在这个流程中, γ_cur 和 γ_trial 在同一个 SCF 迭代内, 使用同一个 deltap_match.dat (iter=1 保存的)。branch 一致性由 L1 freeze 保证。

### 5.2 手动 single-point λ 测试协议

1. 运行 iter=1 (λ=0), 输出 deltap_match.dat
2. 保存 charge density
3. 以 iter=1 的 charge 做 NSCF restart, 加载 deltap_match.dat
4. 分别设置不同 λ, 测 γ(λ)
5. 差值 Δγ/Δλ = 真正的抑制响应

但此协议仍需代码支持 (加载 match 文件 + 固定 λ + NSCF 模式)。当前代码在 NSCF 模式下不会自动加载 match 或应用 HK 修正。

---

## 六、内循环有效性的前置条件

要使 accept_trial 的插值有意义, 需要:

1. **L1 match freeze 已生效**: inner loop 中 γ_cur 和 γ_trial 使用相同的 match_to[] — 已实现 (iter=1 保存, iter=2+ 加载)
2. **Δγ >> branch spacing**: 若 λ 改变产生的真实 Δγ > 2π·w_sum (约 5 rad), branch 选择可以确定。当前 dγ/dλ≈0.02, 需 λ≈250 → 远超 λ=0.5 即跳变的窗口
3. **HK 响应量程**: 在 λ 稳定窗口 (|λ|<0.5) 内, 最大 Δγ ≈ 0.02×0.5 = 0.01 rad, 远小于 statistical noise (约 0.05 rad) → 信号被噪声淹没
4. **替代方案**: 不在内循环中插值, 而是在外循环中用二阶有限差分或 Gaussian process 拟合 γ(λ)

---

## 七、裁决

**内循环 accept_trial 插值公式在当前参数下不可工作**, 原因:
1. 分支跳变使 γ(λ) 不是 λ 的连续函数
2. 线性区内 dγ/dλ 太小 (0.02 rad/λ), 测量噪声淹没信号
3. max_step=0.005 将 α_opt 压制至不产生有效 λ 移动

**推荐的短期方向**: 删除 accept_trial 的线性插值, 改用纯梯度下降 (dλ = -α × dγ 方向), 去耦合内循环与 SCF 迭代 (每 n 个 SCF iter 做一次 λ update)。这与 DeltaSpin 的 "threshold" 模式一致。
