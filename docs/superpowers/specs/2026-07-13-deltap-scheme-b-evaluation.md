# 方案 B (A_nk 直接积分) 可行性评估

> 日期: 2026-07-13

---

## 一、两种方案对比

### 方案 A: Wilson Loop (已实现)

```
每条 k-string:
  O_j = C†(k_j)·S(dk)·C(k_{j+1})          O(Nbasis² × nocc)
  W = O_0 × O_1 × …                        O(N_links × nocc³)
  diagonalize W → evals[1..nocc]             O(nocc³)
  unwrap + Hungarian matching                O(nocc³)
  gamma = Σ w^I_n × arg(eval_n)              O(nocc × nat)
→ 平均所有 string → γ^I → P^I
```

**特点**: 对规范变换不变 (只依赖 det W), 以 nocc³ 的代价自动处理 2π 分支。

### 方案 B: A_nk 直接积分 (部分实现)

```
每条 k-string 的每个 k 点:
  A_nk[alpha] = i⟨u_nk|∂/∂k_alpha|u_nk⟩   O(Nbasis² × nocc)
  term1 = conj(C)·dS·C                       O(Nbasis² × nocc)
  term2 = conj(D_I)·d(D_I)                   O(nat × nproj × nocc)
→ 求和所有 k 点 → γ^I → P^I
```

**特点**: 逐 k 点直接求和, 无需矩阵对角化, O(nocc) 更简单。但对规范变换敏感。

---

## 二、实测差距分析

在 BN 2×2×2 上实测:

| | Wilson (方案 A) | A_nk (方案 B, 当前) | 比值 |
|------|:---:|:---:|:---:|
| P_total | ~1.2×10⁻² | ~3×10⁻⁶ | **~4000×** |

差距来源:

| 因素 | 贡献 |
|------|------|
| **k 点采样**: Wilson 平均 4 strings × 3 kpts = 12 kpts; A_nk 只用 string 0 = 3 kpts | ~4× |
| **归一化**: Wilson 用 `spin × R/(2πV)`; A_nk 用 `-R/(2πV) × dk` | 符号 + 因子 |
| **gauge**: Wilson 规范不变; A_nk 依赖 gauge_fix 质量 | 未知量级 |

4× 的采样差距不能解释 4000× 的差值。根本原因在**归一化公式和积分范围**——当前 `integrate_polarization` 的公式与 Wilson 不是同一物理量。

---

## 三、内存代价

### 方案 A (Wilson Loop)

| 结构 | 大小 (BN) | 通用公式 |
|------|------|------|
| S_dk_ (3 方向) | 33 KB | 3 × N_basis² × 16B |
| O_kpair | 256 B | nocc² × 16B |
| W_mat, evals | ~400 B | 3 × nocc² × 16B |
| 临时 (SC, etc.) | ~2 KB | N_basis × nocc × 16B |
| **合计** | **~36 KB** | O(N_basis²) |

### 方案 B (A_nk)

| 结构 | 大小 (BN) | 通用公式 |
|------|------|------|
| A_nk_ (3 方向, 全 strings) | 1.2 KB | nat × N_kpts × nocc × 3 × 16B |
| kstring_data_ (S_k, D_I) | ~100 KB | nat × nproj × nocc × N_kpts × 16B |
| **合计** | **~100 KB** | O(nat × nproj × nocc × N_kpts) |

对于大体系 (100 原子, 500 带, 20 k 点/string):

| 方案 | 内存 |
|------|------|
| A (Wilson) | ~50 MB (S_dk_ × 3) |
| B (A_nk) | ~500 MB (A_nk_ + kstring_data_) |

**方案 B 的内存代价约为方案 A 的 10 倍**, 主要来自 kstring_data_ (已存在于方案 A 的 O_kpair 路径中, 但 A 每条 string 复用, B 需要同时保存所有 k 点)。

---

## 四、收益分析

### B 的优势

| 优势 | 说明 | 重要程度 |
|------|------|:---:|
| **无分支选择** | 不经过 2π unwrapping, 无匈牙利匹配, 确定性保证更强 | 高 |
| **逐 k 点诊断** | 可以定位哪个 k 点的 Berry 连接异常 | 中 |
| **无矩阵对角化** | 跳过 O(nocc³) 的 zgeev, nocc 大时有利 | 低 (nocc≤100) |
| **独立验证** | 与 Wilson 交叉验证, 增加可信度 | 高 |

### B 的劣势

| 劣势 | 说明 | 严重程度 |
|------|------|:---:|
| **gauge 敏感** | A_nk 依赖规范选择, gauge_fix 出错则 A_nk 全错 | 致命 |
| **需全 BZ 积分** | 仅 string 0 的 3 点不够, 需要所有 strings 的所有 k 点 | 高 |
| **公式不一致** | 当前 normalize 方式与 Wilson 不同, 需重新推导 | 高 |
| **内存更大** | ~10× (主要来自 kstring_data_) | 中 |

---

## 五、结论与建议

### 评估: 值得实现作为**验证工具**, 不建议替代方案 A

1. **方案 A (Wilson Loop) 是正确的主力方法** — 规范不变、自动分支处理、已验证三方向等价

2. **方案 B 的独特价值在于交叉验证** — 当 Wilson 出现异常时 (分支不一致、NaN), A_nk 提供独立检查

3. **方案 B 依赖 gauge_fix 质量** — 如果 gauge_fix_smo_anchored 有问题 (已发现 H4 bug), A_nk 结果不可信。Wilson 不受影响

### 实施路径

| 步骤 | 工作量 | 产出 |
|------|:---:|------|
| 1. 拓展 A_nk 到所有 strings (所有 k 点) | 1 天 | 全 BZ A_nk 积分 |
| 2. 统一归一化公式 | 半天 | A_nk 与 Wilson 量级一致 |
| 3. 添加对比输出 | 半天 | Wilson vs A_nk per-direction 报告 |
| 4. 回归测试 (BN, BTO) | 1 天 | 确认两种方法 >95% 一致 |

**总估计**: ~3 工作日。**建议在 Stage 3 收敛通过后实施**。
