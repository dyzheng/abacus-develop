# DeltaP 分支选择失效与 SCF 震荡 — 根因分析与改进方案

> 日期: 2026-07-14  
> 上下文: 三方向 Wilson Loop (方案A) 中 Stage 1 和 Stage 2 测试失败

---

## 一、Stage 1 失败症状

### 1.1 BN 2×2×2

| iter | P_x | P_y | P_z | Px=Py=Pz? |
|:---:|:---:|:---:|:---:|:---:|
| 1 | -9.71e-3 | -7.42e-3 | -1.23e-2 | ❌ |
| 2 | **1.25e-2** | **1.25e-2** | **1.25e-2** | ✅ |
| 3 | -1.20e-2 | -1.20e-2 | **-1.34e-2** | ❌ Pz ≠ |

立方对称的 BN 在 iter 3 中 Pz 偏离 Px=Py。差距 1.34e-2 vs 1.20e-2 ≈ 12%。

### 1.2 H₂O 2×2×2

| iter | P_x | P_y | P_z | 物理判断 |
|:---:|:---:|:---:|:---:|:---|
| 1 | 1.80e-2 | **1.80e-2** | 1.57e-2 | ❌ Py 应与 Px 不同 |
| 2 | 1.25e-2 | **2.71e-3** | 1.25e-2 | ✅ Py ≪ Px=Pz |
| 3 | 1.69e-2 | **1.69e-2** | 1.69e-2 | ❌ Py 不应等于 Px |

H₂O 分子在 xz 平面 (y=0.5 for all atoms), 极化应在 xz 面内 → Py 应 ≈ 0。iter 1 和 iter 3 中 Py 被虚假地等化到 Px 和 Pz。

---

## 二、根因分析

### 2.1 Bug 1: 跨 Alpha 的累加器共享

**现象**: iter 2 的 BN 和 H₂O 都显示 Px=Py=Pz, 但这不是物理正确的结果 (H₂O 的 Py 应为 0)。

**根因**: `gamma_accum`、`n_strings_processed` 等累加器在 alpha 循环外声明, 被三个方向共享。

```
当前代码结构 (错误):

  gamma_accum[nat] = {0};         ← 循环外
  n_strings_processed = 0;        ← 循环外
  for alpha = 0..2:
      for each string:
          gamma_accum[iat] += gamma_string   ← 累加所有方向!
          n_strings_processed++
      gamma_I[iat][alpha] = gamma_accum[iat] / n_strings_processed
      ← alpha=0: 只含 x-strings 的 gamma
      ← alpha=1: 含 x-strings + y-strings 的 gamma!
      ← alpha=2: 含全部三个方向的 gamma!
```

alpha=1 的 gamma_I 被 alpha=0 的累加值污染, alpha=2 被 alpha=0+1 污染。当三个方向的 gamma 恰巧相近时 (如 BN 立方对称或 H₂O 稀疏 k 点下的偶然平均), 三个分量"看起来"相等, **但这不是独立测量的结果**——它们是同一次累加的不同快照。

具体影响:

| alpha | gamma_accum 包含 | P_total[alpha] 的物理含义 |
|:---:|------|------|
| 0 | 仅 x-strings | 近似 P_x |
| 1 | x-strings **+** y-strings | 不再是 P_y, 而是 (P_x+P_y)/2 |
| 2 | 全部 | 不再是 P_z, 而是 (P_x+P_y+P_z)/3 |

这就是 H₂O 中 Py 被 "等化" 的原因——alpha=1 和 alpha=2 的累加器包含了其他方向的数据。

### 2.2 Bug 2: 分支状态的跨 SCF 迭代污染

**现象**: SCF 迭代间 gamma 符号跳变 (+1.25e-2 ↔ -1.20e-2)。

**根因**: `W_prev_` (保存到 deltap_branch.dat) 用于初始化下一轮的 `prev_gamma`。但 `prev_gamma` 原本只处理同一个 gdir 方向内的跨 string 分支选择。在三方向代码中:

```
SCF iter 1:
  alpha=0: prev_gamma = NaN (无历史) → 第一条 string 自由选分支
  alpha=1: prev_gamma = NaN (无历史) → 第一条 string 自由选分支
  alpha=2: prev_gamma = NaN (无历史) → 第一条 string 自由选分支
  保存 W_prev_[iat] = gamma_z (最后一个 alpha=2 的结果)

SCF iter 2:
  alpha=0: prev_gamma = W_prev_ (来自 iter-1 的 alpha=2 的 gamma_z!)
           → 用 z 方向的 gamma 作为 x 方向的 branch 参考!
```

不同方向的分支被错误的 prev_gamma 互锁, 导致:
- iter 1 中 alpha=0 和 alpha=2 独立选分支 → 可能选到不同的 2π offset
- iter 2 中 alpha=0 被迫沿用 alpha=2 的分支 → 如果 gamma 波动超过 π, 触发错误的分支校正 → 符号跳变

### 2.3 Bug 3: 跨 String 的 prev_gamma 初始化缺失

**当前**: 每个 alpha 的 `prev_gamma` 初始化为 NaN。第一条 string 使用原始 gamma (无分支校正)。后续 string 的 branch 选择基于第一条 string 的结果。

**问题**: 第一条 string 的随机选择决定了整个 alpha 的 branch 走向。不同 alpha 的第一条 string 有不同的 k 点配置, 它们的 raw gamma 可能符号相反:

```
alpha=0 (x-strings), String 0: raw γ ≈ +0.13 → sel γ ≈ +1.67 (正分支)
alpha=1 (y-strings), String 0: raw γ ≈ -0.04 → sel γ ≈ -1.15 (负分支!)
```

这种跨方向的符号不一致, 在被累加器混合后, 产生符号振荡。

### 2.4 Stage 2 失效: Lambda sweep 中的非确定性

**现象**: λ=0.00→Pz=-1.23e-2, λ=0.05→Pz=+1.54e-2 (符号跳变), 无 dγ/dλ 趋势。

**根因**: 三个 bug 共同作用:

1. **累加器共享** (Bug 1): Pz 实际上包含 Px 和 Py 的混合, 不反映 z 方向的真实极化
2. **跨 SCF 污染** (Bug 2): 每个 λ 值独立运行时, deltap_branch.dat 可能被上一轮污染 (或不存在时从头开始, 初始分支不同)
3. **第一条 string 的随机性** (Bug 3): NaN prev_gamma 导致不同 λ 值锁到不同的 branch 上

这些因素共同导致: 即使 mixing_beta=0 (冻结电荷), 不同 λ 之间的 gamma 也无法建立稳定的 dγ/dλ 关系。gamma 在 ±1e-2 之间随 λ 无规律振荡。

---

## 三、改进方案

### 3.1 修复累加器隔离 (P0, ~10 行)

将 `gamma_accum`、`n_strings_processed`、`zeta_list` 等从 alpha 循环外移入循环内:

```
for alpha = 0..2:
    std::vector<double> gamma_accum(nat_, 0.0);    ← 每个 alpha 独立
    int n_strings_processed = 0;                    ← 每个 alpha 独立
    std::vector<double> prev_gamma(nat_, NaN);      ← 每个 alpha 独立
    
    setup_kstring(*kv_);
    for each string:
        ... 累加只属于当前 alpha ...
    
    gamma_I[iat][alpha] = gamma_accum[iat] / n_strings_processed;  ← 正确的独立值
```

**效果**: Px、Py、Pz 成为三个真正独立的测量, H₂O 的 Py≈0 自然恢复。

### 3.2 跨 SCF 分支持久化 (P0, ~20 行)

`W_prev_` 改为 Vector3, 保存三个方向各自的分支:

```
// W_prev_[iat][alpha] 保存方向 alpha 的历史 gamma
std::vector<ModuleBase::Vector3<double>> W_prev_3d;

save_branch(): 写入 (gamma_x, gamma_y, gamma_z) 每个原子一行
load_branch(): 读取后, 在 alpha 循环内用 prev_gamma = W_prev_3d[iat][alpha]
```

**效果**: 每个方向的分支参考值独立保存和恢复, 不交叉污染。

### 3.3 第一条 String 的分支锚定 (P1, ~15 行)

对每个 alpha, 如果 `has_prev_` 为 true, 用历史 gamma 初始化 prev_gamma:
```
prev_gamma = has_prev_ ? W_prev_3d[iat][alpha] : NaN
```

如果 `has_prev_` 为 false (首次运行), 保持 NaN → 先让第一条 string 自由选, **但记录选中的 gamma**, 在 alpha 循环结束后保存。

**效果**: 首次运行的分支选择稳定后, 后续 SCF 迭代复用。避免每次 SCF 迭代重新随机选分支。

### 3.4 Stage 2 Lambda Sweep 的专用模式 (P2, ~30 行)

为 Stage 2 添加一个 "冻结分支" 模式:

```
if (freeze_branch_) {
    // 对每个 alpha:
    //   从 deltap_branch.dat 加载 prev_gamma
    //   DISABLE 分支选择 (跳过 ±2π·w_In 搜索)
    //   直接使用 zeta-rescaled gamma 不做任何校正
    //   这样 dγ/dλ 只反映 HK 修正的直接效应
}
```

**效果**: Lambda sweep 中的 gamma 变化完全来自 HK 修正, 不受分支选择干扰。dγ/dλ 直接可测。

---

## 四、SCF 震荡的物理机制

独立于上述代码 bug, SCF 本身有物理上的收敛困难:

### 4.1 电荷-Lambda 耦合

```
ρ → H[ρ] → ψ → γ(ψ) → λ → HK(λ) → H' → ψ' → ρ' → ...
     ↑                                              |
     └──────────────── 反馈循环 ─────────────────────┘
```

当 λ 更新 (约束极化) 时:
1. HK 修正改变 H → ψ 改变 → ρ 改变
2. ρ 改变 → Hartree/XC 势改变 → H 改变 → ψ 改变 → γ 改变
3. γ 改变 → λ 更新 (梯度下降) → HK 修正改变 → 回到步骤 1

在固定电荷 (mixing_beta=0) 下, 步骤 2 被切断: ρ 不变, H_XC 不变, 只有 H_HK 随 λ 变。此时 γ→λ 是一个单变量优化问题, 用梯度下降或 CG 应容易收敛。

在自洽电荷 (mixing_beta>0) 下, 步骤 2 激活: ρ 和 λ 同时更新, 形成二维耦合动力学。如果 ρ 和 λ 的响应时间尺度不同:
- ρ 响应快 (mixing_beta 大) → λ 来不及适应 → 振荡
- ρ 响应慢 (mixing_beta 小) → λ 过冲 → 振荡

**改进**: 用 `inner_loop_cooldown` 机制 (已实现) 在 λ 更新后冻结 N 步, 让 ρ 驰豫后再继续优化。

### 4.2 稀疏 k 点的分支不稳定性

2×2×2 仅 8 个 k 点, 每个方向 4 条 string × 2 links = 8 个 O_kpair。Wilson loop 矩阵的相空间覆盖不足, 特征值对 k 点密度敏感。

当 SCF 更新电荷密度时, 波函数微小改变 → O_j 矩阵微小改变 → evals 微小改变 → Hungarian 匹配的 cost 矩阵改变 → 可能切换到不同的 matching → gamma 跳变 → branch 选择判断为 >π → 错误偏移。

**改进**: 
1. 使用 ≥3×3×3 k-mesh (需要更多计算资源)
2. 在 Hungarian 匹配中添加 "close match" 检查: 如果两个匹配的 cost 差异 < ε, 标记为 ambiguous 并保持上次的匹配

---

## 五、实施优先级

| 优先级 | 任务 | 预计行数 | 阻塞 |
|:---:|------|:---:|:---:|
| P0 | 修复累加器隔离 (Bug 1) | ~10 | Stage 1 H₂O 的 Py 等化 |
| P0 | 跨 SCF 分支持久化 (Bug 2) | ~20 | Stage 1 BN iter 3 偏离 |
| P1 | 第一条 string 锚定 (Bug 3) | ~15 | Stage 2 lambda sweep |
| P2 | Stage 2 冻结分支模式 | ~30 | 稳定的 dγ/dλ 测量 |
| P3 | Hungarian close-match guard | ~10 | 密集 k 点前的鲁棒性 |

---

## 六、结论

三方向 Wilson Loop 代码存在两个 P0 级 bug:

1. **累加器跨方向共享** — 导致 P_y 被 P_x+P_z 污染, H₂O 的面外极化被虚假等化
2. **分支状态跨 SCF 污染** — 导致 SCF 迭代间符号跳变, P_total 在 ±1.2e-2 之间振荡

修复后, Stage 1 应能独立验证两个体系的 Px、Py、Pz (H₂O 的 Py≈0 自然恢复), Stage 2 应能测量稳定的 dγ/dλ。修复量约 45 行代码, 可在 1-2 小时内完成。
