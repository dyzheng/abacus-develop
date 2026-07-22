# DeltaP 多k-String 分支选择测试

> 日期: 2026-07-13  
> 对应新增代码: `deltap_wannier.cpp` 中 Cross-String Branch Consistency 诊断输出

---

## 一、问题背景

### 1.1 分支选择的两个层面

| 层面 | 问题 | 难度 | 当前状态 |
|------|------|------|---------|
| **层1 — 单条 k-string 内的相位展开** | Wilson loop 对角化后, 连续 k 点间的特征值相位需要展开 (±2π 消除) | 相对容易 | P0 匈牙利算法已修复 |
| **层2 — 不同 k-string 间的分支选择** | 每条 string 独立展开后, per-atom gamma 可能相差 2π·w^I_n | 困难 | 通过 `prev_gamma` 跨 string 参考选择 |

### 1.2 层2 的核心风险

每条 k-string 独立计算 `gamma_I_per_atom[iat] = Σ_n w_I_n[iat] * gamma_unwrapped[n]`, 其中:
- `w_I_n = |D_I[lm][n]|²` — 由 SMO 投影决定, 相同原子在不同 (ix,iy) 的 k-string 上权重不同
- `gamma_unwrapped[n]` — 特征值相位, 含 unwrapping 的整数倍 2π

当一条 string 的 unwrapped gamma 与相邻 string 的相差 > π 时, 内联分支选择 (H3 修复后) 会搜索 ±2π·w_In 的偏移。但:
1. **String 顺序依赖**: `prev_gamma` 来自上一条 string, 处理顺序影响最终选择
2. **初始值敏感**: 第一条 string 没有参考值, 其分支由 `has_prev_` (deltap_branch.dat) 决定

---

## 二、新增诊断: Cross-String Branch Consistency

### 2.1 诊断输出内容

在 `compute_wannier_polarization` 末尾, 新增逐原子跨 string 统计:

```
=== Cross-String Branch Consistency ===
Strings processed: 4 / 4
Atom 0 per-string gamma: [0] raw=-1.234e-02 sel=-1.234e-02 Δ=0.000e+00 [1] raw=-1.100e-02 sel=-1.100e-02 Δ=0.000e+00 ...
  spread: min=-1.234e-02 max=-1.100e-02 mean=-1.165e-02 σ=5.430e-04
  w_sum^I=5.000e-01 2π·w_sum^I=3.142e+00 OK
```

**关键指标**:

| 指标 | 含义 | 正常值 | 异常信号 |
|------|------|--------|---------|
| `raw` | 分支选择前的 per-atom gamma | — | — |
| `sel` | 分支选择后的 per-atom gamma | — | — |
| `Δ` (= sel-raw) | 分支偏移量 | =0 (未跳分支) 或 =±2π·w_In (跳了一个 branch) | 非整数倍 2π·w_In |
| `σ` | 跨 string 的 sel 标准差 | ≪ 2π·w_sum^I/10 | σ > 2π·w_sum^I/10 → 分支不一致 |
| `2π·w_sum^I` | 单带全偏移对应的 per-atom gamma 跳变 | 原子 I 的参考标度 | — |

### 2.2 判定规则

```
if γ_std > 0.1 * 2π * w_sum^I  →  "BRANCH INCONSISTENT!"  (不同 string 选了不同分支)
else                             →  "OK"                   (分支一致)
```

---

## 三、测试方案

### 3.1 单方向测试 (当前架构支持)

**INPUT 参数**:

```
symmetry            -1         # 强制全 k 网格, 确保所有 k_index_ 在范围内
berry_phase         0
gdir                3          # 极化方向 z
deltap_switch       1
deltap_corr         1
deltap_gdir         3
deltap_gauge_mode   smo_anchored
deltap_lambda_init  0          # 无约束, 仅诊断
deltap_lambda_step  0
```

**k 点网格**: 建议 3×3×3 (产生 27 条 z-string, nppstr_=4, 每 string 3 个 link) 或 2×2×2 (4 条 string, 2 link)

**测试步骤**:

1. 运行 SCF, 检查诊断输出
2. 关注:
   - Δ 列是否都是 0 或整数倍 2π·w_In
   - σ 是否远小于 2π·w_sum^I
3. 若出现 `BRANCH INCONSISTENT`, 检查第一条 string 的 prev_gamma 是否合理

### 3.2 多方向相交测试 (需代码扩展)

**目标**: 检验不同 gdir 方向上的 string 在相交 k 点是否一致。

当前代码每条 string 沿**单一方向** (gdir) 展开。要测试相交 string, 需:

1. **分别运行**: 对 gdir=1, gdir=2, gdir=3 各运行一次 `compute_wannier_polarization`
2. **比较相交点**: 在相交 k-point 处的 gauge-fixed ψ 应当一致
3. **验证路径无关性**: 从 k_a 到 k_b 沿不同 string 累积的相位差应仅差 2π 的整数倍

### 3.3 相交网络设计

对于 3×3×3 k 网格, gdir=3 产生 9 条 z-string:

```
ix=0,iy=0: (0,0,0)→(0,0,1)→(0,0,2)→(0,0,0)
ix=0,iy=1: (0,1,0)→(0,1,1)→(0,1,2)→(0,1,0)
ix=0,iy=2: (0,2,0)→(0,2,1)→(0,2,2)→(0,2,0)
ix=1,iy=0: (1,0,0)→(1,0,1)→(1,0,2)→(1,0,0)
...
```

每条 string 的 `k_index_[istring]` 存储全网格索引。String 在 `iz=0` 处闭合 (首尾 k-point 相同)。

**一致性验证**:
- 对于每条 z-string, `gamma_I_per_atom[iat]` 对应原子 iat 在给定 (ix,iy) 下的总极化
- 不同 (ix,iy) 的 z-string 应有相近的 `gamma_I_per_atom[iat]` (物理上, 宏观极化不依赖于选哪个 (ix,iy))
- σ 是所有 (ix,iy) 的差异的统计量度

---

## 四、当前限制与下一步

### 4.1 架构限制

| 限制 | 说明 |
|------|------|
| 单方向 | 当前 `setup_kstring` 仅创建 gdir_ 方向的 string, 无法同时获取 x/y/z 相交信息 |
| 无 per-k-point gamma | gamma 仅在 string 末尾计算 (累积乘积), 非逐 k 点 |
| 无 HK per-k-point | HK 修正沿 string 累加, 非逐 k 点独立计算 |

### 4.2 所需扩展 (如需要相交网络)

为支持真正的相交 string 网络:

1. **多方向初始 k 点共享**: 不同 gdir 的 string 在同一个 k-point 出发时, 应使用相同的 gauge-fixed ψ 和 eigenphase 排序
2. **Per-k-point Wilson loop**: 沿每个 k-string 在每一步保存 W_j 矩阵的谱, 供相交比较
3. **闭合回路验证**: 沿 (ix,iy,0)→(ix+1,iy,0)→(ix+1,iy+1,0)→(ix,iy+1,0)→(ix,iy,0) 四边回路, 累积相位应为 0

### 4.3 建议的迭代路径

| 步骤 | 任务 | 产出 |
|------|------|------|
| Step 1 | 用 symmetry=-1 跑 3×3×3 的 z-string 单方向测试, 验证当前诊断 | per-string per-atom gamma σ 报告 |
| Step 2 | 在不同方向上分别跑, 对比结果 | 确认 z-only 的 string 选择已正确 |
| Step 3 | 扩展代码支持 "相交 string 网络", 增加回路一致性检查 | 多方向交叉验证能力 |
| Step 4 | 用相交测试发现的分支不一致, 改进跨 string 分支选择算法 | 全局一致的 per-k-point γ |
