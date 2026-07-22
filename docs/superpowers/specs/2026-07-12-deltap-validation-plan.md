# DeltaP 实现有效性验证：完整评估与执行计划 (2026-07-12)

## 背景

当前 DeltaP 内循环 BFGS 约束 SCF 实现已具备：
- HK correction 算符: F = -(i/2)*w_eff*SC（已完成 5 个 bug 修复）
- BFGS inner loop（6/6 单测通过）
- 全 k-串修正 + per-atom preconditioner（B13）
- 并行 MPI（2-proc 验证）
- Löwdin S^{-1/2} 精度 6e-15

**已知局限**：
- HK correction 单向性：只能推 gamma 朝向一个方向，无法双向约束
- 电荷-λ 耦合导致 limit cycle 振荡
- 全靶值 sweep 仅 1/5 target 通过

**但是**：以上局限不影响 core feature——在可收敛方向上，ΔP 应能约束 gamma 到指定 target。验证的核心任务是证明这一点。

---

## 三组测试的目标与判定标准

### Test 1: dγ/dλ 基础响应（2-3h）
**问题**：在固定电荷密度下，λ 变化是否引起 gamma 的单调、线性响应？
**方法**：单电子步 SCF（scf_nmax=2），关闭内循环(nscf=0)，手动设 λ。
**λ 序列**：0, ±0.005, ±0.01, ±0.02, ±0.05（9 点）
**判定通过**：
  - dγ/dλ 符号一致、量级合理（~O(0.1-1.0)）
  - 线性拟合 R² > 0.95
**所需文件**：BN test 66 的 INPUT/STRU/KPT，加上手动 λ 扫描脚本

### Test 2: 完整 SCF target sweep（4-6h）
**问题**：约束 SCF 能否使 gamma 稳定收敛到指定 target？
**方法**：全 SCF（scf_nmax=30），内循环开启(nscf=10)，全串修正+preconditioner。
**靶值**：[γ₀-0.02, γ₀-0.01, γ₀, γ₀+0.01, γ₀+0.02]（仅在 dγ/dλ 可到达的范围内选靶值）
**判定通过**：
  - 可收敛方向上 ≥ 3/5 target 内 gamma 与 target 偏差 < 0.002
  - 不可收敛方向（如有）有明确的诊断信息（如 λ 饱和）
**所需文件**：BN test 66 的 INPUT/STRU/KPT，每靶值一份 INPUT

### Test 3: Wannier90 分原子极化对比（4-6h）
**问题**：DeltaP 的 per-atom Berry phase 分解是否与 Wannier90 的 Wannier center 一致？
**方法**：
  1. 跑 Wannier90 NSCF（使用已有 `towannier90=1` pipeline）
  2. 跑 Wannier90 主程序得到 Wannier centers
  3. 将 WFs 按空间距离分配到原子
  4. 与 DeltaP 的 `r_elec_center[iat]`、`gamma_I[iat]` 对比
**判定通过**：
  - 原子排序一致（哪个原子极化大、哪个小 → 两组方法一致）
  - 分原子 Berry phase 数值接近（差异 < 20%）
**所需文件**：BN test 66 的 INPUT/STRU/KPT，Wannier90 .win 文件，wannier90.x 可执行文件

---

## 验证路径设计

```
         dγ/dλ 响应
              │
      ┌───────┴───────┐
      ▼               ▼
  线性(R²>0.95)    非线性或不单调
      │               │
      ▼               ▼
 完整 SCF sweep    检查 HK correction
      │            公式推导/符号
  ┌───┴───┐
  ▼       ▼
收敛     不收敛
  │       │
  ▼       ▼
Wannier90 调试 limit cycle
对比       (λ mixing, max_step)
```

关键原则：
- **Test 1 是关隘**：如果 dγ/dλ 不线性，一切后续都无意义
- **Test 2 验证约束能力**：在 dγ/dλ 正确的方向上展示 target 可达性
- **Test 3 提供独立验证**：Wannier90 不依赖 DeltaP 的任何公式，是 gold standard

---

## 系统选择

**BN (zinc blende)**，使用 test 66 的配置：
- 立方晶格，8 atoms/cell
- k-mesh: 3×3×3
- 既有 DeltaP(wannier) 又有 Wannier90 预制结果
- 有 Wannier90 .win/.nnkp 可直接复用
- 已知原因导致小 gamma 的缺陷不存在（BN 不是孤立分子）
- Orbital: B `2s2p1d` N `2s2p1d`

---

## 所需代码修改

### Test 1 所需修改（最小）

**文件** `source/source_lcao/module_deltap/deltap_oper.cpp` (或等效位置):
  - 在 `DeltaPOperator::contribute_hk()` 中增加"lambda_override"模式
  - 当 `deltap_lambda_override != 0` 时，跳过 BFGS inner loop，直接用 `lambda_override` 值

**文件** `source/source_io/module_parameter/input_parameter.h`:
  - 增加 `deltap_lambda_override` 参数（默认 999.0 表示禁用）

**更简单方案**（无需代码修改）：
  - 在 sync 模式(`deltap_nscf=0`, `deltap_corr=1`)下
  - 推导 HK correction 的 λ 依赖关系
  - γ 是在 HSolver 之后的 `compute_gamma_scf()` 中计算的
  - Lambda 通过 `DeltaPOperator::contribute_hk()` 影响 Hamiltonian
  - 对固定电荷密度，γ 的变化仅来自 H 的修正

**实际执行方案**：直接在当前代码上测试 —— sync mode + 手动设 `lambda_init`
  1. `deltap_nscf=0`（无 inner loop）
  2. `deltap_lambda_init=<value>`（初始 λ）
  3. `scf_nmax=2`（2步：iter=1 不加修正建电荷，iter=2 用 λ 修正哈密顿量）
  4. 从 iter=2 的 `compute_gamma_scf` 输出读取 γ
  5. 对于 λ=0 baseline，γ 是未扰动值

**需验证的一行**：`deltap_lambda_init` 在 nscf=0 sync mode 下是否被使用。
查看 `esolver_ks_lcao.cpp` iter_finish 中 sync mode branch。

### Test 2 所需修改

**无需修改**。仅需配置 input：
```
deltap_switch   1
deltap_corr     1
deltap_method   wannier
deltap_gdir     3
deltap_nscf     10
deltap_inner_thr    5e-7
deltap_conv_thr     5e-6
scf_nmax        30
scf_thr         1e-8
```
对每个 target gamma 值运行一次完整 SCF。

### Test 3 所需修改

**无需代码修改**。需要：
1. Wannier90 预跑 (`wannier90 -pp bn`)
2. ABACUS NSCF (`towannier90=1`)
3. Wannier90 主跑 (`wannier90 bn`)
4. 后处理脚本：解析 `bn.wout` 中的 WF centers + DeltaP 中的 `deltap_results.dat`

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| dγ/dλ=0（lambda 不影响 gamma） | 中 | 检查 `contribute_hk()` 是否正确连接到 Hamiltonian；可以用有限差分验证 |
| BN gamma 值太小、受噪声污染 | 低 | test 66 已有参考值；如有问题换 BTO(test 18) |
| Wannier90 WF projection 与 DeltaP SMO 不可比 | 中 | 对比的是排序而非绝对值；如果投影根本不同可尝试相同 NAO 子空间 |
| 并行 MPI 引入不一致 | 低 | 使用串行单进程跑 |
| 已有 limit cycle 在 BN 上重现 | 高 | 如果存在，先加 λ mixing（β≈0.3），不影响核心结论 |

---

## 执行清单

### 阶段 0: 环境准备（共 30 min）
- [ ] 确认 `abacus_basic_para` 可运行
- [ ] 确认 `wannier90.x` 可运行（已有确认）
- [ ] 复制 BN test 66 到 `/tmp/opencode/test_bn/`

### 阶段 1: dγ/dλ 响应（共 2-3h）
- [ ] 用 sync mode 跑 λ=0 baseline，记录 γ₀
- [ ] λ sweep (9 点)，每点 scf_nmax=2
- [ ] 拟合 γ=α·λ+β，记录 R² 和 α
- [ ] 判定通过条件

### 阶段 2: 完整 SCF sweep（共 4-6h）
- [ ] 设定 target 序列（在 dγ/dλ 可到达方向选取 3 个 target）
- [ ] 跑 full SCF，每个 target 一次
- [ ] 记录每个 SCF 步骤的 γ 收敛轨迹
- [ ] 判定通过条件

### 阶段 3: Wannier90 对比（共 4-6h）
- [ ] 跑 Wannier90 pre-processing → ABACUS NSCF → Wannier90 main
- [ ] 解析 .wout 的 WF centers，做原子分配
- [ ] 与 DeltaP 的 per-atom gamma/r_elec_center 对比
- [ ] 判定通过条件

### 阶段 4: 文档化
- [ ] 更新 `deltap-development-log.md` 三组测试结果
- [ ] 写 per-round spec 文档

---

## 两个根本性问题（需在 Test 1 前回答）

### Q1: sync mode 下 lambda_init 是否生效？
需要追代码：
- `esolver_ks_lcao.cpp` 初始化时 deltaP_ 如何被设置
- sync mode branch (deltap_nscf=0) 中使用 `current_lambda_value` 还是 `lambda_init`
- `DeltaPOperator::contribute_hk()` 是否有 lambda 依赖

### Q2: 当前分支代码是否已编译？
检查 `abacus_basic_para` 的编译日期和 commit hash。

在写计划文档的同时，让我检查这两个关键点。

---

**结论**：三组测试构成完整验证链路——基础公式正确性 → 约束收敛能力 → 独立第三方验证。Test 1 是关隘，必须优先执行。
