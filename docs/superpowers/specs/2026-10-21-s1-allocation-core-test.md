# S1 分配算子核心与网格观测：测试记录

**日期**：2026-10-26
**阶段**：S1（统一局域磁矩分配执行方案）
**结论**：PASS。S1.1–S1.3 全部单元测试通过；`max_i ||Σ_i A_i x−x|| =
3.109e-15`；空目标集与 Becke 读数一致到 `1e-13`；`n_c >= 0`；
目标基条件数 `cond = 5.615422`，低于 `1e6` 闸门。

## 1. 测试计划

1. 验证无 Eigen 的 R×R Jacobi 对称特征分解满足
   `A = V diag(w) V^T`、`V^T V = I`，且 `S^{-1/2} S S^{-1/2} = I`。
2. 验证网格分配算符 `A_i = Q_i + Qc W_i Qc` 的求和规则、正定性、空目标集与
   完备目标集两个插值极限，以及稠密定义逐元一致性。
3. 验证三量分解 `n = n_T + n_c`、`m = m_T + m_c`、`Σn = N`、`Σm = M`、
   `n_c(r) >= 0`，并确认空目标集退化为既有 Becke 读数。

## 2. 测试设置

- 构建：`build/`（CMake debug，`ENABLE_MPI=ON`）。
- 网格后端测试体系：20 Bohr 立方 H2O，O `(10,10,10)` Bohr，
  H `(11.2,10,10)` / `(8.8,10,10)` Bohr。
- 主要网格：`24^3`（13,824 点）；完备/稠密极限用例：`4^3`（64 点）。
- 目标函数：O/H1 上 sigma=1 Bohr 的三维高斯；完备极限用每格点一个归一化
  delta 函数。
- 权重：Becke，半径 `(1.5,0.5,0.5)` Bohr，约束映射显式固定为
  `{{0},{1},{2}}`。
- 观测 toy state：up = O 上高斯；down = H2 上高斯（H2 不在目标子空间内，
  因此互补自旋矩非零）。
- 命令：

```bash
cmake --build build -j 32 --target \
  MODULE_ESTATE_constraint_alloc_linalg \
  MODULE_ESTATE_constraint_alloc_operator \
  MODULE_ESTATE_constraint_alloc_observe
ctest --test-dir build -R '^MODULE_ESTATE_constraint_alloc_(linalg|operator|observe)$' -V
ctest --test-dir build -R constraint --output-on-failure
cmake --build build -j 32 --target abacus_basic_para
```

## 3. 结果

### 3.1 S1.1 线性代数

`MODULE_ESTATE_constraint_alloc_linalg`：4/4 PASS（0.00 s）。

- `SymEigReconstruction`：重建与正交性误差 `< 1e-10`，特征值升序。
- `InvSqrt`：`S^{-1/2} S S^{-1/2}=I` 误差 `< 1e-9`。
- `RankTruncationDropsNullDirection`：`diag(4,0)` 的零方向未被求逆。
- `EmptyAndScalarInputs`：`n=0/1` 边界稳定。

### 3.2 S1.2 分配算子

`MODULE_ESTATE_constraint_alloc_operator`：3/3 PASS（0.04 s）。

```text
alloc_operator: ntarget=2 nrxx=13824 cond=5.615422e+00 build=0.551 ms
alloc_operator audit: max_sum_error=3.109e-15 min_q=2.292484e+03
```

| 项目 | 判据 | 实测 |
|---|---:|---:|
| `Σ_i A_i x` 对任意网格函数 `x` | `< 1e-10` | `3.109e-15` |
| 每个 `A_i` 的二次型 `⟨x|A_i|x⟩` | `>= 0` | 最小 `2.292484e+03` |
| 目标基条件数 | `< 1e6` | `5.615422` |
| 空目标集 `A_i=diag(w_i)` | `1e-12` | PASS |
| 完备正交目标集 `A_i=Q_i` | `1e-10` | PASS |
| 稠密定义 `Q_i+Qc W_i Qc` 对拍 | `1e-10` | PASS |
| 2-target 构建耗时 (`24^3`) | 记录 | `0.551 ms` |

### 3.3 S1.3 三量观测

`MODULE_ESTATE_constraint_alloc_observe`：2/2 PASS（0.02 s）。

```text
alloc_observe: cond=5.615422e+00
  sum_n=4.489690342809e-02 / 4.489690342809e-02
  sum_m=5.826800748409e-08 / 5.826800746066e-08
  m_c=(-2.093327e-03 -7.297238e-04 -5.968792e-03)
```

| 项目 | 判据 | 实测 |
|---|---:|---:|
| `Σn − N_ref` | `< 1e-8` | 机器精度内相等 |
| `Σm − M_ref` | `< 1e-8` | `2.34e-16` |
| `n_tot = n_T+n_c` 每原子 | `< 1e-10` | PASS |
| `m_tot = m_T+m_c` 每原子 | `< 1e-10` | PASS |
| `n_c(r)` 最小本征值 | `>= -1e-14` | PASS |
| 空目标集 vs Becke 直接积分 | `< 1e-13` | PASS |
| `m^c` 量级（toy state） | 记录 | `-5.97e-3` 至 `-7.30e-4` |

### 3.4 回归与生产构建

```text
ctest -R constraint: 16/16 PASS, 49.45 s
abacus_basic_para: build and link PASS
```

包含既有 13 个约束测试、S0 两个新用例，以及 S1 三个新目标；MPI
`weight_grid_mpi` / `_4np` 均通过。S1 新模块在 `abacus_basic_para` 的
`elecstate` 目标中成功编译链接。

## 4. 分析

- `AllocOperator` 先对 `S_T = C^T C` 做 R×R 对称正交化，再在正交目标上构造
  `P` 与 `Qc`。求和规则误差为 `3.1e-15`，说明 `Q_i + Qc W_i Qc` 的运算顺序
  和 `Qc^2=Qc` 在离散网格上成立。
- 空目标集路径按定义返回 `A_i=diag(w_i)`，与直接 Becke 积分一致，保证 S2
  生产钩子可以在“默认无目标轨道”时零影响。
- toy state 的 down band 放在 H2（目标集外），因此 `m_c` 为 `10^-3` 量级，
  可验证互补通道不是零占位；该数值不是物理结论，只用于接口/账目测试。
- `n_c(r)` 由残差范数直接累加，未通过 `n_tot-n_T` 相减，因此正性不依赖抵消。
- S1 仅验证网格原型；LCAO/PW projector 重叠、生产 INPUT 开关和默认输出
  逐字节不变仍属于 S2。

## 5. 下一步

1. 进入 S2.1：从 NAO 轨道与 AO 重叠 `S` 构造 `S_T=C^TSC` 与 LCAO 三量
   投影，完成小基组 DMR 对拍。
2. S2.2：新增默认 False 的 `constraint_alloc_report`，在 LCAO 密度矩阵可用
   处输出 `ALLOC` 审计行；先验证关闭时日志逐字节不变。
3. 在 h2o_dimer LCAO 上运行 warm relax 两帧，记录 `m_T/m_c` 与 `onsite=`
   对照表；未达到 S2 闸门前不进入 S3 约束势注入。
