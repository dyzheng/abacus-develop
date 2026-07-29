# 2026-07-29 DeltaP 第二阶段修复计划

> 前置：`2026-07-29-deltap-risk-assessment-review.md`（61 项风险评估）
> 第一阶段（已完成）：14 项简单/中等难度修复，7 commits on `feat/deltap`
> 本阶段：剩余 3 项 Critical + 1 项 Important，均为需要设计 + 验证才能动手的复杂问题

---

## 1. 范围与优先级

| 编号 | 问题 | 严重度 | 复杂性 | 建议优先级 | 预计工期 |
|------|------|--------|--------|------------|----------|
| C-02 | 约束力/应力与哈密顿量不自洽 | Critical | 高 | **P0**（阻塞 relax/MD） | 3-5 天 |
| C-05 | gdir≠3 时 HK 修正方向错配 | Critical | 中 | **P0**（阻塞 gdir=1,2） | 1 天 |
| C-11 | 分支平移晶格三套口径不一致 | Critical | 高 | **P1**（影响约束精度） | 2-3 天 |
| C-07 | compute_resta_z MPI 系数错误 | Important | 低 | P2（实验性功能） | 0.5 天 |

> C-16/C-17（berry_connection 路径多个问题）已在第一阶段通过 C-15（默认方法改为 wannier）绕过，建议从计划中移除——仅当有用户明确需要 berry_connection 路径时再修复。

---

## 2. C-02：力/应力与哈密顿量不自洽

### 2.1 问题分解

当前 DeltaP 的约束哈密顿量为两部分之和：

```
H_total = H_HR + H_HK
```

其中：
- **H_HR** (contributeHR, `deltap_lcao.cpp:72-105`)：实空间投影算符部分
  `ΔH_R = Σ_I dλ_I · τ_α(I) · P̂_I`
  其中 `P̂_I = Σ_{μν∈I-shell} |φ_μ⟩⟨φ_ν|` （SMO 壳层投影），`τ_α(I)` 为原子 I 沿 gdir 的分数坐标。
  
- **H_HK** (contributeHk, `deltap_lcao.cpp:295-316`)：k 空间 Berry 联络部分
  `H_sym(ik_L) = (M + M†)/2`，`M = (i/2)·w_eff·S_dk·C_R·C_L†`

力的定义：`F_J = −∂L/∂R_J`，其中 `L = E_KS[ψ] + Σ_I λ_I·(γ_I[ψ] − γ_I_target)`

约束项对力的贡献有三类：

| 来源 | 贡献类型 | 公式 | 实现状态 |
|------|----------|------|----------|
| H_HR 中 P̂ 对原子位置的导数 | Pulay 型 | `−Σ_I λ_I·τ_α(I)·∂Tr(DM·P̂_I)/∂R_J` | **部分实现但有 4 个错误** |
| H_HR 中 τ_α(I) 对 R_J 的导数 | Hellmann-Feynman 型 | `−Σ_I λ_I·(∂τ_α(I)/∂R_J)·Tr(DM·P̂_I)` | **未实现** |
| H_HK 对 R_J 的全导数 | Kohn-Sham + Pulay | `−Σ_I λ_I·∂γ_I/∂R_J` | **未实现** |
| 能量修正 escon 对 R_J 的导数 | — | `∂(escon)/∂R_J = −∂Σ(λ·γ)/∂R_J` | **不计入（escon 仅用于报告）** |

### 2.2 已实现的力/应力代码中的 4 个错误

**错误 1：τ_α 因子缺失**（`deltap_force_stress.hpp:255-260` vs `deltap_lcao.cpp:80-81`）

H_HR 含 `dλ·τ_α(I)`，力用 `λ`（无 τ_α）。
- 修复：力公式中乘 `τ_α(I) = ucell.atoms[T0].tau[I0][alpha_idx]`。

**错误 2：×2 因子无依据**（`deltap_force_stress.hpp:180`）

`force = force * 2.0` 的 "Hermitian conjugate contribution" 注释与模板来源 DeltaSpin（`dspin_force_stress.hpp` 中无此因子）矛盾。
- 修复：需要通过有限差分（FD）验证确定正确因子。

**错误 3：∂τ_α/∂R 项缺失**

`τ_α(I)` 是原子 I 沿 gdir 的分数坐标，`∂τ_α(I)/∂R_J = δ_IJ · ê_α / (lat0·|a_α|)`。
- 力贡献：`F_J (gdir component) = −λ_J · Tr(DM·P̂_J) / (lat0·|a_gdir|)`
- 修复：在力向量的 gdir 分量上加此项。

**错误 4：应力公式量纲错误**（`deltap_force_stress.hpp:337-343`）

`stress[ipol*3+k] += F_alpha * r_vector[k]` 中 r_vector 是整数晶格矢量（无量纲），但应力应为 `F_α · R_cart_β / Ω`。
- 修复：r_vector 需转为笛卡尔坐标：`R_cart = r[0]·a1 + r[1]·a2 + r[2]·a3`，再除以 `Ω`。

### 2.3 最大难点：H_HK 部分的力

H_HK 的贡献无法通过模板复制解决——这是 DeltaP 独有的 Berry 联络算符。严格讲：

```
F_J^HK = −∂⟨H_HK⟩/∂R_J = −Tr(∂DM/∂R_J · H_HK) − Tr(DM · ∂H_HK/∂R_J)
```

其中 `∂H_HK/∂R_J` 涉及 `∂S_dk/∂R_J`（overlap 对原子位置的导数）和 `∂C/∂R_J`（LCAO 系数对位置的导数 = Pulay 力的一部分，已在标准力中计入）。

**绕行方案（推荐，可在短期内实现）**：不解析计算 H_HK 对力的贡献，而是通过**总能的有限差分**来验证力。具体做法：

1. 在 `deltap_corr=1` 的单点 SCF 收敛后，调用 `cal_force`
2. 对每个原子做 ±δ 位移，重新收敛 SCF，计算 `ΔE/ΔR`
3. 比较解析力与 FD 力，差异即为全约束力贡献的缺失部分

如果能量 FD 与解析力的差异在可接受范围（< 5% 或 < 1 mRy/Bohr），可以暂时接受当前力近似，并在文档中声明 "constraint forces are approximate; relax/MD with deltap_corr is experimental"。

### 2.4 修复方案（推荐分步）

**Step 1（1 天）：补 τ_α + ∂τ_α/∂R + 应力修正**

修改 `deltap_force_stress.hpp`：
```cpp
// 在 cal_force_stress 中，计算力时加入 τ_α 因子和 ∂τ_α/∂R 项
double tau_alpha = this->ucell->atoms[T0].tau[I0][gdir];  // 分数坐标
double lam = this->lambda_[iat0];

// Pulay: force1 += lam * tau_alpha * dbb  （已有 dbb 的符号需 FD 确认）
// 加上 ∂τ/∂R 项（仅对 iat0 在 gdir 分量上）：
double dtau_dR = 1.0 / (this->ucell->lat0 * this->ucell->latvec_alpha_norm(gdir));
// force1[gdir] += lam * Tr(DM·P) * dtau_dR  [需从 cal_force_IJR 积累 Tr(DM·P)]
```

应力修正：
```cpp
// r_vector → 笛卡尔，除以 Ω
ModuleBase::Vector3<double> R_cart = r[0] * ucell.a1 + r[1] * ucell.a2 + r[2] * ucell.a3;
R_cart *= ucell.lat0;
stress[ipol*3 + 0] += F_alpha * R_cart.x / ucell.omega;
stress[ipol*3 + 1] += F_alpha * R_cart.y / ucell.omega;
stress[ipol*3 + 2] += F_alpha * R_cart.z / ucell.omega;
```

**Step 2（1 天）：FD 验证**

在测试目录下写 FD 力验证脚本：
1. 对 H2O 单分子（已在约束模式下验证过 dF/dλ）
2. 固定 λ=0.05 Ry，对每个原子做 ±0.005 Bohr FD
3. 收敛单点 SCF，计算 `F_FD = −(E_{+δ} − E_{−δ})/(2δ)`
4. 与 `cal_force` 输出对比，验证 ×2 因子

**Step 3（留待后续）：H_HK 力贡献**

若 FD 验证发现显著偏差（> 5%），需单独分析 H_HK 的力贡献。可考虑：
- 方案 A（精确）：推导 ∂γ/∂R_J 的解析表达式（含 ∂S_dk/∂R 和 Wilson loop 特征向量导数）
- 方案 B（近似）：用 `dF/dλ × dλ/dR ≈ 0` 忽略 HK 部分（当前实际行为）；在文档声明限制
- 方案 C（数值）：在约束模式下用 FD 算力（每次原子位移后重新收敛 SCF + λ），代价是计算量 ×(2·nat) 倍

---

## 3. C-05：gdir≠3 时 HK 修正方向错配

### 3.1 根因

`compute_wannier_polarization`（`deltap_wannier.cpp:287-1545`）对 alpha=0,1,2 三个方向循环，每次调用 `setup_kstring(*kv_)`（344 行）。循环结束后 `k_index_` 保留的是 alpha=2（z 方向）的 string 索引，且 `kstring_data_` 中残留的是**最后一条 string** 的 D_I。

`compute_hk_correction`（1627 行及以下）用 `k_index_[0]`（z 方向 link 对）和当前 `gdir_`（输入值）重新计算 S_dk。当输入 gdir=1 或 2 时，link 对（z 邻居）与 S_dk 位移（x/y 方向）不匹配。

同时也存在 S-01 问题：`kstring_data_` 中的 D_I 属于最后一条 string（可能非 string 0），而 w_eff 权重用的是 `kstring_data_[j].D_I` 其中 j 沿 string 0 的 k 点索引——两者对不上（多 string 网格时）。

### 3.2 修复方案

**Step 1（主修复）：compute_hk_correction 之前重建正确的 k-string 数据**

在 `compute_hk_correction` 开头（约 1596 行），增加：

```cpp
// Rebuild kstring_data_ for the INPUT gdir and string 0.
// compute_gamma_scf left kstring_data_ from the last alpha (gdir=3).
if (gdir_ != 3 && k_index_.empty() == false)
{
    // Re-check: is k_index_ from the correct direction?
    // If gdir_ was restored and k_index_ is from a different setup,
    // we need to re-run setup_kstring and rebuild D_I.
    setup_kstring(*kv_);
    kstring_data_.resize(nppstr_);
    // Rebuild D_I for string 0
    for (int j = 0; j < nppstr_; ++j)
    {
        int ik = k_index_[0][j];
        if (ik >= nks) continue;
        psi->fix_k(ik);
        kstring_data_[j].kvec_d = kv_->kvec_d[ik];
        compute_S_k(j);
        compute_D_I(j, psi->get_pointer(), nbands, nrow);
    }
    // MPI reduce D_I ...
}
```

**优化考虑**：这会在每个 SCF 迭代的 `iter_finish` → `deltap_update_lambda` → `compute_hk_correction` 路径中执行。对于 gdir=3（当前默认），不做额外工作。对 gdir=1,2，额外执行一次 `setup_kstring` + 一条 string 的 S_k/D_I 计算——开销约等于多算 1 条 Wilson string。

**Step 2：w_eff 与 k 点对齐**

随 Step 1 自然解决，因为重建后 `kstring_data_[j]` 对应 `k_index_[0][j]`。

**Step 3（并行优化，可延后）**：缓存三个方向的 kstring_data_ 避免重复计算。

### 3.3 验证

- 对 BN/H2O 分别跑 gdir=1,2,3 的约束测试（deltap_corr=1），验证 γ 收敛到 target
- 三个方向的 λ 应该量级相当（极化率张量元素之间差 ~2× 以内）

---

## 4. C-11：分支平移晶格三套口径不一致

### 4.1 三套口径的证据

| 位置 | 用途 | 平移量 Δγ_I per band phase shift 2π |
|------|------|-------------------------------------|
| `deltap_wannier.cpp:1034-1037` | **γ_I 定义**（本方案） | `2π·w_In(n,I) / w_tot(n)` — per-band 归一 |
| `deltap_wannier.cpp:1138-1142` | Step-3 跨 string 分支匹配 | `2π·w_In(n,I)` — 未归一 |
| `deltap_wannier.cpp:1289` / `1396` / `1457` | 全局 target 搜索 | `2π·w_In(n,I) / w_total(I)` — per-atom 归一 |

其中 `w_tot(n) = Σ_I w_In(n,I)`（≤1，SMO 完备时 =1），`w_total(I) = Σ_n w_In(n,I)`（可达 1.6）。

**物理上正确的平移量**：由 γ_I 的定义直接推出——能带 n 的相位增加 2π，γ_I 的绝对变化 = `2π · (∂γ_I/∂γ_n) = 2π · w_norm(n,I) = 2π · w_In(n,I) / w_tot(n)`。

所以 **Step-3 和全局搜索的平移量都与定义不自洽**。

### 4.2 修复方案

**Step 1：统一为 per-band 归一平移量**

在全局搜索代码（三处 K=5 穷举）和 Step-3 中，统一使用：
```cpp
double shift_per_band = 2.0 * M_PI * w_In_matrix[n][iat] / w_tot_n[n];
// w_tot_n[n] = sum over atoms of w_In_matrix[n][iat]
```

**Step 2：加单元测试验证分支命中正确性**

构造一个确定性场景：
1. 取已知的 Wilson loop 结果（例如 BN 2×2×2 来自存储文件）
2. 手动将第 0 条 band 的 γ 移 ±2π
3. 重新计算 per-atom γ
4. 调用 `select_branch_set`（需先增强为使用 per-band 归一）
5. 验证 `γ_selected − γ_original < 1e-6`

**Step 3：与 zeta rescale 的交互**

Zeta rescale（`deltap_wannier.cpp:1087-1111`）用一个 scale 因子缩放 per-atom γ，使得 Σ γ_I = Σ γ_n（unwrapped sum）。rescaling 之后，平移量也应乘以 scale：
```cpp
double eff_shift = scale * 2.0 * M_PI * w_In_matrix[n][iat] / w_tot_n[n];
```

当前代码中 rescale 发生在 Step-2（zeta rescale）之后、Step-3（branch set）之前，而全局搜索在跨 string 平均之后、zeta 已作用过的值上做平移——所以全局搜索的平移量应该已经是 post-scale 的。需确认 scale 在搜索前的值。

实际上在代码中：
1. 每个 string 内：计算 `gamma_I_per_atom`（无 rescale）
2. Zeta rescale：`gamma_I_per_atom *= scale`（累加到 `gamma_accum`）
3. Step-3 branch-set：用 zeta-rescaled 后的值，平移用未归一 `2π·w_In`
4. 全局搜索：用平均值，平移用 per-atom 归一 `2π·w_In/w_total`

所以 rescale 已经内建在值中，但平移量没有乘 scale——正确做法应是移动 target-aware 搜索到 rescale 之前，或者将平移量也乘上 scale。

**推荐方案**：将 zeta rescale 移到全局搜索之后，即先做原始值上的分支匹配，再统一 rescale。这样平移量定义清晰（未缩放值），且 rescale 是纯标量变换，不改变分支选择结果（因为缩放是正的）。

### 4.3 验证

- 构造分支跳变的回归测试：对已知体系（如 BN 2×2×2），注入人为 2π 跳变，验证 γ 恢复到正确分支
- 与 07-21 diag_minus 发散案例（γ 从 (3.90,3.40)→(−4.59,15.12)）的旧输出文件对比，确认修复后不再出现跳变
- 跑 BN 9-point PES 扫描确认 9/9 点约束收敛（已有测试数据在 tests/deltap_bn_sampling/）

---

## 5. C-07：compute_resta_z MPI 错误

### 5.1 问题

`deltap_wannier.cpp:2130-2134`（MPI 分支）：
```cpp
c_mu = psi_k[n * nrow + lr];
c_nu = psi_k[n * nrow + lc];  // This is wrong for 2D block-cyclic
```

`lr` 是局部行索引（从 `global2local_row(gmu)`），`lc` 是局部列索引（从 `global2local_col(gnu)`）。在 2D 块循环分布中，psi 的本地布局是 `nrow × ncol`，用行索引按**行主序**寻址 `psi_k[n*nrow + lc]` 实际读到的是**某列的本地行**元素，而非列索引对应的列元素。正确的访问方式应通过并行轨道的本地索引映射函数。

### 5.2 修复方案

由于该函数本身是实验性的（仅处理 ik=0、用 Mulliken 近似、注释中有多处 TODO），建议：

```cpp
// 改用正确的 2D 块循环索引
// psi 本地布局: psi_k[local_index] 其中 local_index = irow + icol * nrow
// c_mu 对应 row index lr（本地行），c_nu 对应 col index lc（本地列）
c_mu = psi_k[n * nrow + lr];          // 行索引正确
c_nu = psi_k[n * ncol * nrow + lc];   // 需确认 psi 的列优先索引

// 如果 ABACUS 的 psi 在 2D 块循环中存储整个本地块 (nrow x ncol)，
// 每列的起始偏移是 nrow：psi_k[n * nrow * ncol + lr + lc * nrow]
```

需要确认 ABACUS Psi 的内存布局。若无法迅速确认，可先将该函数整体 `#ifdef __MPI ... #endif` + `WARNING` 输出，限制为串行使用。

### 5.3 优先级说明

此函数仅在 `deltap_method == "wannier"` 下作为后处理（在 wannier 分解之后调用，`deltap_wannier.cpp:46` 行 `compute_resta_z`）。在 SCF 约束路径中从不调用。考虑到：
- 仅处理 ik=0（`if (ik != 0) continue;`）
- 使用 Mulliken 近似而非真正的 Resta-Z 算符
- 输出为诊断信息

建议将其标为实验性功能，在 MPI 下输出 `WARNING` 并跳过，或完全移除 MPI 分支。

---

## 6. 执行顺序与依赖关系

```
Week 1:
  Day 1-2: C-05 (gdir≠3)       ← 无依赖，可独立做
  Day 3-5: C-02 Step 1 (τ_α + ∂τ/∂R + 应力修正) ← 无依赖

Week 2:
  Day 1-2: C-02 Step 2 (FD 验证)  ← 依赖 C-02 Step 1
  Day 3-4: C-11 (分支晶格统一)    ← 依赖 C-05（确保 gdir 正确后再测）
  Day 5:   C-07 (Resta-Z MPI)    ← 最低优先，可不做

Week 3+:   C-02 Step 3 (H_HK 力贡献，如 FD 验证发现显著偏差)
```

关键依赖：
- C-05 必须在 C-11 之前，因为 C-11 的验证需要 gdir≠3 的测试数据
- C-02 Step 2（FD 验证）必须在 Step 1（代码修正）之后，且需要 C-05 修复后全方向覆盖

---

## 7. 每项修复的验收标准

### C-05
- [ ] `deltap_gdir=1,2,3` 各跑一次 BN 约束 SCF，γ 收敛到 target ± 0.01 rad
- [ ] 三个方向的最终 λ 量级相当（同量级，不差数量级）
- [ ] MPI np=2 不 crash

### C-02 Step 1+2
- [ ] H2O 单分子：解析力 vs FD 力差异 < 5% 对所有原子
- [ ] BN 2×2×2：同上
- [ ] 应力输出不再有静默量纲错误（手动验算一个体系的一个应力分量）
- [ ] ×2 因子经 FD 确认为正确值（或移除）

### C-11
- [ ] 注入已知 2π 跳变的单元测试通过
- [ ] BN 9-point PES 扫描 9/9 点约束收敛（|γ-t| < 0.05 rad）
- [ ] diag_minus 案例 γ 不再跳变

### C-07
- [ ] MPI 下不 crash，或有明确 WARNING 说明限制
- [ ] 串行下输出与之前一致

---

## 8. 风险和缓解

| 风险 | 可能性 | 缓解 |
|------|--------|------|
| C-02 FD 验证发现 H_HK 力贡献不可忽略 | 中 | 优先采用绕行方案 B（声明限制）+ 文档标注；若有业务需求再投入方案 A |
| C-05 修复引入性能退化 | 低 | 仅 gdir≠3 触发额外计算；gdir=3（当前默认）路径不变 |
| C-11 统一晶格后约束不再收敛 | 低 | 旧晶格是错误晶格，"收敛"可能是假象；用 FD 验证 γ_target 物理可达 |
| 与 DeltaSpin/future merge 冲突 | 中 | 本阶段不修改 deltaspin 代码（除 spin_constrain_test.cpp 已完成） |

---

## 9. 暂不修复项说明

| 编号 | 问题 | 不修复理由 |
|------|------|------------|
| C-16 | berry_connection 路径闭合双计权 | C-15 已将默认方法改为 wannier，berry 路径需显式指定才触发 |
| C-17 | berry_connection 路径 term2 方向混用 | 同上 |
| C-14/PW | PW per-atom 精度限制 | 这是近似方案 B3 的本质限制，非代码 bug；已在文档 T-15 中标定 |
| S-06 | nspin=2/4 无防护 | 代码改动能量大（涉及输入校验 + deltap_init + psi 索引），建议在文档中声明限制 |
| S-07 | 非正交晶胞无防护 | 同上，与 C-02 的修正耦合（应力修正需要晶格矩阵），可随 C-02 一起部分解决 |

---

## 10. 审核要点

请审核以下策略决策：

1. **C-02 力/应力绕行方案**：同意 "先修正 τ_α/∂τ_α/应力量纲 + FD 验证 ×2 因子 → 声明 H_HK 力近似 → 暂缓严格力" 的路径吗？还是要求精确实现 ∂γ/∂R？
2. **C-05 修复方案**：在 `compute_hk_correction` 中重建 k-string 数据的方案（每次迭代额外 ~1 string 计算开销）是否可以接受？
3. **C-11 晶格统一**：同意以 per-band 归一（`2π·w_In/w_tot(n)`）作为唯一平移口径，并移除 Step-3 和全局搜索中的 per-atom 归一变体吗？
4. **C-07 优先级**：当前函数为实验性诊断输出。是否接受 "MPI 下标 WARNING 跳过" 的处理，还是要求完整修复 2D 块循环索引？
