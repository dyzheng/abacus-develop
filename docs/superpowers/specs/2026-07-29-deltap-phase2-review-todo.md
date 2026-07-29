# 2026-07-29 第二阶段修复计划 Review + TODO 指导

> 评审对象：`2026-07-29-deltap-phase2-repair-plan.md`
> 评审人：AI code reviewer
> 结论：**计划总体方向正确，但有 4 处技术细节错误/不完整，3 处流程需要精确化。修正如可执行。**

---

## A. 评审结论（按章节）

### A.1 §2 C-02 力/应力 — 方向正确，2 处需修正

**评审意见：**

1. **§2.2 错误 3 的 ∂τ_α/∂R 公式有误**。计划写 `∂τ_α(I)/∂R_J = δ_IJ · ê_α / (lat0·|a_α|)`，但实际代码中 `tau` 是分数坐标（0..1），而贡献 H_HR 用的是 `atoms[T0].tau[I0][alpha_idx]`。**物理推导**：
   - `τ_α`（分数坐标）= `r_α / (lat0·|a_α|)`，其中 `r_α` 为笛卡尔位置分量
   - `∂τ_α/∂r_α = 1/(lat0·|a_α|)` — 计划中此公式正确
   - **但代码里直接用了 `tau[I0][alpha_idx]` 作为 τ_α 乘到 HR 算符上**——这是**分数坐标**，不是物理位置。HR 算符中 λ·τ_α·P̂ 的 τ_α 是否有物理意义？或者说这是作者引入的一个量纲有误的因子？

   **需要源码确认**：HR 算符中 τ_α 的引入是否有推导依据（`deltap_lcao.cpp:87`），还是纯属"位置权重"的启发式因子？如果纯属启发式，**不应该在力中加入 ∂τ/∂R 项**——那只会放大一个本来就未经推导的近似。

   **TODO（先行确认）**：在 deltap_lcao.cpp 的 git 历史中搜索 tau_alpha 的引入提交，查看是否有设计文档（docs/superpowers/specs/2026-07-09-deltap-operator-derivation.md）中有 τ_α 的依据。若无推导依据，**仅修复错误 1（τ_α 乘到力）即可，不要加 ∂τ/∂R 项**。

2. **§2.3 FD 验证协议不正确**。计划的 FD 方法比较"总能 FD 力" vs "解析力"，但解析力包含所有力的贡献（Kohn-Sham + Ewald + 约束），而"缺失力"只是 ΔP 约束项的一部分。**用总力做 FD 验证无法隔离 ΔP 特异性错误**。正确做法：

   **DeltaP 力 FD 验证协议**：
   ```
   对每个原子 J 的 gdir 分量：
     1. 完整收敛 SCF (λ=λ₀)
     2. 记录 E₀ = etot + dp_escon （即不带约束修正的物理能量）
     3. 将原子 J 位移 +δ，**重新收敛 SCF（保持 λ=λ₀ 不变）**
     4. 记录 E₊
     5. 将原子 J 位移 −δ，重新收敛 SCF（λ=λ₀ 不变）
     6. 记录 E₋
     7. F_FD = −(E₊ − E₋)/(2δ)
     8. 与 cal_force 中 force_deltap 对比（只比较 ΔP 修正部分）
   ```
   注意：**关键约束是 λ 必须冻结**（Δλ=0），否则 FD 能量差包含了 λ 随结构变化的响应，那不是纯力。

3. **§2.2 错误 1 的 Tr(DM·P̂) 不可行**：∂τ/∂R 项需要 `Tr(DM·P̂_I)`（不带导数的投影算符矩阵元）。当前 `cal_force_IJR` 只计算带导数的部分（`nlm1[index+length]·nlm2[index]·dm`），不积累无导数的 `nlm1[index]·nlm2[index]·dm`。**需要在 cal_force_IJR 中加一个额外累加器** `tr_dm_p`：
   ```cpp
   // 在每个 (l,m) 循环中额外累加（放在 dbb 计算之后）：
   double tr_dbb = nlm1[index] * nlm2[index] * dm_pointer[step_trace[is]];
   *tr_dm_p += tr_dbb;  // tr_dm_p 是 cal_force_IJR 新增的输出参数
   ```
   然后在 `cal_force_stress` 中：
   ```cpp
   double dtau_dR = 1.0 / (this->ucell->lat0 * a_norm);
   force_local(iat0, gdir) += lam * tr_dm_p * dtau_dR;
   ```

4. **§2.2 错误 2 的 ×2 因子——计划做法正确但优先级应提高**。FD 验证前必须先修正错误 1 和 3，否则 FD 力（正确物理）与解析力（缺 τ_α 和 ∂τ/∂R）之间的偏差会淹没 ×2 因子的判断。

---

### A.2 §3 C-05 gdir≠3 — 方向正确，伪代码有 3 处 bug

**评审意见：**

1. **伪代码条件判断不正确**。计划写 `if (gdir_ != 3 && ...)`，但这个条件在 gdir=3 且 `kstring_data_` 已经被 gamma 计算填充时**永远为 false**，但 gdir=3 时 `kstring_data_` 中仍然是最后一条 string 的 D_I（S-01 问题），并非 string 0。**无论 gdir 值是多少，kstring_data_ 在 compute_gamma_scf 之后都包含错误数据**（最后一条 string，来自最后一个 alpha 循环）。

   **正确条件**：需要追踪 `kstring_data_` 属于哪个 gdir 和哪个 string。添加成员变量：
   ```cpp
   // deltap.h 私有成员
   int kstring_gdir_ = -1;  // which direction kstring_data_ belongs to (-1 = invalid)
   int kstring_string_ = -1; // which string index (-1 = invalid)
   ```
   在 `compute_wannier_polarization` 的 string 循环中设置这些标记，在 `compute_hk_correction` 中检查 `kstring_gdir_ != gdir_ || kstring_string_ != 0` 作为重建条件。

2. **伪代码变量顺序错误**。计划中第 160 行用了 `nks`（第 160 行 `if (ik >= nks) continue;`），第 169 行用了 `nrow`——但这两个变量在 `compute_hk_correction` 中是在 `S_dk` 检查之后（当前代码 1606-1608 行）才声明的。**需要把变量声明移到重建代码之前**。

3. **缺少 MPI Allreduce**。重建 D_I 后必须做 `MPI_Allreduce`（与 `deltap_wannier.cpp:424-441` 中现有代码相同的模式），否则 D_I 只包含本地行的部分和。

4. **重建范围不完整**。计划中只重建了 `S_k` 和 `D_I`，但 `compute_hk_correction` 的 `w_eff` 计算用 `kstring_data_[j].D_I`（1650 行），S_dk 在 1604 行重建。但**还需要重建 kvec_d**（1673 行使用了 `kstring_data_[j].kvec_d`）。计划中遗漏了 `kstring_data_[j].kvec_d = kv_->kvec_d[ik]`——实际上这行在伪代码里有（167 行），但如果只重建 string 0，其他 string 的 kvec_d 为旧值——**不过 hk_correction 只用 string 0，所以 kvec_d 只需对 string 0 正确**。

---

### A.3 §4 C-11 分支晶格 — 分析有深度，但忽略了 Step-3 连续性

**评审意见：**

1. **计划将 zeta rescale 移到全局搜索之后——正确，但遗漏了 Step-3**。当前代码流程：
   ```
   每 string:
     γ_raw = Σ w_norm·γ_unw        (no rescale)
     zeta rescale: γ *= scale       (Step 2)
     Step-3 branch-set: 用未归一 2π·w_In 平移  (Step 3)
     累加到 gamma_accum
   全局搜索: 用 per-atom 归一 2π·w_In/w_total 平移
   ```

   计划建议：rescale 移到全局搜索后。但 Step-3 在 string 内做分支连续性检查（`prev_gamma` 来自上一 SCF 迭代的 zeta-rescaled 值）。如果把 rescale 移到全局搜索后，Step-3 就要在**未 rescale 的值**上做分支匹配，而 `prev_gamma` 是 rescaled 值——**量纲不匹配**。

   **正确做法**：rescale 移到全局搜索之后，同时 Step-3 也在未 rescale 值上做，prev_gamma 改为存储**未 rescale 的原始值**（`gamma_raw_accum` 已有此数据，代码中 1070-1075 行已经在积累了）。

2. **推荐方案的替代方案（更简单）**：既然 zeta rescale 是纯标量乘法（scale > 0），它不改变分支选择的最优解（因为分支搜索是按 |γ − target| 最小化，正标量不改变排序）。所以**不需要移动 rescale**，只需让三套平移量统一为 per-band 归一，且乘上当前的 scale 因子即可。这比改动流程顺序风险更小。

   **TODO**：在代码中搜索 `scale` 变量的作用域，确认在全局搜索的位置它仍然有效。当前代码中 `scale` 在 Step-2 的 if 块内定义（1097-1111 行），是局部变量。需要把它提升到函数作用域（在 string 循环外定义）以便全局搜索使用。

3. **计划中 per-band 归一的分母 `w_tot(n)` 需要明确获取方式**。`w_In_matrix` 已经在 `deltap_wannier.cpp:1013` 处计算，其定义为 `w_In_matrix[n][iat]`。`w_tot(n) = Σ_{iat} w_In_matrix[n][iat]`。这在代码中可以现场计算（每 string 都已存在）。**计划中没写清楚在哪个作用域能拿到 `w_In_matrix`**——全局搜索在 string 循环之后，此时 `w_In_matrix` 是最后一个 string 的值（非 string 0）。应使用 `w_In_first_string_`（1080 行已保存）。

---

### A.4 §5 C-07 — 计划合理但验收标准需收紧

**评审意见：**

计划中建议在 MPI 下 WARNING 跳过。但更彻底的做法是：**直接删掉 MPI 分支中那两行错误的 c_mu/c_nu 读取**，让 MPI 和串行走同一个代码路径（但结果在 MPI 下仍然不正确，因为 psi 是分布式的）。

验收标准应改为：
- [ ] MPI 下 `compute_resta_z` 输出 WARNING("compute_resta_z not implemented for MPI, skipping") 并提前返回
- [ ] 串行下行为不变（回归测试）
- [ ] 在函数注释中标记为 `experimental / serial-only`

---

## B. 具体 TODO 清单（按执行顺序）

### TODO-1 [C-05, 1 天]

**目标**：修复 gdir≠3 时 hk_correction 方向错配 + w_eff 用错 string

**修改文件**：`source/source_lcao/module_deltap/deltap.h`、`deltap_wannier.cpp`

**步骤**：

1. **`deltap.h`**：在私有成员区添加
   ```cpp
   int kstring_gdir_ = -1;   // gdir of last compute_S_k/D_I fill (-1 = none)
   int kstring_string_ = -1; // string index of last fill (-1 = none)
   ```

2. **`deltap_wannier.cpp` `compute_wannier_polarization`**：在 string 循环（约 403 行）末尾设置标记：
   ```cpp
   kstring_gdir_ = gdir_;
   kstring_string_ = istring;
   ```

3. **`deltap_wannier.cpp` `compute_hk_correction`**：在 nrow==ncol 守卫之后（约 1619 行），插入重建逻辑：
   ```cpp
   // Rebuild S_k/D_I if they belong to a different direction or string.
   // compute_gamma_scf leaves kstring_data_ from the last alpha (gdir=3)
   // and the last string; we need the INPUT gdir and string 0.
   if (kstring_gdir_ != gdir_ || kstring_string_ != 0)
   {
       setup_kstring(*kv_);
       kstring_data_.assign(nppstr_, KSpaceData());
       for (int j = 0; j < nppstr_; ++j)
       {
           int ik = k_index_[0][j];
           if (ik >= nks) continue;
           kstring_data_[j].kvec_d = kv_->kvec_d[ik];
           psi->fix_k(ik);
           compute_S_k(j);
           compute_D_I(j, psi->get_pointer(), nbands, nrow);
       }
       kstring_gdir_ = gdir_;
       kstring_string_ = 0;
       // MPI Allreduce D_I (same pattern as deltap_wannier.cpp:424-441)
#ifdef __MPI
       for (int j = 0; j < nppstr_; ++j)
           for (int iat = 0; iat < nat_; ++iat)
           {
               int r = nproj_per_atom_[iat];
               for (int lm = 0; lm < r; ++lm)
               {
                   if (kstring_data_[j].D_I.size() <= (size_t)iat) continue;
                   if (kstring_data_[j].D_I[iat].size() <= (size_t)lm) continue;
                   int sz = kstring_data_[j].D_I[iat][lm].size();
                   if (sz > 0)
                       MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                     2*sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
               }
           }
#endif
   }
   ```

4. **验证**：对 H2O 分别用 gdir=1,2,3 跑 `deltap_corr=1` SCF，检查 γ 是否收敛到 target。MPI np=2 跑一次确认不 crash。

**commit 信息**：
```
fix(C-05,S-01): rebuild kstring_data_ for correct gdir/string in compute_hk_correction
```

---

### TODO-2 [C-11, 2 天]

**目标**：统一分支平移晶格为 per-band 归一 + 正确处理 zeta scale

**修改文件**：`source/source_lcao/module_deltap/deltap_wannier.cpp`

**步骤**：

1. **在 string 循环外定义 `double current_zeta_scale = 1.0;`**（提升到函数作用域）

2. **Step-2 zeta rescale 时保存 scale**：在 `deltap_wannier.cpp:1104`（`scale = gamma_unw_sum / gamma_raw_sum;`）后添加
   ```cpp
   current_zeta_scale = scale;
   ```

3. **Step-3 平移量改为 per-band 归一**（1138-1142 行）：
   ```cpp
   // Before (WRONG - unnormalized):
   // double candidate = g + sign * 2.0 * M_PI * w_In;
   // After (per-band normalized):
   double w_tot_n = 0.0;
   for (int j = 0; j < nat_; ++j) w_tot_n += w_In_matrix[n][j];
   if (w_tot_n < 1e-12) continue;
   double candidate = g + sign * 2.0 * M_PI * w_In_matrix[n][iat] / w_tot_n;
   ```

4. **全局搜索三处（1289, 1396, 1457 行）统一为 per-band 归一 + zeta scale**：
   ```cpp
   // 从 w_In_first_string_ 取 w_In
   double w_tot_n = 0.0;
   for (int iat2 = 0; iat2 < nat_; ++iat2)
       w_tot_n += w_In_first_string_[n][iat2];
   if (w_tot_n < 1e-12) continue;
   shift_amps.push_back(current_zeta_scale * 2.0 * M_PI
                        * w_In_first_string_[n][iat] / w_tot_n);
   ```
   **注意**：三处代码结构略有不同（约束矩阵模式 / total 模式 / per_atom 模式），需分别定位修改。约束矩阵模式（1289 行）已有 `w_total`（per-atom），改为 `w_tot_n`。

5. **新增单元测试** `deltap_branch_shift_test.cpp`：
   ```cpp
   TEST(BranchShiftTest, PerBandNormalizedShift)
   {
       // 构造已知 Wilson loop 结果
       // 手动将 band 0 的相位 +2π
       // 验证 select_branch_set 恢复到原值
   }
   ```

6. **验证**：跑 BN 9-point PES 扫描（已有数据在 tests/deltap_bn_sampling/），确认 9/9 点 |γ−t| < 0.05 rad 不劣化。

**commit 信息**：
```
fix(C-11): unify branch shift lattice to per-band normalized amplitudes
```

---

### TODO-3 [C-02, 3 天]

**目标**：修力/应力的 3 个可修错误 + FD 验证 + 声明 H_HK 限制

**前置**：TODO-1（C-05）完成——FD 验证需要全方向正确

**Step 1（1 天）：修正 3 个错误**

修改 `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`：

1. **错误 1（τ_α）**：在 `cal_force_stress` 的 iat0 循环中（第 37 行），加：
   ```cpp
   const int alpha_idx = this->gdir_ - 1;
   double tau_alpha = this->ucell->atoms[T0].tau[I0][alpha_idx];
   ```
   将 `lam` 的传递改为 `lam * tau_alpha`（传参给 cal_force_IJR 和 cal_stress_IJR）。

2. **错误 3（∂τ/∂R）**——**先确认 HR 算符中 τ_α 是否有推导依据**：
   - 查 `git log --oneline -- source_lcao/module_operator_lcao/deltap_lcao.cpp` 中 tau_alpha 的引入提交
   - 查 `docs/superpowers/specs/2026-07-09-deltap-operator-derivation.md` 是否有 τ_α
   - **若有依据**：在 cal_force_IJR 中加 `tr_dm_p` 累加器（见 A.1-3），在 cal_force_stress 中加 ∂τ/∂R 力项
   - **若无依据**：**不要加 ∂τ/∂R**——只修 τ_α 因子，力公式保持 `λ·τ_α·(∂P̂)·DM` 即可

3. **错误 4（应力量纲）**：cal_stress_IJR 中 r_vector 转笛卡尔：
   ```cpp
   ModuleBase::Vector3<double> R_cart(
       r_vector[0] * this->ucell->a1.x + r_vector[1] * this->ucell->a2.x + r_vector[2] * this->ucell->a3.x,
       r_vector[0] * this->ucell->a1.y + r_vector[1] * this->ucell->a2.y + r_vector[2] * this->ucell->a3.y,
       r_vector[0] * this->ucell->a1.z + r_vector[1] * this->ucell->a2.z + r_vector[2] * this->ucell->a3.z);
   R_cart *= this->ucell->lat0;
   stress[ipol*3+0] += F_alpha * R_cart.x / this->ucell->omega;
   stress[ipol*3+1] += F_alpha * R_cart.y / this->ucell->omega;
   stress[ipol*3+2] += F_alpha * R_cart.z / this->ucell->omega;
   ```
   注意：`ucell->a1` 是 `ModuleBase::Vector3<double>`，直接用 `.x` `.y` `.z` 访问。

**Step 2（1 天）：FD 验证 ×2 因子**

写 FD 验证脚本（可放在 `tests/deltap_fd_force/`）：
```
对每个原子 J 的 gdir 分量:
  1. SCF 收敛 (λ=λ₀ 固定), 记录 E₀ = etot + dp_escon
  2. R_J[gdir] += δ → 重新收敛 SCF (λ=λ₀ 不变), 记录 E₊
  3. R_J[gdir] -= δ → 重新收敛 SCF (λ=λ₀ 不变), 记录 E₋
  4. F_FD = -(E₊ - E₋) / (2δ)
  5. 对比 cal_force 输出中 force_deltap(J, gdir)
```
验证 ×2 因子：将 `force * 2.0` 分别改为 `force * 1.0` 和 `force * 2.0` 各跑一次，哪个与 FD 力一致用哪个。

**Step 3（半天）：声明 H_HK 限制**

在 `deltap_force_stress.hpp` 头部注释中写明：
```cpp
/**
 * LIMITATION: This force/stress only covers the real-space projector
 * (H_HR) contribution.  The k-space Berry-connection part (H_HK)
 * does not have an analytic force contribution.
 * Relax/MD with deltap_corr is therefore experimental.
 */
```

在输入参数文档（read_input_item_other.cpp）的 `deltap_corr` description 中加：
> "Constraint forces are approximate (missing H_HK contribution); relax/MD is experimental."

**Step 4（验证）**：H2O + BN 各跑一次 cal_force，确认无 crash、力分量有限。

**commit 信息**：
```
fix(C-02): force/stress tau_alpha factor, stress Cartesian conversion, FD validation
```

---

### TODO-4 [C-07, 0.5 天]

**目标**：MPI 下标 WARNING 跳过 compute_resta_z

修改 `source/source_lcao/module_deltap/deltap_wannier.cpp`：

在 `compute_resta_z` 函数开头（约 1919 行）添加：
```cpp
#ifdef __MPI
    if (GlobalV::MY_RANK == 0)
    {
        std::cerr << "WARNING: compute_resta_z is not implemented for MPI, "
                     "results will be incorrect (experimental, serial-only)\n";
    }
    ModuleBase::timer::end("DeltaP", "compute_resta_z");
    return;
#endif
```

同时在函数注释中添加 `/// @warning Experimental feature, serial-only`。

**commit 信息**：
```
fix(C-07): skip compute_resta_z under MPI with warning
```

---

## C. 执行时间线

```
Day 1:  TODO-1 (C-05)        — 1 commit
Day 2:  TODO-2 (C-11)        — 1 commit
Day 3:  TODO-3 Step 1 (τ_α + ∂τ/∂R 确认 + 应力) — 1 commit
Day 4:  TODO-3 Step 2 (FD 验证 ×2)             — 1 commit
Day 5:  TODO-3 Step 3+4 (声明 + 验证)           — 1 commit
Day 6:  TODO-4 (C-07)        — 1 commit
Day 7:  dev log 更新 + 总回归测试
```

---

## D. 对计划文档的修订建议

建议在 `2026-07-29-deltap-phase2-repair-plan.md` 中补充：

1. §2.2 错误 3 处添加注记："τ_α 的引入依据需先源码确认；若无推导，仅修 τ_α 因子，不加 ∂τ/∂R"
2. §2.3 FD 协议改为 λ 冻结的 delta-force 验证（见 A.1-2）
3. §3.2 伪代码条件改为 `kstring_gdir_ != gdir_ || kstring_string_ != 0`（不依赖 gdir 值判断）
4. §4.2 移除 "将 zeta rescale 移到全局搜索后" 的建议，改为 "在全局搜索中使用 `current_zeta_scale` 因子"（流程不变，风险更小）
5. §4.2 明确 `w_In_matrix` 应取 `w_In_first_string_`（第一个 string 的缓存，非最后一个 string 的残留）
6. §7 C-07 验收标准收紧为 "WARNING + 提前返回 + 函数注释标 experimental/serial-only"
