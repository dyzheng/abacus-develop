# PW DFTU 迁移代码审查报告

> **审查日期**: 2026-04-19
> **审查分支**: `feat/dftu-pw-port`
> **对比基准**: `zdy-tmp` (`/root/abacus-zdy-tmp`)
> **审查者**: Hermes Agent (Nous Research)

---

## 一、调试文档审查

### 1.1 DEBUG_PROGRESS.md 评估

**优点**:
- 关键数据表格清晰，iter=1 ik=1 的第 4/5 次 Hψ 调用范数对比直接定位了分叉点
- "已排除"列表很有价值，避免了在已被证伪的假设上浪费时间
- 推论" H 算子本身没问题，问题出在 Davidson 子空间迭代产生的 trial vector"是合理的

**改进建议**:
- 建议补充 iter=2 的完整日志（包括 preconditioner 输出），因为 iter=1 的分叉最终恢复了，而 iter=2 没有
- 建议补充 preconditioner 数组的前几个值对比（zdy-tmp vs pw-port）
- 建议补充 `nbase`（子空间维度）和 `notconv`（未收敛 band 数）在 iter=1 和 iter=2 之间的变化

### 1.2 TEST_STATUS.md 评估

**状态文档已过时**: TEST_STATUS.md 中记录的"eff_pot_pw 10^51 垃圾值"问题已被后续修复解决（见 DEBUG_PROGRESS.md 的"已排除"列表）。建议：
- 更新 TEST_STATUS.md 以反映当前最新状态
- 将 DEBUG_PROGRESS.md 和 TEST_STATUS.md 合并为单一文档，避免信息不一致

---

## 二、核心代码修改审查

### 2.1 `source/source_lcao/module_dftu/dftu.cpp`

**审查结果**: ✅ 基本正确

| 检查项 | 状态 | 说明 |
|--------|------|------|
| `orbital_corr[it] == -1` 防护 | ✅ 一致 | 与 zdy-tmp 完全一致 (zdy-tmp: L69-72, pw-port: L92-95) |
| eff_pot_pw 分配逻辑 | ✅ 一致 | nspin=2 时 `pot_index *= 2` 正确 |
| API 适配 | ✅ 正确 | `GlobalV::NSPIN` → `PARAM.inp.nspin`, `GlobalV::NPOL` → `PARAM.globalv.npol` |
| 内存管理 | ⚠️ 注意 | `ucell` 指针仅在 `#ifdef __LCAO` 下赋值，但 PW 路径不调用 LCAO 方法，可接受 |
| `Plus_U::energy_u` 静态变量 | ⚠️ 注意 | 静态变量在多线程/多实例场景下可能有问题，但 ABACUS 当前是单实例模式 |

**发现的问题**:

1. **[P2] 缩进不一致**: 第 406 行 `}` 缩进不匹配（少了 4 个空格），但仅影响代码风格。
   ```cpp
   // 第 404-406 行
       if (this->uramping < 0.01) {
           return;
   }  // ← 此处应缩进到与 if 同级
   ```

2. **[P2] 冗余的 `#include <vector>`**: 第 20 行 `#include <vector>` 在新代码中是必要的（静态成员声明使用了 `std::vector`），不是冗余。✅ 保留正确。

### 2.2 `source/source_lcao/module_dftu/dftu_pw.cpp`

**审查结果**: ⚠️ 有潜在问题

| 检查项 | 状态 | 说明 |
|--------|------|------|
| cal_occ_pw 核心逻辑 | ✅ 一致 | nspin=2/4 的 locale 计算与 zdy-tmp 完全一致 |
| spin 判定逻辑 | ✅ 一致 | `ik >= psi_p->get_nk()/2` 判定 spin-down 正确 |
| uom_array 索引 | ✅ 一致 | nspin=2 时 spin-down 偏移 `size` 正确 |
| VU 矩阵计算 | ✅ 一致 | `diag_coeff` 和 `weight_eu` 与 zdy-tmp 一致 |
| `initialed_locale = false` | ✅ 正确 | 与 zdy-tmp 一致，每次调用后重置 |

**发现的问题**:

1. **[P1 - 关键] `cal_occ_pw` 中的 `copy_locale` 调用时机**:
   - 第 17 行 `this->copy_locale(cell);` 在 `initialed_locale == false` 检查之前调用
   - 当 `initialed_locale == false` 时（SCF 的前几个 iter），`copy_locale` 将 `locale` 复制到 `locale_save`，但紧接着第 21 行 `zero_locale(cell)` 又将 `locale` 清零
   - **这在逻辑上是正确的**（保存旧的 locale 用于 mixing），但存在冗余操作。zdy-tmp 中也是这样写的，所以保持一致。

2. **[P1] 诊断代码未受编译宏保护**: 大量 `std::cout` 诊断打印未用 `#ifdef __DEBUG` 保护，会严重影响生产环境性能。
   - 建议：将所有诊断打印包裹在 `#ifdef __DFTU_DEBUG` 或类似宏中

3. **[P2] 第 314-341 行的诊断打印格式不一致**: 有些 `if(iter <= 2)` 没有大括号，导致后续多行语句逻辑混乱：
   ```cpp
   if(iter <= 2)                    // 只控制下一行
   std::cout << "...";              // 这行受 if 控制
   if(iter <= 2)                    // 新的 if
   std::cout << "...";              // 这行受新 if 控制
   if(iter <= 2 && this->uom_array.size() > 0) {  // 有括号
       ...                          // 多行受 if 控制
   }
   ```
   虽然语法上正确（C++ 的 if 无括号时只控制一条语句），但可读性差且容易出错。

### 2.3 `source/source_pw/module_pwdft/op_pw_proj.cpp`

**审查结果**: ⚠️ 有重要发现

| 检查项 | 状态 | 说明 |
|--------|------|------|
| OnsiteProj 构造函数 | ✅ 正确 | 参数传递与基类适配正确 |
| `update_becp` 调用 | ✅ 正确 | `onsite_p->overlap_proj_psi(m, psi_in)` 正确 |
| `cal_ps_dftu` 实现 | ✅ 正确 | vu_device sync 和 onsite_ps_op 调用正确 |
| nspin=2 vu_device 选择 | ✅ 正确 | `ik == 1` 时选择后半部分正确（见 commit 9642f93fe） |
| `add_onsite_proj` GEMM | ✅ 正确 | 参数 `npw, npm, tnp` 正确 |

**发现的关键问题**:

1. **[P0 - 严重] `add_onsite_proj` 中 GEMM 使用 `&this->one` 作为 alpha/beta**:
   - 第 115-129 行：`gemm_op()(..., &this->one, tab_atomic, npw, this->ps, npm, &this->one, hpsi_in, npwx)`
   - **这里 `this->one` 是 `T` 类型**（模板参数），对于 `std::complex<double>` 是 `{1, 0}`
   - 但 GEMM 的 `alpha` 和 `beta` 参数应该是标量，当 `T = std::complex<double>` 时传 `&this->one` 是正确的
   - 当 `T = std::complex<float>` 时，空特化函数不执行任何操作（L399-422），也是正确的
   - **结论**: 此问题不成立，GEMM 用法正确 ✅

2. **[P1] `ps` 内存的初始化条件有隐患**:
   - 第 258-261 行：
     ```cpp
     if(!this->has_delta_spin) 
     {
         setmem_complex_op()(this->ps, 0, tnp * m);
     }
     ```
   - 当 `has_delta_spin == true` 时，`ps` 不会被清零，而是由 `cal_ps_delta_spin` 中的 `setmem_complex_op()(this->ps, 0, tnp * m)` 清零（第 173 行）
   - **但是**，如果 `init_delta_spin` 已经为 `true`（从之前的 k-point 继承），则 `cal_ps_delta_spin` 中的第 173 行不会执行！
   - **风险场景**: 
     - k-point 0: `has_delta_spin = true`, `cal_ps_delta_spin` 走 `!init_delta_spin` 分支，清零 ps
     - k-point 1: `has_delta_spin = true`, `init_delta_spin` 已经为 true，不清零 ps
     - k-point 1 的 DFTU 计算会叠加 k-point 0 的残余数据到 `ps` 上
   - **验证**: 检查 `cal_ps_delta_spin` 第 169-191 行的逻辑...实际上第 173 行 `setmem_complex_op()(this->ps, 0, tnp * m)` **在 `if(!this->init_delta_spin)` 之外**，所以每次调用都会清零。✅ 无问题。

3. **[P1] `tab_atomic_` 的更新时机**:
   - 在 `OnsiteProj::init(ik_in)` 中只调用了 `onsite_p->tabulate_atomic(ik_in)`
   - 而 `tabulate_atomic` 的实现（onsite_projector.cpp 第 279-336 行）几乎是空的——核心计算被注释掉了
   - 这意味着 `tab_atomic_` 的值依赖于 `Onsite_Proj_tools`（fs_tools）的 `cal_becp` 内部的计算
   - 需要确认 `tabulate_atomic` 在 pw-port 中是否正确实现了 `tab_atomic_` 的计算

4. **[P0 - 严重] 诊断打印中的 `m == 28` 硬编码**:
   - 第 82、97、132 行使用了 `if(m == 28 && (this->ik == 0 || this->ik == 1))`
   - `m = 28` 是硬编码的特定 band 索引，这在不同测试案例中可能对应不同的 band
   - 建议改为可配置的参数或基于 band 能量的选择

---

## 三、诊断代码评估

### 3.1 当前诊断策略评估

| 诊断标记 | 位置 | 有效性 | 建议 |
|----------|------|--------|------|
| `PSI-BEFORE` | dftu_pw.cpp L51-54 | ✅ 有价值 | 保留，确认 psi 输入正确 |
| `BECP` | dftu_pw.cpp L61-65 | ✅ 有价值 | 保留，确认投影计算正确 |
| `DFTU-IK` | dftu_pw.cpp L135-146 | ✅ 有价值 | 保留，逐 k-point 验证 locale |
| `DIAG-PW` | dftu_pw.cpp L24-31, L315-441 | ⚠️ 过多 | 精简为关键路径点 |
| `HPSI-PW` | op_pw_proj.cpp L84-87, L132-137 | ✅ 有价值 | 保留 |
| `HPSI-NORM-PW` | op_pw_proj.cpp L97-111 | ✅ 核心诊断 | 保留并扩展到所有 bands |
| `DIAG-INIT` | op_pw_proj.cpp L58 | ⚠️ 低价值 | 可移除 |
| `DIAG-UB` | op_pw_proj.cpp L148-151 | ⚠️ 过多 | 每次 update_becp 都打印，信息过载 |
| `DIAG-OP` | op_pw_proj.cpp L472-489 | ✅ 有价值 | 保留 becp 和 ps 对比 |

### 3.2 建议的改进诊断策略

**当前诊断的不足**:
- 所有诊断都在 `cal_occ_pw` 和 `add_onsite_proj` 层面，但问题出在 Davidson 子空间迭代中
- 没有直接诊断 preconditioner 的输出值
- 没有直接诊断 trial vector（新基向量）的范数和分布

**建议新增的诊断点**:

1. **Preconditioner 输出诊断**（最高优先级）:
   ```cpp
   // 在 diago_david.cpp cal_grad() 的 preconditioning 部分后（约 L488）
   #ifdef __DFTU_DEBUG
   if(dav_iter == 2 && m == 0) {
       // 打印 preconditioner 作用后的 residual 的范数和前几个值
       // 特别关注 precondition[ig] 接近 0 的情况
       double min_prec = 1e300, max_prec = 0;
       for(int ig=0; ig<dim; ++ig) {
           double p = this->precondition[ig];
           if(p < min_prec) min_prec = p;
           if(p > max_prec) max_prec = p;
       }
       std::cout << "[PREC-DBG] iter=" << dav_iter << " min_prec=" << min_prec 
                 << " max_prec=" << max_prec << std::endl;
   }
   #endif
   ```

2. **Trial Vector 诊断**:
   ```cpp
   // 在 cal_grad() 中 basis + dim*nbase 计算完成后（约 L385 后）
   #ifdef __DFTU_DEBUG
   for(int m=0; m<notconv; ++m) {
       double trial_norm = 0;
       for(int ig=0; ig<dim; ++ig) {
           trial_norm += std::norm(basis[dim*(nbase+m) + ig]);
       }
       std::cout << "[TRIAL-DBG] iter=" << dav_iter << " m=" << m 
                 << " unconv[" << m << "]=" << unconv[m] 
                 << " trial_norm=" << trial_norm << std::endl;
   }
   #endif
   ```

3. **Preconditioner 数组来源诊断**:
   ```cpp
   // 在 hsolver_pw.cpp update_precondition() 中
   #ifdef __DFTU_DEBUG
   // 打印 preconditioner 的前 10 个值和最小/最大值
   // 对比 nspin=1 和 nspin=2 的情况
   #endif
   ```

---

## 四、Davidson 求解器分叉问题的根因分析

### 4.1 问题复现

```
iter=1, ik=1, m=28:
  调用 #4: zdy-tmp 范数 ~32  vs pw-port 范数 ~390  ← 分叉！
  调用 #5: zdy-tmp 范数 ~37  vs pw-port 范数 ~41   ← 恢复

iter=2, ik=0:
  hpsi → 1e84 → NaN → assertion failure
```

### 4.2 根因假设（按可能性排序）

#### 假设 H1: Preconditioner 包含接近零的值（可能性：高）

**机理**: 
- Davidson 的 `cal_grad` 函数计算残差 `r = (H - λS)ψ` 后，用 preconditioner 进行缩放：`p = r / precondition`
- 如果 `precondition[ig]` 中有值接近 0（例如来自 DFT+U 的 VU 贡献未正确包含在 preconditioner 中），会导致 `p[ig]` 爆炸
- 爆炸的 trial vector 在下一次 Hψ 调用中产生异常大的输出

**支持证据**:
- 第 4 次调用范数激增 10x，说明输入的 trial vector 有异常大的分量
- 第 5 次调用恢复，说明子空间投影抑制了异常分量
- iter=2 时异常持续累积导致 NaN

**验证方案**:
1. 在 `update_precondition` 后打印 preconditioner 的 min/max/mean
2. 对比 nspin=1（通过）和 nspin=2（失败）的 preconditioner 差异
3. 检查 preconditioner 是否包含了 DFT+U 的 on-site 势贡献

#### 假设 H2: locale 在不同 k-point 之间的状态污染（可能性：中）

**机理**:
- `cal_occ_pw` 遍历所有 k-point 计算 locale
- 在 nspin=2 时，ik=0 对应 spin-up，ik>=nk/2 对应 spin-down
- 如果某个 k-point 的 locale 计算覆盖了另一个 k-point 的数据，会导致 VU 矩阵错误

**支持证据**:
- psi 符号差异（ik=0/2 符号相反）可能暗示 k-point 处理顺序问题
- 第 4 次调用恰好是在某个特定的 band/k-point 组合上

**验证方案**:
1. 在 `cal_occ_pw` 中打印每个 ik 处理后的 locale 总和
2. 对比 zdy-tmp 和 pw-port 的 locale 累加过程
3. 检查 `reduce_double_allpool` 是否正确同步了所有 k-pool 的数据

#### 假设 H3: OnsiteProjector 的 becp 缓存污染（可能性：中）

**机理**:
- `OnsiteProjector` 是单例模式（`get_instance()`）
- `becp` 缓冲区在多次调用间复用
- 如果 `cal_becp` 没有正确清零旧数据，可能累积残余

**支持证据**:
- 第 4 次调用时异常，但第 5 次恢复
- 可能对应 becp 缓冲区中累积的旧数据在特定条件下被错误使用

**验证方案**:
1. 在 `cal_becp` 调用前打印 becp 缓冲区的旧值
2. 检查 `fs_tools->cal_becp` 是否在执行前清零了输出缓冲区
3. 确认 `size_becp` 在不同调用间是否正确更新

#### 假设 H4: GEMM 操作的数值不稳定（可能性：低）

**机理**:
- `add_onsite_proj` 中的 GEMM 操作 `hpsi += tab_atomic * ps`
- 如果 `tab_atomic_` 或 `ps` 中有异常大的值，GEMM 结果会爆炸

**验证方案**:
1. 在 GEMM 前后打印 `tab_atomic_` 和 `ps` 的范数
2. 对比 zdy-tmp 和 pw-port 的 tab_atomic 值

### 4.3 推荐的排查顺序

```
步骤 1: 验证 H1 (Preconditioner)
  → 添加 preconditioner 诊断打印
  → 对比 nspin=1 vs nspin=2 的 preconditioner
  → 检查 DFT+U VU 是否包含在 preconditioner 中

步骤 2: 如果 H1 被排除，验证 H3 (Becp 缓存)
  → 在 cal_becp 调用前后打印缓冲区状态
  → 确认输出缓冲区被正确清零

步骤 3: 如果 H3 被排除，验证 H2 (Locale 污染)
  → 逐 ik 打印 locale 累加过程
  → 对比 reduce 前后的值

步骤 4: 如果以上都被排除，验证 H4 (GEMM 数值)
  → 打印 GEMM 输入输出的范数
```

---

## 五、代码质量检查

### 5.1 C++11 兼容性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| `auto` 关键字 | ✅ | 正确使用 |
| Lambda 表达式 | ✅ | hsolver_pw.cpp 中正确使用 |
| `std::function` | ✅ | diago_david.h 中正确使用 |
| 范围 for 循环 | ✅ | 多处使用，兼容 C++11 |
| `nullptr` | ✅ | 统一使用 nullptr 而非 NULL |
| `override` | ✅ | op_pw_proj.h 中正确使用 |

**未发现 C++11 不兼容的语法**。

### 5.2 内存管理安全性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| `resmem_*` / `delmem_*` 配对 | ✅ | OnsiteProj 构造函数/析构函数中正确配对 |
| `ps` 缓冲区大小跟踪 | ✅ | `nkb_m` 正确跟踪当前大小 |
| Singleton 模式 | ⚠️ | `OnsiteProjector::get_instance()` 使用静态局部变量，线程安全但需要注意多实例场景 |
| `mutable` 使用 | ✅ | `cal_ps_dftu` 和 `cal_ps_delta_spin` 是 `const` 方法，`mutable` 成员是合理的 |

### 5.3 代码风格一致性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 缩进 | ⚠️ | 部分文件使用 tab，部分使用空格，不一致 |
| 命名约定 | ✅ | 遵循 ABACUS 的 snake_case 和 PascalCase 混合约定 |
| 注释风格 | ⚠️ | 诊断打印注释风格不统一（`// DIAGNOSTIC` vs `// DIAG:` vs `[DIAG-xxx]`） |
| 大括号风格 | ⚠️ | 部分 `if` 语句有大括号，部分没有 |

### 5.4 潜在的竞态条件或状态污染

| 检查项 | 风险 | 说明 |
|--------|------|------|
| `OnsiteProjector` 单例 | 低 | 单线程 SCF 循环中安全，但多线程计算可能有问题 |
| `Plus_U::energy_u` 静态变量 | 低 | 当前为单实例模式，但设计上有隐患 |
| `becp` 缓冲区复用 | ⚠️ 中 | 需要在每次调用前确认清零逻辑 |
| `locale` 跨 iter 复用 | ✅ 安全 | 通过 `initialed_locale` 标志正确管理 |

---

## 六、优先级排序的修复建议

### P0（必须在合入前修复）

| # | 问题 | 修复建议 | 影响范围 |
|---|------|----------|----------|
| P0-1 | **Preconditioner 不包含 DFT+U 贡献**（假设 H1） | 检查 `update_precondition` 函数，确认是否需要在 preconditioner 中加入 DFT+U 的 VU 贡献。如果不加入，需要在 VU 很大时使用更保守的 preconditioning 策略 | nspin=2 所有 PW DFTU 测试 |
| P0-2 | **`tab_atomic_` 计算缺失** | `OnsiteProjector::tabulate_atomic` 的核心代码被注释掉了，需要确认 `tab_atomic_` 的值是否在 `fs_tools` 中正确计算。如果 `tab_atomic_` 未被更新，`add_onsite_proj` 中的 GEMM 会使用旧值 | 所有 PW onsite 计算 |

### P1（建议在合入前修复）

| # | 问题 | 修复建议 | 影响范围 |
|---|------|----------|----------|
| P1-1 | **诊断代码未受编译宏保护** | 将所有 `std::cout` 诊断打印包裹在 `#ifdef __DFTU_DEBUG` 中 | 性能 |
| P1-2 | **becp 缓冲区复用安全性** | 在 `overlap_proj_psi` 中明确清零 becp 缓冲区，或确认 `fs_tools->cal_becp` 内部已清零 | 潜在的数值污染 |
| P1-3 | **`ps` 初始化条件复杂** | 简化 `cal_ps_dftu` 和 `cal_ps_delta_spin` 的 `ps` 初始化逻辑，确保每次调用前清零 | 代码可维护性 |
| P1-4 | **`vu_device` 同步的边界条件** | 确认 nspin=2 spin-down 时 `half_size` 计算的边界情况（当 `eff_pot_pw.size()` 为奇数时 `/2` 的行为） | 数值正确性 |

### P2（可以后续修复）

| # | 问题 | 修复建议 | 影响范围 |
|---|------|----------|----------|
| P2-1 | **代码缩进不一致** | 运行 clang-format 统一缩进 | 代码风格 |
| P2-2 | **诊断打印格式混乱** | 统一诊断日志格式，使用 `[模块-级别]` 前缀 | 可读性 |
| P2-3 | **`m == 28` 硬编码** | 将诊断 band 索引改为可配置参数 | 诊断灵活性 |
| P2-4 | **DEBUG_PROGRESS.md 和 TEST_STATUS.md 合并** | 合并为单一文档 | 文档维护 |

---

## 七、发现的严重 Bug

### Bug #1: `tabulate_atomic` 实现缺失（严重性：高）

**位置**: `source/source_pw/module_pwdft/onsite_projector.cpp` 第 278-337 行

**描述**: `OnsiteProjector::tabulate_atomic` 函数的核心计算代码全部被注释掉了（STAGE 1 和 STAGE 2）。这意味着 `tab_atomic_` 数组不会被更新。

**影响**: 
- 如果 `tab_atomic_` 的初始值不为零（例如来自之前的计算），`add_onsite_proj` 中的 GEMM 会使用错误的投影算子
- 如果 `tab_atomic_` 为零，DFT+U 修正将不起作用

**需要确认**:
1. `tab_atomic_` 是否在 `Onsite_Proj_tools::cal_becp` 中被正确计算
2. `OnsiteProjector::tabulate_atomic` 是否应该从 `fs_tools` 获取 `tab_atomic_`

### Bug #2: `uom_array` 的 spin-down 偏移计算（严重性：中）

**位置**: `source/source_lcao/module_dftu/dftu_pw.cpp` 第 275-276 行

**描述**: 
```cpp
this->uom_array[eff_pot_pw_index[iat] + mm + size] = this->locale[iat][target_l][0][1].c[mm];
```

在 zdy-tmp 中，对应的代码是：
```cpp
this->uom_array[eff_pot_pw_index[iat]+mm+size] = this->locale[iat][target_l][0][1].c[mm];
```

两处逻辑一致，但需要注意 `eff_pot_pw_index` 对于所有 atom 是连续的，而 nspin=2 时 spin-down 部分应该从 `eff_pot_pw.size()/2` 开始。当前代码使用 `+ size` 作为偏移，这在单个 atom 内是正确的，但需要确认所有 atom 的 spin-down 部分是否正确放置在数组的后半部分。

**验证**: 在 `dftu.cpp` 的 `init` 函数中，`pot_index` 只计算了 spin-up 部分的大小，然后 nspin=2 时 `pot_index *= 2`。这意味着：
- atom 0: spin-up at 0, spin-down at size_0
- atom 1: spin-up at size_0, spin-down at size_0 + size_1

但 `uom_array` 的索引应该是：
- atom 0 spin-up: `[eff_pot_pw_index[0] ... eff_pot_pw_index[0]+size_0)`
- atom 0 spin-down: `[eff_pot_pw_index[0]+size_0 ... eff_pot_pw_index[0]+2*size_0)`

这与 nspin=2 时整个数组被翻倍的布局是**一致的**，因为 `eff_pot_pw_index[iat]` 指向的是 spin-up 部分的起始位置，`+ size` 正好是同一 atom 的 spin-down 部分。

**结论**: ✅ 索引计算正确。

---

## 八、总结

### 总体评价

PW DFTU 迁移的**核心逻辑是正确的**，与 zdy-tmp 参考实现保持了良好的一致性。API 适配（GlobalV → PARAM, GlobalC → 参数传递等）处理得当。

**主要风险**:
1. Davidson 求解器在 nspin=2 时的 preconditioner 问题是最可能的根因
2. `tabulate_atomic` 实现缺失需要尽快确认
3. 大量诊断代码需要在合入前清理或添加编译宏保护

### 下一步行动建议

1. **立即**: 验证 P0-1（preconditioner）假设，这是最有可能的根因
2. **立即**: 确认 P0-2（tab_atomic）是否真的缺失功能
3. **短期**: 清理诊断代码（P1-1）
4. **中期**: 完善 becp 缓冲区的清零逻辑（P1-2）
5. **长期**: 代码风格统一（P2）

---

*报告结束*
