# DFT+U PW & DeltaSpin 移植开发引导框架

> 版本: v1.0 (2026-04-30)
> 状态: 开发中
> 适用范围: nspin=1/2/4, PW/LCAO 基组, CPU/GPU

---

## 一、算法全景与数据流

### 1.1 SCF 循环中的执行顺序

```
ESolver_KS_PW::iter_init (esolver_ks_pw.cpp:180)
├── module_charge::chgmixing_ks_pw()        ← 电荷混合
├── pelec->cal_delta_eband()                ← Harris 泛函
└── pw::iter_init_dftu_pw()                 ← Gate A: DFT+U 占据矩阵更新 (dftu_pw.cpp:7)
    └── dftu.cal_occ_pw()                   ←   核心: becp → locale → vu → energy_u

ESolver_KS_PW::hamilt2rho_single (esolver_ks_pw.cpp:197)
├── setup_diago_params_pw()                 ← need_subspace 设置 (见 1.3)
├── pw::run_deltaspin_lambda_loop()         ← Gate B: DeltaSpin λ 优化 (deltaspin_pw.cpp:8)
│   └── sc.run_lambda_loop()                ←   核心: cal_mw_from_lambda → 子空间对角化
└── hsolver_pw.solve()                      ← Gate C: 全空间对角化 (仅当 skip_solve=false)

ESolver_KS_PW::iter_finish (esolver_ks_pw.cpp:240)
└── pw::check_deltaspin_oscillation()       ← SCF 振荡检测
```

**关键时序约束**: Gate A (DFT+U) 必须在 Gate B (DeltaSpin) 之前执行，因为 DeltaSpin 的 `cal_mw_from_lambda` 使用当前波函数计算磁矩，而波函数已经包含了 DFT+U 的有效势贡献。

### 1.2 三大分支矩阵

| 维度 | nspin=1 | nspin=2 | nspin=4 |
|------|---------|---------|---------|
| **nk (k点总数)** | nk | 2×nk | nk |
| **spin 区分方式** | 无 | k 点划分: ik<nk → ↑, ik≥nk → ↓ | 自旋极化: npol=2 |
| **npol** | 1 | 1 | 2 |
| **wg 形状** | (nk, nbands) | (2nk, nbands) | (nk, nbands) |
| **becp 形状** | (nbands, nkb) | (nbands, nkb) | (2nbands, nkb) |
| **locale 存储** | [size] | [size×2] (up/dn) | [size×4] (Pauli) |
| **vu 存储** | [size] | [size×2] (up/dn) | [size×4] (Pauli→spin) |
| **有效势耦合** | DFT+U only | DFT+U only | DFT+U + DeltaSpin |

### 1.3 iter=1 时 HSolver 的二次求解逻辑

```cpp
// diago_params.cpp:15
need_subspace = ((istep == 0 || istep == 1) && iter == 1) ? false : true;
```

**DFT+U 场景**:
- `iter=1, istep=0/1`: `need_subspace=false` → 全空间对角化，不保存子空间数据
- `iter>1` 或 `istep>1`: `need_subspace=true` → 保存子空间 H/S 矩阵和 becp
- **影响**: `iter_init_dftu_pw` 中 `iter==1 && istep==0` 跳过 DFT+U 计算，因为此时没有电荷混合历史，locale 未初始化

**DeltaSpin 场景**:
- `cal_mw_from_lambda(i_step=-1)`: 首次调用，`i_step=-1` 触发 `cal_hs_subspace` 保存 H/S/becp
- `cal_mw_from_lambda(i_step>=0)`: 使用保存的子空间数据，仅应用 delta_lambda 修正
- **关键**: `sub_h_save`, `sub_s_save`, `becp_save` 在首次调用时分配，在 `update_psi_charge_pw` 中释放
- **冲突点**: DFT+U 和 DeltaSpin 都依赖 `need_subspace=true` 时的子空间数据，但分配/释放时机不同

### 1.4 DFT+U 和 DeltaSpin 的耦合与解耦

#### 当前耦合点

| 耦合位置 | 文件 | 耦合性质 | 解耦难度 |
|---------|------|---------|---------|
| `iter_init` 顺序 | esolver_ks_pw.cpp:192 | DFT+U 先于 DeltaSpin | 低 (固定顺序即可) |
| `OnsiteProj` 单例 | op_pw_proj.cpp | 共享 OnsiteProjector | 低 (只读) |
| `becp` 计算 | dftu_pw.cpp / cal_mw.cpp | 各自独立计算 becp | 中 (可共享) |
| `need_subspace` 数据 | diago_params.cpp | 两者都需要子空间数据 | 高 (生命周期不同) |
| `eff_pot_pw` 存储 | dftu.h | DFT+U 独占 | 无耦合 |
| `lambda_` 存储 | spin_constrain.h | DeltaSpin 独占 | 无耦合 |

#### 解耦原则

1. **计算不重复**: OnsiteProjector 的 `overlap_proj_psi` 在 DFT+U 和 DeltaSpin 中各自调用一次。PW 基组中，每个 k 点的 becp 计算成本高，应共享。
2. **状态不交叉**: DFT+U 的 `locale`/`eff_pot_pw` 和 DeltaSpin 的 `lambda_`/`Mi_` 各自管理，不互相读写。
3. **生命周期分离**: DFT+U 数据贯穿整个 SCF；DeltaSpin 的子空间数据仅在 lambda_loop 内有效。

---

## 二、nspin 维度详细设计

### 2.1 nspin=2: k 点划分自旋通道

**核心特征**: 通过 k 点索引区分自旋，而非通过 npol

```cpp
// dftu_pw.cpp:28-31 - spin 通道判定
int is = 0;
if(PARAM.inp.nspin == 2 && ik >= psi_p->get_nk()/2)
{
    is = 1;  // spin-down
}

// cal_mw.cpp:108-126 - DeltaSpin Mi 计算 (npol=1)
const int sign = this->pelec->klist->isk[ik] == 0 ? 1 : -1;  // ↑=+1, ↓=-1
this->Mi_[iat].z += weight * occ * sign;  // Mi = ρ_up - ρ_down
```

**becp 索引公式**:
```
index = ib * nkb + begin_ih + m
```
- `ib`: 能带索引 (0..nbands-1)
- `nkb`: 每个 k 点的投影子总数
- `begin_ih`: 当前原子之前的投影子偏移
- `m`: 磁量子数偏移 (m_begin + m)

**locale 累加 (nspin=2)**:
```cpp
// dftu_pw.cpp:86-90 (nspin=2 分支)
const int index_m1 = ib*nkb + begin_ih + m_begin + m1;
const int index_m2 = ib*nkb + begin_ih + m_begin + m2;
this->locale[iat][l][0][is].c[ind_m1m2] += weight * (conj(becp[index_m1]) * becp[index_m2]).real();
```

**vu 计算 (nspin=2)**:
```cpp
// dftu_pw.cpp:331-358
// spin-up: vu_iat = &eff_pot_pw[eff_pot_pw_index[iat]]
// spin-down: vu_iat1 = &eff_pot_pw[eff_pot_pw.size()/2 + eff_pot_pw_index[iat]]
```

**OnsiteProj 传递 (nspin=2)**:
```cpp
// op_pw_proj.cpp:258-290
if(PARAM.inp.nspin == 2 && this->isk[this->ik] == 1)
{
    // spin-down: 取 eff_pot_pw 的后半部分
    syncmem_complex_h2d_op()(this->vu_device, 
        dftu->get_eff_pot_pw(0) + size_eff_pot_pw, size_eff_pot_pw);
}
else
{
    // spin-up: 取 eff_pot_pw 的前半部分
    syncmem_complex_h2d_op()(this->vu_device, dftu->get_eff_pot_pw(0), dftu->get_size_eff_pot_pw());
}
```

### 2.2 nspin=4: npol=2 自旋极化

**核心特征**: 自旋信息编码在波函数的 npol=2 分量中，使用 Pauli 矩阵表示

**becp 索引公式**:
```
index = ib * npol * nkb + begin_ih + m    (spinor 分量 0)
index = ib * npol * nkb + begin_ih + m + nkb  (spinor 分量 1)
```

**Pauli 占据矩阵 (occ[0..3])**:
```cpp
// dftu_pw.cpp:64-72 (nspin=4)
occ[0] = weight * conj(becp[index_m1]) * becp[index_m2];        // ↑↑
occ[1] = weight * conj(becp[index_m1]) * becp[index_m2 + nkb];  // ↑↓
occ[2] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2];  // ↓↑
occ[3] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];  // ↓↓

// 转换到 Pauli 基:
locale[0] = (occ[0] + occ[3]).real();   // 单位矩阵分量 (总占据)
locale[1] = (occ[1] + occ[2]).real();   // σ_x 分量
locale[2] = (occ[1] - occ[2]).imag();   // σ_y 分量
locale[3] = (occ[0] - occ[3]).real();   // σ_z 分量 (磁化)
```

**vu 的 Pauli→自旋表示转换**:
```cpp
// dftu_pw.cpp:309-329
// 先计算 Pauli 基下的 vu:
vu[0] = U * (diag_coeff * δ(m1,m2) - locale[0][m2,m1])  // 单位分量
vu[1] = U * (0 - locale[1][m2,m1])                       // σ_x
vu[2] = U * (0 - locale[2][m2,m1])                       // σ_y
vu[3] = U * (0 - locale[3][m2,m1])                       // σ_z

// 再转换到自旋表示 (σ_z 本征基):
vu[0] = 0.5 * (vu_tmp[0] + vu_tmp[3])   // ↑↑
vu[3] = 0.5 * (vu_tmp[0] - vu_tmp[3])   // ↓↓
vu[1] = 0.5 * (vu_tmp[1] + i * vu_tmp[2])   // ↑↓
vu[2] = 0.5 * (vu_tmp[1] - i * vu_tmp[2])   // ↓↑
```

**onsite_ps_op kernel (npol=2)**:
```cpp
// onsite_op.cpp:70-101
// npol=2 时，每个 band 有 2 个 spinor 分量
for (int ib = 0; ib < npm / npol; ib++)  // npm/npol = nbands
{
    for (int ip = 0; ip < tnp; ip++)
    {
        int ib2 = ib * npol;  // ib2 = ib*2, 指向 spinor 对
        int iat = ip_iat[ip];
        const std::complex<FPTYPE>* vu_iat = vu + vu_begin_iat[iat];
        int tlp1 = 2 * orb_l + 1;
        int tlp1_2 = tlp1 * tlp1;
        int ip2_begin = ip - m1;
        int ip2_end = ip - m1 + tlp1;
        
        // vu_iat 布局: [0..tlp1_2-1] = ↑↑, [tlp1_2..2*tlp1_2-1] = ↑↓,
        //              [2*tlp1_2..3*tlp1_2-1] = ↓↑, [3*tlp1_2..4*tlp1_2-1] = ↓↓
        ps[psind]   += vu_iat[index_mm]          * becp[becpind]
                    + vu_iat[index_mm + tlp1_2*2] * becp[becpind + tnp];  // ↑ 分量
        ps[psind+1] += vu_iat[index_mm + tlp1_2*1] * becp[becpind]
                    + vu_iat[index_mm + tlp1_2*3] * becp[becpind + tnp];  // ↓ 分量
    }
}
```

### 2.3 nspin=1: 无自旋

**最简单情况**:
- nk 个 k 点，npol=1
- locale 只有 [size] 一个通道
- vu 计算使用 `diag_coeff=0.5`, `weight_eu=1.0`
- DeltaSpin 不适用 (需要自旋自由度)

---

## 三、DeltaSpin 约束机制详解

### 3.1 约束标志 (constrain_)

```cpp
// spin_constrain.h:270
std::vector<ModuleBase::Vector3<int>> constrain_;  // 每个原子 3 个分量 (x,y,z)

// 含义:
// constrain_[iat].x = 1 → 约束 Mi.x 到 target_mag_[iat].x
// constrain_[iat].y = 1 → 约束 Mi.y 到 target_mag_[iat].y
// constrain_[iat].z = 1 → 约束 Mi.z 到 target_mag_[iat].z
// constrain_[iat].* = 0 → 不约束该分量
```

### 3.2 sc_direction_only: 仅约束方向

```cpp
// spin_constrain.h:274
bool direction_only_ = false;  // 仅优化磁化方向，不约束大小
```

**方向约束的数学处理**:

1. **lambda 正交化** (lambda_loop.cpp:159-174):
```cpp
if(this->direction_only_)
{
    // 将 lambda 投影到垂直于 target_mag 的平面
    const ModuleBase::Vector3<double> dir = target / norm;
    double parallel = lambda.x*dir.x + lambda.y*dir.y + lambda.z*dir.z;
    lambda.x -= parallel * dir.x;  // 去除平行分量
    lambda.y -= parallel * dir.y;
    lambda.z -= parallel * dir.z;
}
```

2. **残差正交化** (lambda_loop.cpp:203-225):
```cpp
// delta_spin = Mi - target_mag
// 计算垂直于 target 方向的残差平方
const double parallel = delta_spin.x*dir.x + delta_spin.y*dir.y + delta_spin.z*dir.z;
temp_1[ia][0] = |delta_spin|² - parallel²;  // 垂直分量的平方
```

3. **增量投影** (lambda_loop.cpp:295-307, 335-347):
```cpp
// dnu (lambda 增量) 也需要投影到垂直方向
double parallel = dnu.x*dir.x + dnu.y*dir.y + dnu.z*dir.z;
dnu.x -= parallel * dir.x;  // 确保增量不改变磁化大小
```

**方向约束 vs 全约束的差异**:

| 特性 | 全约束 | direction_only |
|------|--------|----------------|
| lambda 自由度 | 3 | 2 (垂直于 target) |
| 残差度量 | |Mi-target|² | |Mi⊥-target⊥|² |
| target 修正 | 不变 | 动态更新平行分量 |
| 物理意义 | 固定磁矩大小和方向 | 只固定磁矩方向 |

### 3.3 Lambda 更新策略

```cpp
// lambda_loop.cpp: 标准流程
for (int i_step = -1; i_step < this->nsc_; i_step++)
{
    // i_step=-1: 初始评估，cal_mw_from_lambda(-1) 保存子空间数据
    // i_step=0:  设置 current_sc_thr_，第一次 lambda 更新
    // i_step>=1: CG 加速 (beta = mean_error / mean_error_old)
    //            最优步长 (alpha_opt)
    //            梯度衰减检查 (check_gradient_decay)
}
```

**关键变量**:
- `initial_lambda`: 约束为 0 的分量归零后的 lambda
- `delta_lambda`: lambda 相对于 initial_lambda 的变化
- `dnu`: 累积的 lambda 更新量 (CG 搜索方向)
- `search`: 当前搜索方向 (delta_spin + beta * search_old)

---

## 四、iter=1 时两个算法的特殊考虑

### 4.1 DFT+U 在 iter=1

```cpp
// dftu_pw.cpp:22-25
if (iter == 1 && istep == 0)
{
    return;  // 跳过: 没有电荷混合历史，locale 未初始化
}
```

**原因**:
1. `iter_init_dftu_pw` 在 `iter_finish` 之后、`hamilt2rho_single` 之前调用
2. 第一次 SCF 迭代 (`iter=1`) 使用初始猜测波函数，没有先前的 locale 用于混合
3. 但 `iter=2` 时需要 `iter=1` 的 locale 作为混合参考

**zdy-tmp 行为**:
- `initialed_locale` 在 dftu_pw.cpp 中**从未设置为 true**
- 每次迭代都从零重新计算 locale（与我们的代码一致）

### 4.2 DeltaSpin 在 iter=1

```cpp
// cal_mw_from_lambda.cpp:346-367
if(this->sub_h_save == nullptr)
{
    initial_hs = 1;
    this->sub_h_save = new std::complex<double>[nbands * nbands * nk];
    this->sub_s_save = new std::complex<double>[nbands * nbands * nk];
    this->becp_save = new std::complex<double>[size_becp * nk];
}
// ...
if(initial_hs)
{
    hamilt_t->updateHk(ik);  // 构建 H(k)
    hsolver::DiagoIterAssist::cal_hs_subspace(hamilt_t, psi_t[0], h_k, s_k);
    memcpy(becp_k, onsite_p->get_becp(), sizeof(std::complex<double>) * size_becp);
}
```

**关键**:
1. `i_step=-1`: 首次调用，保存 H/S/becp 子空间数据
2. `i_step>=0`: 使用保存的数据，应用 delta_lambda 修正
3. `update_psi_charge_pw`: 使用子空间数据后释放（一次性消费）

**与 DFT+U 的交互**:
- DFT+U 在 `iter_init` 阶段计算 vu 并更新 Hamiltonian
- DeltaSpin 的 `cal_hs_subspace` 获取的是**已经包含 DFT+U 修正的** H(k)
- 这意味着 DeltaSpin 的子空间对角化自动包含 DFT+U 效应

### 4.3 数据生命周期对比

```
SCF iter=1 (istep=0):
├── iter_init: DFT+U 跳过 (iter==1 && istep==0)
├── hamilt2rho_single:
│   ├── run_deltaspin_lambda_loop: skip_solve=true (drho 未收敛)
│   └── HSolverPW::solve: need_subspace=false → 全空间对角化
└── iter_finish: check_oscillation

SCF iter=2:
├── iter_init: DFT+U 计算 (iter>1)
│   └── cal_occ_pw: becp → locale → vu → energy_u
├── hamilt2rho_single:
│   ├── run_deltaspin_lambda_loop: skip_solve=true (drho > sc_scf_thr)
│   └── HSolverPW::solve: need_subspace=true → 保存子空间数据
└── iter_finish: check_oscillation

SCF iter=N (drho < sc_scf_thr):
├── iter_init: DFT+U 计算
├── hamilt2rho_single:
│   ├── run_deltaspin_lambda_loop:
│   │   ├── i_step=-1: cal_mw_from_lambda → 保存 H/S/becp
│   │   ├── i_step=0..nsc: 子空间对角化 + lambda 更新
│   │   └── update_psi_charge_pw → 释放子空间数据
│   └── HSolverPW::solve: 跳过 (skip_solve=true)
└── iter_finish: check_oscillation
```

---

## 五、解耦开发步骤

### Step 1: becp 共享机制

**目标**: DFT+U 和 DeltaSpin 共享 OnsiteProjector 的 becp 计算，避免重复

**当前状态**: 各自独立调用 `onsite_p->overlap_proj_psi()`

**改造方案**:
```cpp
// 在 OnsiteProj 中缓存 becp
class OnsiteProj {
    bool becp_ready = false;
    void ensure_becp(const T* psi_in, int npol, int m) {
        if(!becp_ready) {
            update_becp(psi_in, npol, m);
            becp_ready = true;
        }
    }
    // cal_ps_delta_spin / cal_ps_dftu 都调用 ensure_becp
};
```

**验证**: 对比共享前后的 locale/Mi 数值一致性

### Step 2: 子空间数据生命周期管理

**目标**: DFT+U 和 DeltaSpin 对子空间数据的访问不冲突

**当前冲突**:
- DeltaSpin 在 `cal_mw_from_lambda(-1)` 分配 `sub_h_save`/`becp_save`
- DeltaSpin 在 `update_psi_charge_pw` 释放这些数据
- DFT+U 不直接使用子空间数据，但依赖 `need_subspace` 设置

**改造方案**:
```cpp
// 引入 SubspaceData 管理器
class SubspaceDataManager {
    std::complex<double>* sub_h_save = nullptr;
    std::complex<double>* sub_s_save = nullptr;
    std::complex<double>* becp_save = nullptr;
    
    void allocate(int nk, int nbands, int size_becp);
    void release();
    bool is_ready() const;
};
```

**验证**: 确保 DeltaSpin lambda_loop 期间子空间数据完整

### Step 3: nspin=2 自旋通道隔离测试

**目标**: 独立验证 spin-up 和 spin-down 路径

**测试点**:
1. `is` 分配: `ik < nk/2 → is=0`, `ik >= nk/2 → is=1`
2. `isk` 映射: `klist->isk[ik] == 0 → sign=+1`, `== 1 → sign=-1`
3. vu 传递: `nspin==2 && isk[ik]==1` → 取 eff_pot_pw 后半部分
4. 能量权重: `weight_eu = 0.5` (nspin=2)

**验证**: 对比 nspin=1 (仅 spin-up) 和 nspin=2 (spin-up) 的 locale 数值

### Step 4: nspin=4 Pauli 矩阵转换验证

**目标**: 验证 occ[0..3] → Pauli → vu → 自旋表示 的完整链路

**测试点**:
1. `occ[0..3]` 计算: 4 个 spinor 分量组合
2. Pauli 基转换: `(occ[0]+occ[3]).real()` 等
3. vu 计算: `diag_coeff=1.0` (nspin=4)
4. Pauli→自旋转换: `0.5*(vu_tmp[0]±vu_tmp[3])` 等
5. kernel 应用: `vu_iat[index_mm + tlp1_2*N]` 索引

**验证**: 使用已知 Pauli 矩阵输入，验证输出自旋表示正确

### Step 5: direction_only 约束隔离

**目标**: 验证方向约束不改变磁化大小

**测试点**:
1. lambda 正交化: `lambda -= (lambda·dir) * dir`
2. 残差正交化: `|delta_spin|² - parallel²`
3. 增量投影: `dnu -= (dnu·dir) * dir`
4. target 动态更新: `target += parallel * dir`

**验证**: 比较 direction_only 和全约束下的 |Mi| 演化

### Step 6: ESolver 层集成测试

**目标**: 验证 DFT+U + DeltaSpin 组合在 SCF 中的正确性

**测试矩阵**:

| Case | nspin | DFT+U | DeltaSpin | 验证目标 |
|------|-------|-------|-----------|---------|
| PW_DFTU_S2 | 2 | ✓ | ✗ | locale/vu/energy |
| PW_DS_S2 | 2 | ✗ | ✓ | Mi/lambda 收敛 |
| PW_DFTU+DS_S2 | 2 | ✓ | ✓ | 组合效应 |
| PW_DFTU_S4 | 4 | ✓ | ✗ | Pauli 转换 |
| PW_DS_S4 | 4 | ✗ | ✓ | npol=2 kernel |
| PW_DFTU+DS_S4 | 4 | ✓ | ✓ | 完整链路 |

---

## 六、调试方法论

### 6.1 分离验证法（已建立）

**原则**: 每个组件独立验证，再逐步集成

**已完成的单元测试**:
- ✅ `VU_Calculation_Nspin2_FullPath` — vu 算术逻辑
- ✅ `VU_DeviceSync_Nspin2` — vu 设备同步
- ✅ `OnsitePsOpKernel_Nspin2_Npol1` — kernel 应用
- ✅ `SpinUpOnly_Path_Nspin2` — spin-up 隔离
- ✅ `SpinDownOnly_Path_Nspin2` — spin-down 隔离

### 6.2 数值对比法

**原则**: 与 zdy-tmp 参考实现对比关键中间量

**对比点**:
1. `becp` 值 (每个 k 点、每个原子)
2. `locale` 对角元 (每个原子、每个自旋通道)
3. `vu` 矩阵对角元 (每个原子、每个自旋通道)
4. `energy_u` (总能量修正)
5. `Mi` (原子磁矩)

**工具**: 在关键位置添加条件编译的 debug dump

### 6.3 冻结变量法

**原则**: 冻结怀疑有问题的变量，观察系统行为

**应用场景**:
- 冻结 vu 为 iter=3 的值，观察 iter=4 是否仍发散
- 冻结 locale 为初始值，观察 SCF 是否收敛
- 冻结 lambda 为 0，观察 DFT+U 单独行为

---

## 七、文件索引

### DFT+U 核心文件

| 文件 | 功能 | 关键行 |
|------|------|--------|
| `module_dftu/dftu.h` | Plus_U 类定义 | 全部 |
| `module_dftu/dftu_pw.cpp` | cal_occ_pw, cal_VU_pot_pw | 9-363 |
| `module_dftu/dftu_occup.cpp` | LCAO 占据矩阵 | - |
| `module_dftu/dftu_hamilt.cpp` | Hamiltonian 贡献 | - |
| `module_pwdft/dftu_pw.cpp` | iter_init_dftu_pw | 7-33 |
| `module_pwdft/op_pw_proj.cpp` | OnsiteProj::cal_ps_dftu | 174-291 |
| `module_pwdft/kernels/onsite_op.cpp` | onsite_ps_op kernel | 58-131 |

### DeltaSpin 核心文件

| 文件 | 功能 | 关键行 |
|------|------|--------|
| `module_deltaspin/spin_constrain.h` | SpinConstrain 类定义 | 全部 |
| `module_deltaspin/lambda_loop.cpp` | run_lambda_loop | 101-366 |
| `module_deltaspin/cal_mw_from_lambda.cpp` | cal_mw_from_lambda, update_psi_charge_pw | 全部 |
| `module_deltaspin/cal_mw.cpp` | cal_mi_pw, cal_mi_lcao | 55-211 |
| `module_pwdft/deltaspin_pw.cpp` | run_deltaspin_lambda_loop | 8-42 |

### ESolver 集成

| 文件 | 功能 | 关键行 |
|------|------|--------|
| `esolver_ks_pw.cpp` | PW SCF 主循环 | 180-236 |
| `esolver_ks_lcao.cpp` | LCAO SCF 主循环 | 408-426 |
| `hsolver/diago_params.cpp` | need_subspace 设置 | 7-44 |
| `hsolver/diago_iter_assist.cpp` | 子空间对角化 | 474+ |

---

## 八、常见问题与陷阱

### 8.1 nspin=2 自旋通道混淆

**陷阱**: `is` (locale 索引) 和 `isk` (k 点自旋标记) 是不同的概念

- `is`: 通过 `ik >= nk/2` 判定，用于 locale 的 up/down 通道
- `isk`: 通过 `klist->isk[ik]` 获取，用于 DeltaSpin 的 sign 和 vu 传递

**正确做法**: 各自独立使用，不混用

### 8.2 vu 存储布局不一致

**陷阱**: nspin=2 时 vu 在 eff_pot_pw 中存储为 `[up_block | down_block]`

- `get_eff_pot_pw(0)` 返回 up_block 起始
- spin-down 需要 `get_eff_pot_pw(0) + size_eff_pot_pw/2`

**验证**: 打印 `eff_pot_pw.size()` 和 `get_size_eff_pot_pw()` 确认

### 8.3 DeltaSpin 子空间数据一次性消费

**陷阱**: `update_psi_charge_pw` 会释放 `sub_h_save`/`becp_save`

- 如果后续还需要这些数据（如第二次 lambda_loop），会访问已释放内存
- `sub_h_save == nullptr` 是重新分配的触发条件

**正确做法**: 确保每个 SCF 迭代只调用一次 `run_lambda_loop`

### 8.4 nspin=4 Pauli 转换顺序

**陷阱**: vu 必须先计算 Pauli 基下的值，再转换到自旋表示

- 转换公式: `vu[0] = 0.5*(vu_tmp[0]+vu_tmp[3])` 等
- 必须使用临时数组 `vu_tmp` 保存原始值

**验证**: 检查 `vu_tmp` 是否正确保存了转换前的值

---

## 九、下一步行动计划

### 短期（本周）

1. **Step 1**: 实现 becp 共享机制，减少重复计算
2. **Step 3**: 完成 nspin=2 自旋通道隔离测试
3. **调试 P0-1**: 使用 Step 3 验证结果定位发散根因

### 中期（下周）

4. **Step 4**: nspin=4 Pauli 矩阵转换验证
5. **Step 5**: direction_only 约束隔离测试
6. **Step 2**: 子空间数据生命周期管理

### 长期

7. **Step 6**: ESolver 层集成测试矩阵
8. **文档**: 补充 API 文档和使用示例

---

## 十、P0-1 调试进展 (2026-04-30)

### 关键发现

1. **iter=3 时 locale_dn 爆炸**: 在 reduce 前，locale_dn 从 iter=2 的 ~0.3 爆炸到 10^16
2. **locale_up 正常**: iter=3 时 locale_up 保持正常值 (~0.9)
3. **问题在 locale 累加**: 爆炸发生在 k 点循环累加过程中，不在 MPI reduce
4. **mix_uom nspin=2 分支缺失**: `charge_mixing.cpp:281-285` 中只处理 nspin=1/4，遗漏 nspin=2

### becp 数值追踪 (2026-04-30 更新)

**iter=2 ik=4 (spin-down)**: `becp[0] = (0.0026, 0.0009)` — 正常
**iter=3 ik=4 (spin-down)**: `becp[0] = (495850104, 311362186)` — 垃圾值!
**iter=3 ik=5 (spin-down)**: `becp[0] = (0.0000, -0.0025)` — 正常

**结论**: spin-down k-point ik=4 的波函数在 iter=2→3 之间被破坏，而 ik=5 正常。
这表明 HSolverPW::solve 在处理 nspin=2 时，部分 spin-down k 点的波函数被写入了错误数据。

**内存崩溃**: 测试以 signal 6 (Aborted) 终止，"free(): invalid next size" — 进一步证实内存损坏。

### 待验证假设

1. **Hypothesis A**: HSolverPW::solve 对 nspin=2 的 spin-down k 点写入越界
2. **Hypothesis B**: Psi 对象的 spin-down 部分内存布局与 solver 期望不一致
3. **Hypothesis C**: 电荷混合 (chgmixing) 在 iter=2→3 时破坏了波函数

---

*本文档是动态更新的开发指导框架，随着调试进展和代码重构持续完善。*
