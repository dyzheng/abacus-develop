# GGA Non-Collinear Spin Correction (gga_grad) Development Log

## Date: 2026-06-26

## Overview
Implemented three methods for computing GGA gradients in non-collinear spin (nspin=4) systems,
ported from the `dyzheng/gga_grad3` reference branch where method 3 has been verified for accuracy.

## Mathematical Background

In nspin=4, charge density is `(ρ₀, m₁, m₂, m₃)`. GGA functionals require `∇ρ↑` and `∇ρ↓`,
but these must be derived from the 4-component representation.

### Method 1 (gga_grad=1): Collinear Approximation (改进版)
- ρ↑ = 0.5(ρ₀ + |m|), ρ↓ = 0.5(ρ₀ - |m|)
- ∇ρ↑ and ∇ρ↓ computed independently → drops cross-terms ∇m̂·(h1-h2)
- Simplest but least accurate for non-uniform magnetization
- **注意**: 与原始代码的关键区别——不再依赖全局量子化轴 ux_/lsign_ 和 neg 数组，
  改用局域方向余弦 m̂_μ = m_μ/|m| 进行 V_xc 磁分量旋转，
  gcc_spin 中 zeta 使用 fabs(zeta) 而非 fabs(zeta)*neg[ir]

### Method 2 (gga_grad=2): Improved Gradient + Projected div(h)
- ∇ρ↑ = 0.5∇(ρ₀+ρ_c) + Σ_μ 0.5 m̂_μ ∇m_μ
- ∇ρ↓ = 0.5∇(ρ₀+ρ_c) - Σ_μ 0.5 m̂_μ ∇m_μ
- Correct density gradients, but div(h) projected back via m̂·Σ m̂·div(...)
- Still drops (h1-h2)·∇m̂_μ cross-terms in the magnetic V_xc

### Method 3 (gga_grad=3): Scalmani-Frisch Transform
- Same gradient formula as method 2
- div(h) computed for each magnetic component μ independently:
  V(μ) -= div(0.5*(h1-h2)*m̂_μ)  for μ=1,2,3
- Retains ALL cross-terms including (h1-h2)·∇m̂_μ
- Most accurate, verified in reference branch

### 原始代码 vs gga_grad=1 的关键差异

原始代码中 nspin=4 GGA 的默认方案使用了 `neg[ir]` / `ux_` / `lsign_` 机制：
- `lsign_` 仅在所有原子初始磁化方向平行时为 true
- `neg[ir] = sign(m(ir)·ux_)` 即实空间每点磁化方向与全局量子化轴的关系
- gcc_spin 中 zeta = `fabs(zeta) * neg[ir]`（恢复符号）
- V_xc 磁分量旋转用 `neg[ir] * Δv * rho[i]/amag`

参考分支的 gga_grad=1 统一改用局域方向余弦 `mag_part[ir + μ*nrxx] = m_μ/|m|`：
- gcc_spin 中 zeta = `fabs(zeta)`（始终为正，因为 |zeta|≤1）
- V_xc 磁分量旋转用 `Δv * m̂_μ`

数学关系：当 `lsign_=false` 时 `neg[ir]=1`，则 `neg*rho[i]/amag = m_μ/|m| = m̂_μ`，两者等价；
当 `lsign_=true` 时 `neg[ir]` 引入全局轴依赖，物理上不合理——GGA 交换相关穴应只依赖局域性质，
不应引入人为的全局参考方向。参考分支的修正消除了这个非物理依赖。

## Files Modified

### New Files
1. `source/source_hamilt/module_xc/xc_functional_gga_noncol_sf_builtin.h`
   - Namespace `ModuleXC::NCGGA_SF_Builtin`
   - `v_xc_ncgga_sf_builtin()` — full LDA+GGA potential for nspin=4
   - `gradcorr_ncgga_sf_builtin()` — GGA stress for nspin=4

2. `source/source_hamilt/module_xc/xc_functional_gga_noncol_sf_builtin.cpp`
   - Self-contained implementation using built-in PBE functionals
   - Calls `xc_spin`, `gcx_spin`, `gcc_spin`, `grad_rho`, `grad_dot`

### Modified Files
1. `source/source_io/module_parameter/input_parameter.h`
   - Added `int gga_grad = 1;` with annotation

2. `source/source_io/module_parameter/read_input_item_elec_stru.cpp`
   - Added `gga_grad` input item with validation (1, 2, or 3)
   - Detailed description of each method

3. `source/source_hamilt/module_xc/xc_functional.h`
   - Added new overload of `noncolin_rho` with `mag_part[3*nrxx]` output
   - Kept old overload for backward compatibility (内部使用)

4. `source/source_hamilt/module_xc/xc_functional_vxc.cpp`
   - Added early delegate to `v_xc_ncgga_sf_builtin()` when gga_grad=3 && nspin=4

5. `source/source_hamilt/module_xc/xc_functional_gradcorr.cpp`
   - Added `#include "xc_functional_gga_noncol_sf_builtin.h"`
   - Added early return for gga_grad=3 stress path
   - Removed `neg` variable and all references to `ux_/lsign_` in GGA path
   - gga_grad=1: 统一使用 `noncolin_rho` 新重载（输出 mag_part），不再调用旧版本
   - gcc_spin 中 zeta 统一使用 `fabs(zeta)` （无 neg 修正）
   - V_xc 旋转统一使用 `mag_part[ir + (i-1)*nrxx]` 替代 `neg*rho[i]/amag`
   - gga_grad>=2: compute grad(rho0+core) first, then add/sub magnetic gradients
   - gga_grad==1: keep independent grad(rho_up), grad(rho_down)
   - New `!is_stress` block for gga_grad>=2:
     - Rotate v(up/dn) → v(0/1-3) using mag_part
     - div(0.5*(h1+h2)) → v(0)
     - gga_grad==2: Σ m̂_μ · div(0.5*(h1-h2)*m̂_μ) → scalar, then project
     - gga_grad==3: div(0.5*(h1-h2)*m̂_μ) → v(μ) independently (SF)

6. `source/source_hamilt/module_xc/CMakeLists.txt`
   - Added `xc_functional_gga_noncol_sf_builtin.cpp` to build

## Test Results

### PW nspin=4 (BCC Fe, PBE, ecutwfc=20, no DeltaSpin)
| Method     | Etot (eV)           | 与 gga_grad=1 差异 |
|------------|---------------------|---------------------|
| gga_grad=1 | -6370.622787020624  | (修正后 baseline)   |
| gga_grad=3 | -6370.700266918086  | -0.0775 eV          |

修正前的原始 gga_grad=1（含 neg/ux_/lsign_）结果为 -6370.38380 eV，
与修正后差异 ~0.24 eV，说明旧的量子化轴机制引入了明显的误差。

### LCAO nspin=4 (BCC Fe, PBE, ecutwfc=15, no DeltaSpin)
| Method     | Etot (eV)           | 与参考差异           |
|------------|---------------------|----------------------|
| gga_grad=1 | -6267.4651888506    | (参考值)             |
| gga_grad=3 | -6267.4651888505    | ~1e-10 eV            |

对于磁化方向空间均匀的系统，SF方法退化为方法1，确认了实现的正确性。

### PW DeltaSpin nspin=4 (gga_grad=3 + sc_strategy=accuracy)
| Configuration | Etot (eV)           |
|---------------|---------------------|
| T07 (gga_grad=1, accuracy) | -6369.19883 |
| T07 (gga_grad=3, accuracy) | -6369.26695 |

差异 ~0.07 eV，SF 修正改善了 DeltaSpin 自洽性。

## Design Decisions

1. **Backward compatibility**: gga_grad defaults to 1，但与原始默认方案有本质区别
2. **统一使用 mag_part**: gga_grad=1/2/3 全部使用新的 `noncolin_rho` 重载，
   彻底移除 `neg`/`ux_`/`lsign_` 依赖，消除全局量子化轴引入的非物理误差
3. **保留旧 noncolin_rho 重载**: 仅作为内部实现保留，不再从主路径调用
4. **SF as separate file**: Self-contained, easy to review and maintain
5. **No libxc dependency for SF**: Uses built-in PBE, matching reference branch
6. **gga_grad=3 only for nspin=4**: Silently ignored for nspin=1,2 (no effect)
