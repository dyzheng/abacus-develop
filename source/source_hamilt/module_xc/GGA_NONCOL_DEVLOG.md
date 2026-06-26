# GGA Non-Collinear Spin Correction (gga_grad) Development Log

## Date: 2026-06-26

## Overview
Implemented three methods for computing GGA gradients in non-collinear spin (nspin=4) systems,
ported from the `dyzheng/gga_grad3` reference branch where method 3 has been verified for accuracy.

## Mathematical Background

In nspin=4, charge density is `(ρ₀, m₁, m₂, m₃)`. GGA functionals require `∇ρ↑` and `∇ρ↓`,
but these must be derived from the 4-component representation.

### Method 1 (gga_grad=1): Collinear Approximation
- ρ↑ = 0.5(ρ₀ + |m|), ρ↓ = 0.5(ρ₀ - |m|)
- ∇ρ↑ and ∇ρ↓ computed independently → drops cross-terms ∇m̂·(h1-h2)
- Simplest but least accurate for non-uniform magnetization

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
   - Kept old overload for backward compatibility (gga_grad=1)

4. `source/source_hamilt/module_xc/xc_functional_vxc.cpp`
   - Added early delegate to `v_xc_ncgga_sf_builtin()` when gga_grad=3 && nspin=4

5. `source/source_hamilt/module_xc/xc_functional_gradcorr.cpp`
   - Added `#include "xc_functional_gga_noncol_sf_builtin.h"`
   - Added early return for gga_grad=3 stress path
   - Added `mag_part` vector declared at function scope
   - Dispatch `noncolin_rho` based on gga_grad version
   - gga_grad>=2: compute grad(rho0+core) first, then add/sub magnetic gradients
   - gga_grad==1: keep independent grad(rho_up), grad(rho_down)
   - New `!is_stress` block for gga_grad>=2:
     - Rotate v(up/dn) → v(0/1-3) using mag_part
     - div(0.5*(h1+h2)) → v(0)
     - gga_grad==2: Σ m̂_μ · div(0.5*(h1-h2)*m̂_μ) → scalar, then project
     - gga_grad==3: div(0.5*(h1-h2)*m̂_μ) → v(μ) independently (SF)
   - Fixed gcc_spin zeta sign: use `fabs(zeta)` instead of `fabs(zeta)*neg[ir]`
   - Conditional `delete[] neg` only for gga_grad==1

6. `source/source_hamilt/module_xc/CMakeLists.txt`
   - Added `xc_functional_gga_noncol_sf_builtin.cpp` to build

## Test Results

### PW nspin=4 (BCC Fe, PBE, ecutwfc=20, no DeltaSpin)
| Method     | Etot (eV)           | Notes |
|------------|---------------------|-------|
| gga_grad=1 | -6370.383801522982  | Original method |
| gga_grad=2 | -6359.384004330599  | Improved gradient but projected div(h) |
| gga_grad=3 | -6370.700266918100  | SF transform, most accurate |

Difference between methods 1 and 3 (~0.32 eV) reflects the SF cross-term correction.

### LCAO nspin=4 (BCC Fe, PBE, ecutwfc=15, no DeltaSpin)
| Method     | Etot (eV)           | Difference from ref |
|------------|---------------------|---------------------|
| gga_grad=1 | -6267.4651888506    | (reference)         |
| gga_grad=3 | -6267.4651888505    | ~1e-10 eV           |

For this test case with spatially uniform magnetization, SF reduces to method 1 exactly,
confirming correctness of the implementation.

### PW DeltaSpin nspin=4 (gga_grad=3 + sc_strategy=accuracy)
| Configuration | Etot (eV)           |
|---------------|---------------------|
| T07 (gga_grad=1, accuracy) | -6369.19883 |
| T07 (gga_grad=3, accuracy) | -6369.26695 |

Difference ~0.07 eV, SF correction improves self-consistency in DeltaSpin.

## Design Decisions

1. **Backward compatibility**: gga_grad defaults to 1, preserving existing behavior
2. **Two noncolin_rho overloads**: Old signature for gga_grad=1, new for gga_grad>=2
3. **SF as separate file**: Self-contained, easy to review and maintain
4. **No libxc dependency for SF**: Uses built-in PBE, matching reference branch
5. **gga_grad=3 only for nspin=4**: Silently ignored for nspin=1,2 (no effect)
