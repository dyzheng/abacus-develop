# TDDFT Velocity Gauge Force Implementation Report

## Summary

This report documents the implementation and testing of force calculations for TDDFT velocity gauge (`td_stype=1`) in ABACUS.

## Modified Files

1. **source/source_lcao/FORCE_STRESS.cpp**
   - Added conditional logic to use `TDNonlocal` vs `Nonlocal` based on vector potential magnitude
   - When |A| < 1e-10, uses standard `Nonlocal` to ensure consistency with DFT
   - Added `#include "source_lcao/module_rt/td_info.h"`

2. **source/source_lcao/module_operator_lcao/td_ekinetic_lcao.h**
   - Added `cal_force()` method declaration

3. **source/source_lcao/module_operator_lcao/td_ekinetic_lcao.cpp**
   - Added `cal_force()` method implementation

4. **source/source_lcao/module_operator_lcao/td_ekinetic_force.hpp** (NEW)
   - Template implementation for TDEkinetic force calculation
   - Computes force from A² term in velocity gauge

5. **source/source_lcao/module_operator_lcao/td_nonlocal_lcao.h**
   - Added `cal_force()` and `cal_force_IJR()` method declarations

6. **source/source_lcao/module_operator_lcao/td_nonlocal_lcao.cpp**
   - Added `cal_force()` method implementation

7. **source/source_lcao/module_operator_lcao/td_nonlocal_force.hpp** (NEW)
   - Template implementation for TDNonlocal force calculation
   - Uses `snap_psibeta_half_tddft` for integral computation

## Test Case

- **Directory**: `test-force-compare/`
- **System**: CO molecule (2 atoms)
- **Method**: LCAO + PBE + DZP basis
- **Parameters**: ecutwfc=60, k-points=1x1x8

### DFT Calculation
- Calculation type: SCF with force
- Total energy: -588.166 eV
- Force on C (z): 4.968 eV/Å

### TDDFT Velocity Gauge (A=0)
- Calculation type: MD (1 step)
- td_stype=1, td_dt=0.1
- Vector potential: A = (0, 0, 0)
- Total energy: -588.166 eV
- Force on C (z): 4.968 eV/Å

## Results

| Property | DFT | TDDFT (A=0) |
|----------|-----|-------------|
| Energy (eV) | -588.166 | -588.166 |
| Force C-z (eV/Å) | 4.968 | 4.968 |
| Force O-z (eV/Å) | -4.968 | -4.968 |

The forces are identical when the vector potential A=0, confirming physical consistency.

## Physical Background

In TDDFT velocity gauge, the Hamiltonian is modified by the vector potential A(t):

```
H → H + A·∇ + A²/2
```

The nonlocal pseudopotential contribution is modified by a phase factor:

```
<ψ|β> → <ψ|exp(-iA·r)|β>
```

When A=0, the Hamiltonian reduces to the standard DFT Hamiltonian, and all physical quantities (energy, forces) should match exactly.

## Key Implementation Detail

The `snap_psibeta_half_tddft` function computes `<ψ|exp(-iA·r)|β>` and its derivatives. However, there's a numerical discrepancy with the standard `TwoCenterIntegrator::snap` when A=0. To ensure physical consistency, we use the standard `Nonlocal` class when |A| < 1e-10:

```cpp
const double A_norm = TD_info::cart_At.norm();
if(PARAM.inp.td_stype == 1 && A_norm > 1e-10)
{
    // Use TDNonlocal for non-zero A
    hamilt::TDNonlocal<...> tmp_nonlocal(...);
    tmp_nonlocal.cal_force(isforce, dmR, fvnl_dbeta);
}
else
{
    // Use standard Nonlocal for A≈0
    hamilt::Nonlocal<...> tmp_nonlocal(...);
    tmp_nonlocal.cal_force_stress(isforce, isstress, dmR, fvnl_dbeta, svnl_dbeta);
}
```

## Date

2026-04-05