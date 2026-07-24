# Branch Comparison: feat/ds-lcao-subspace-accel vs dyzheng/lcao_solver_subspace

Date: 2026-07-24
Base: merge-base `b9669d3757`

## Summary

| | Current branch `feat/ds-lcao-subspace-unified` | dyzheng `lcao_solver_subspace` |
|---|---|---|
| Unique commits | **120** | **41** (2 own + 39 upstream PRs) |
| gga_grad=3 | Complete (5 commits) | Missing entirely |
| DeltaSpin sign convention | Unified (`H_DS = +lambda.sigma`) | Old (wrong for nspin=4) |
| cal_escon gate | Fixed (empty-check guard) | Bug (`is_Mi_converged` gate) |
| LambdaSolver factory | Not present | BFGS/ChiGuided/Subspace/FDCG |
| HSolverLCAOSubspace | Not present (has lcao_subspace.cpp) | General-purpose for MD/relax |

## dyzheng's 2 Own Commits

### 1. `d6b70003e` - Feature: Add LCAO subspace solver for SCF acceleration (~2527 lines)
- `HSolverLCAOSubspace` class for general subspace diagonalization
- `LambdaSolver` factory with 4 solvers (BFGS, ChiGuided, Subspace, FDCG)
- `cal_PI_sub` + `B_I_data`/`BI_AdjacentData` in dspin_lcao
- New parameters: `lcao_subspace_persistent`, `lcao_subspace_clear_thr`

### 2. `065a84c8d` - Fix: build_subspace_lcao ZGEMM dimension (~137 lines)
- Correct LCAO matrix dimensions (nlocal x nbands) instead of PW dimensions
- `build_subspace_lcao()` for MD/relax subspace cache initialization

## Critical Issues in dyzheng's Branch

### Issue 1: gga_grad=3 Completely Missing
Current branch has 5 commits implementing full gga_grad functionality:
- `396ff46b1` gga_grad parameter for non-collinear GGA gradient methods
- `8d9ff5129` Unify gga_grad=1 to use mag_part
- `b5b289858` gga_grad=3 SF transform in libxc path (default=3)
- `306bb14cd` Stress validation for gga_grad=3
- `21c1ba060` Remove nspin=4 guard on gint_precision

dyzheng has none of these. Not in upstream either.

### Issue 2: DeltaSpin Energy Correction Bugs

**(a) nspin=4 off-diagonal sign error** (`dspin_lcao.cpp` cal_coeff_lambda):

| | H_{up,dn} | H_{dn,up} |
|---|---|---|
| Current (fixed) | `lambda_x - i*lambda_y` | `lambda_x + i*lambda_y` |
| dyzheng (old) | `lambda_x + i*lambda_y` | `lambda_x - i*lambda_y` |

Fixed in current branch by commit `de96b2753`. dyzheng still uses old convention, **nspin=4 results are incorrect**.

**(b) cal_escon() `is_Mi_converged` gate**:

dyzheng:
```cpp
if (!this->is_Mi_converged) { return 0.0; }  // returns 0 before convergence
```

Current branch:
```cpp
if (this->lambda_.empty() || this->Mi_.empty()) { return 0.0; }  // safe empty-check
```

Old code skips energy correction before convergence, causing **energy discontinuity**.

## Recommendation

**Do not directly merge** dyzheng's `lcao_solver_subspace` branch.

**Cherry-pick candidates** (requires sign/escon fixes first):
- `LambdaSolver` factory pattern (FDCG solver is more robust than BFGS)
- `build_subspace_lcao()` for MD/relax subspace cache (current branch only uses subspace inside DeltaSpin lambda loop)
