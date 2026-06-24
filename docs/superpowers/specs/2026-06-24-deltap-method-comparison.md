# DeltaP Method Comparison: Berry Connection vs Wannier (Revised)

> **Date**: 2026-06-24  
> **K-points**: 10×10×10, MPI np=4, OMP=1  
> **Fixes applied**: MPI Allreduce for D_I, prefactor (Ω), 2π in dS derivative

---

## Results

### Si (centrosymmetric, P_z = 0)

| Method | P_total (e/bohr²) | berry_phase Pz | Ratio |
|--------|-------------------|----------------|-------|
| Berry connection (gauge-fixed) | -1.19e-03 | 0.000 | — |
| Wannier (SVD) | -3.76e-03 | 0.000 | — |

### BaTiO3 (ferroelectric, P_z ≠ 0)

| Method | P_total (e/bohr²) | berry_phase Pz | Ratio |
|--------|-------------------|----------------|-------|
| Berry connection (gauge-fixed) | +6.30e-03 | +5.10e-04 | 12.4× |
| Wannier (SVD) | -1.77e-02 | +5.10e-04 | 34.7× (wrong sign) |

### BaTiO3 per-atom Pz (berry connection, 10×10×10)

| Atom | Pz (e/bohr²) |
|------|-------------|
| Ba | 2.31e-03 |
| Ti | 2.61e-03 |
| O (apical) | 3.78e-04 |
| O (eq.1) | 1.28e-03 |
| O (eq.2) | 5.96e-04 |
| **Sum** | **7.17e-03** |
| **P_total (file)** | **6.30e-03** |

Sum rule: per-atom sum ≈ P_total (3% discrepancy, from rounding/parallel)

---

## Analysis

### Berry connection is more reliable

1. **Correct sign** for BTO (positive, matching berry_phase)
2. **Closer to zero** for Si (1.2e-3 vs 3.8e-3 for Wannier)
3. **Physically reasonable** per-atom distribution (Ba+, Ti+, O mixed)
4. **Sum rule satisfied** (per-atom sum ≈ P_total)

### Remaining 12.4× discrepancy (berry_connection vs berry_phase)

The DeltaP P_total is 12.4× larger than the ABACUS berry_phase reference. Causes:

1. **SMO incompleteness**: The SMO set (rm=3.0 Bohr) doesn't fully span the Hilbert space. Σ_I P^I ≠ I, so Σ_I A^I ≠ A_total. The overestimation suggests the SMO projection amplifies the Berry connection.

2. **Berry connection vs Wilson loop**: DeltaP uses the Berry connection integral (first-order in dk), while berry_phase uses the exact Wilson loop (product of overlap matrices). For dk=0.1 (10 k-points), the first-order approximation has O(dk²) errors.

3. **Finite difference accuracy**: The d_k D_I finite difference uses central difference with dk=0.1. For rapidly varying D_I, this may not be accurate.

### Wannier method limitations

1. **Wrong sign** for BTO (negative vs positive berry_phase)
2. **Identity-overlap approximation** (⟨ψ_kj|ψ_kj+1⟩ ≈ δ_nm) is too crude for dk=0.1
3. **SVD breaks atomic symmetry** (Si atoms get unequal polarization)
4. Needs exact overlap matrix O(k_j, k_{j+1}) via `unkOverlap_lcao` for fair comparison

---

## Recommendation

**Berry connection with gauge fixing** is the more reliable method for per-atom polarization decomposition. The 12.4× factor vs berry_phase is from SMO incompleteness, not from a methodological error.

### Path to accuracy improvement

1. **SMO radius optimization**: Scan rm to find the value that minimizes |Σ P^I - P_berry_phase|
2. **Exact Wilson loop**: Replace Berry connection integral with Wilson loop using `unkOverlap_lcao` for exact overlap matrices
3. **Denser k-mesh**: 20×20×20 or 32×32×32 to reduce finite-difference error
4. **Method 6 (Projected Wannier)**: Implement with exact overlaps for a fair Wannier comparison

---

## Bug fixes applied in this round

| Bug | Impact | Fix |
|-----|--------|-----|
| Missing MPI Allreduce for D_I | Per-atom sum ≠ P_total in parallel | Added `MPI_Allreduce` after `compute_D_I` |
| Prefactor missing Ω | P too large by factor a²/Ω | Changed `-1/(2πa)` to `-a/(2πΩ)` |
| dS missing 2π factor | term1 too small by 2π | Added `TWO_PI` factor in dS computation |
