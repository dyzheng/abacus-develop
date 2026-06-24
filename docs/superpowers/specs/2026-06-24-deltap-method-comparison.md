# DeltaP Method Comparison: Berry Connection vs Wannier

> **Date**: 2026-06-24  
> **Systems**: Si (centrosymmetric, P=0), BaTiO3 (ferroelectric, P≠0)  
> **PP/Orb**: `/root/pporb/` APNS precision set

---

## Results Summary

### Si (diamond, centrosymmetric — P_total should be 0)

| Method | Si[0] Pz | Si[1] Pz | Total Pz | |Error| |
|--------|----------|----------|----------|---------|
| Berry connection (gauge-fixed) | 1.237e-03 | 1.265e-03 | 2.503e-03 | 2.5e-3 |
| Wannier (SVD polar decomp) | 0.0 | -3.647e-02 | -3.647e-02 | 3.6e-2 |

**Berry connection wins**: 14× closer to zero. Per-atom values are symmetric (both Si atoms ≈ 1.25e-3), respecting the inversion symmetry. Wannier method breaks the atomic equivalence (Si[0]=0, Si[1]=-3.6e-2).

### BaTiO3 (tetragonal ferroelectric — P_total should be nonzero)

| Method | Ba Pz | Ti Pz | O[0] Pz | O[1] Pz | O[2] Pz | Total Pz |
|--------|-------|-------|---------|---------|---------|----------|
| Berry connection | 2.14e-2 | -1.54e-2 | 1.81e-2 | -1.69e-2 | -1.70e-2 | -9.83e-3 |
| Wannier | 5.63e-2 | -2.50e-3 | -3.75e-2 | 2.70e-2 | 4.87e-3 | 4.82e-2 |

Both methods produce nonzero P_total (correct for ferroelectric). Berry connection gives more physically reasonable per-atom distribution (Ba positive, Ti negative, O mixed). Wannier concentrates polarization on Ba.

Note: ABACUS berry_phase returns P=0 for both systems (separate issue — likely k-sampling or quantum).

---

## Analysis

### Why Berry Connection is More Reliable Currently

1. **Symmetry preservation**: Berry connection respects the equivalence of symmetric atoms (Si[0] ≈ Si[1]). Wannier SVD breaks this because singular vectors are not atomically localized.

2. **Physical reasonableness**: Berry connection per-atom values follow chemical intuition (Ba positive, Ti negative in BTO). Wannier values are less intuitive.

3. **Magnitude accuracy**: For Si (P=0), Berry connection error is 14× smaller.

### Wannier Method Limitations

The current Wannier implementation uses the approximation `⟨ψ_{k_j}|ψ_{k_{j+1}}⟩ ≈ δ_{nm}` (identity overlap), which:
- Is exact only in the dk→0 limit
- Is too crude for typical k-mesh spacing (4×4×4 or 8×8×8)
- Causes the Wilson loop `det(U†·U_next)` to miss the Berry phase accumulated between k-points

**Fix**: Compute the full overlap matrix O(k_j, k_{j+1}) using `unkOverlap_lcao::prepare_midmatrix_pbas`. This would give the exact Wilson loop and likely improve accuracy significantly.

### Wannier Method Advantages (Theoretical)

Despite current numerical limitations, the Wannier method has key theoretical advantages:
- **Gauge invariant by construction** (SVD eliminates arbitrary phases) — no gauge fixing needed
- **Non-iterative** — unique solution, no local minima
- **Cross-structure continuous** — SMOs move continuously with atoms

These advantages would manifest with the exact overlap matrix implementation.

---

## Recommendation

For Phase B (SCF-integrated constraint loop), use the **Berry connection method with SMO-anchored gauge fixing** as the primary P^I computation. It gives more reliable per-atom decomposition with the current implementation.

The Wannier method should be revisited with the exact overlap matrix (`unkOverlap_lcao`) for a fair comparison. The theoretical advantages (gauge invariance, non-iterativity) make it promising for cross-structure sampling in Phase B.

---

## Future Work

1. **Improve Wannier**: Implement full overlap matrix O(k_j, k_{j+1}) via `unkOverlap_lcao` for exact Wilson loop
2. **Fix ABACUS berry_phase**: Investigate why P_abacus = 0 for BaTiO3 (k-sampling, polarization quantum)
3. **Denser k-mesh**: Test with 16×16×16 or 32×32×32 k-mesh to reduce the identity-overlap approximation error
4. **Method 6 (Projected Wannier)**: Implement the Löwdin orthogonalization approach from the evaluation document as a third alternative
