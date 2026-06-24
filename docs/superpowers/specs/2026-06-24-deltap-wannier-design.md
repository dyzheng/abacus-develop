# DeltaP SMO-Projected Wannier Function Method — Design Spec

> **Spec date**: 2026-06-24  
> **Parent**: `docs/superpowers/specs/2026-06-24-deltap-nao-foundation-design.md`  
> **Evaluation**: `SMO_Projected_Wannier_Evaluation.md` (Method 6, CWF-equivalent)  
> **Scope**: Alternative P^I computation via SVD-based Wannier functions, with comparison test against Berry connection method  

---

## 1. Goal

Implement the SMO-projected Wannier function method (Method 6 / CWF-equivalent) as an alternative to the Berry connection method, and compare both on Si and BaTiO3 to determine which gives better sum-rule agreement.

## 2. Algorithm (CWF-aligned, from Evaluation Section 6.2)

```
For each k-point k_j on the string:
  1. D_I(k_j) = ⟨α^I_lmk|ψ_nk⟩  [already computed]
  2. SVD: D_I(k_j) = W(k_j) Σ(k_j) V†(k_j)
  3. U(k_j) = W(k_j) V†(k_j)  [polar decomposition, unitary]

For each pair (k_j, k_{j+1}) on the string:
  4. O(k_j, k_{j+1}) = C†(k_j) · S(dk) · C(k_{j+1})  [wavefunction overlap]
     where S(dk) = Σ_R e^{i·dk·R} ⟨φ_μ(0)|φ_ν(R)⟩
  5. M^I(k_j, k_{j+1}) = U^I†(k_j) · O(k_j, k_{j+1}) · U^I(k_{j+1})
     [Wannier-transformed overlap, U^I = submatrix for atom I's lm channels]

  6. Wilson loop: Π_j det(M^I_j)
  7. P^I_α = -(e / 2π·a_α) · Im[log(Π_j det(M^I_j))]
```

**Key advantage**: SVD polar decomposition automatically eliminates arbitrary phases — gauge invariant by construction, no gauge fixing needed.

## 3. Architecture

### 3.1 New File: `deltap_wannier.cpp`

Implements `compute_wannier_polarization()` — the full Wannier-based P^I computation.

Reuses:
- `D_I` matrix from existing `compute_D_I` (step 1)
- `unkOverlap_lcao` class for wavefunction overlaps O(k_j, k_{j+1}) (step 4)
- LAPACK `zgesvd_` for SVD (step 2)

New code:
- Polar decomposition U = W·V† (step 3)
- Wannier overlap M^I = U†·O·U (step 5)
- Wilson loop and per-atom P^I (steps 6-7)

### 3.2 Input Parameter

```
deltap_method    berry_connection    # "berry_connection" (default) or "wannier"
```

### 3.3 Integration

In `compute_atomic_polarization`, branch on `deltap_method`:
- `"berry_connection"`: existing code path (S/dS → D_I → gauge → berry_connection → integrate)
- `"wannier"`: new code path (D_I → SVD → overlap → Wilson loop → P^I)

### 3.4 Comparison Test

Run Si and BaTiO3 with both methods, compare:
- P_total vs ABACUS berry_phase reference
- Per-atom P^I values
- Sum rule: |Σ P^I - P_total| / |P_total|

## 4. Key Implementation Details

### 4.1 SVD of D_I(k)

D_I is a matrix with rows (I,lm) and columns n (bands). At each k:
- Reshape D_I to a dense matrix A with shape (n_proj_total, n_occ_bands)
- Call LAPACK `zgesvd_` to get A = W · Σ · V†
- U = W · V† (polar decomposition, shape n_occ × n_occ)

### 4.2 Wavefunction Overlap O(k_j, k_{j+1})

Reuse `unkOverlap_lcao`:
- Initialize with `lcao_init(ucell, gd, kv, orb)` — same as berryphase
- For each pair (k_j, k_{j+1}): call `prepare_midmatrix_pbas` to get S(dk)
- Compute O = C†(k_j) · S(dk) · C(k_{j+1}) via zgemm

### 4.3 Per-Atom Wannier Overlap

For atom I with n_proj_I channels:
- Extract U^I (n_occ × n_proj_I submatrix of U)
- M^I = U^I†(k_j) · O(k_j, k_{j+1}) · U^I(k_{j+1}) (n_proj_I × n_proj_I)
- det(M^I) via LAPACK `zgetrf_`

### 4.4 Wilson Loop and Polarization

```
P^I_α = -(1 / 2π·a_α) · Im[log(Π_j det(M^I_j))]
```

Sum rule check: Σ_I P^I should equal total Berry phase polarization.

## 5. Comparison Test Plan

| System | Method | Expected P_total | Key Check |
|--------|--------|-----------------|-----------|
| Si | berry_connection | ≈0 | P_total near zero |
| Si | wannier | ≈0 | P_total near zero, compare with berry |
| BaTiO3 | berry_connection | nonzero | Sum rule vs ABACUS |
| BaTiO3 | wannier | nonzero | Sum rule vs ABACUS, compare |

Pass criteria: the method with better sum-rule agreement is recommended for Phase B.
