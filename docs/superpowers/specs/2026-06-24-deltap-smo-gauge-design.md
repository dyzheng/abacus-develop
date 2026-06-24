# DeltaP SMO-Anchored Gauge Fixing — Design Spec

> **Spec date**: 2026-06-24  
> **Parent spec**: `docs/superpowers/specs/2026-06-24-deltap-nao-foundation-design.md`  
> **Scope**: Gauge-fixing layer (Method 5) for Berry connection continuity, built on top of the existing DeltaP module  
> **Method**: SMO-Anchored Gauge — fix wavefunction phase by requiring max SMO projection to be positive real

---

## 1. Problem Statement

The existing DeltaP Berry connection computation uses finite differences of `D_I = ⟨α|ψ⟩` at neighboring k-points:
```
d_k D_I(lm,n,k_j) ≈ [D_I(k_{j+1}) - D_I(k_{j-1})] / (2·dk)
```

Each k-point's wavefunction `|ψ_{nk}⟩` comes from independent diagonalization with an **arbitrary complex phase**. This makes `D_I(k_{j+1})` and `D_I(k_{j-1})` have unrelated phases, so the finite difference is numerically garbage.

For a single post-processing calculation this causes noise. For the Phase B constraint loop (sampling P across structures), it makes the polarization **discontinuous** — destroying the reliability of the λ inner loop.

## 2. Solution: SMO-Anchored Gauge (Method 5)

### 2.1 Core Idea

For each (band n, k-point k), fix the wavefunction phase by requiring the **largest SMO projection** to be positive real:

```
|ψ̃_{nk}⟩ = e^{-i·arg(⟨α^{ref(n)}_k|ψ_{nk}⟩)} · |ψ_{nk}⟩
```

where `ref(n) = argmax_{I,lm} |⟨α^I_{lmk}|ψ_{nk}⟩|` is the "anchor SMO" — the SMO with the largest projection onto band n at k-point k.

After this rotation: `⟨α^{ref(n)}_k|ψ̃_{nk}⟩` is positive real.

### 2.2 Why This Ensures Continuity

1. SMOs are geometrically defined (they move continuously with atom positions)
2. Wavefunctions change continuously with structure (ground state uniqueness)
3. SMO projections `⟨α|ψ⟩` are continuous mappings of continuous quantities
4. The anchor selection (argmax) is continuous except at "anchor jumps" (Section 2.4)
5. The gauge phase `arg(⟨α^{ref}|ψ⟩)` is continuous

Therefore `|ψ̃_{nk}⟩` is continuous across both k-points and structures.

### 2.3 Continuous Phase Tracking

Even with the same anchor SMO, `arg(D_anchor(k))` can cross the branch cut (jump by π when `D_anchor` crosses the origin). To prevent this:

At each k_j, after computing the raw gauge phase `g(n,k_j)`:
```
if Re(g(n,k_j) · conj(g(n,k_{j-1}))) < 0:
    g(n,k_j) ← -g(n,k_j)   # flip sign to stay continuous
```

This ensures the gauge phase changes smoothly along the k-string.

### 2.4 Anchor Jump Detection and Correction

When the band character changes along the k-string, the maximum-projection SMO may switch from `(I₁, l₁m₁)` to `(I₂, l₂m₂)`. This causes a known, computable phase jump:

```
Δφ = arg(⟨α^{new}_k|ψ_{nk}⟩) - arg(⟨α^{old}_k|ψ_{nk}⟩)
```

**Detection**: if `|D_anchor(n,k_j)| < ε_threshold` (anchor projection too small), re-select the anchor using Strategy A (max projection).

**Correction**: record `Δφ` and apply it as a global phase shift to all subsequent polarization computations for that band.

### 2.5 Existing Code Bug Fix

The current `compute_berry_connection` in `deltap_berry.cpp` computes:
```cpp
bra_grad += std::conj(ds_val) * c_val;  // = ⟨d_kα|ψ⟩ = conj(⟨ψ|d_kα⟩)
```

This computes `conj(⟨ψ|d_kα⟩)` instead of `⟨ψ|d_kα⟩`. Under gauge `ψ → e^{iφ}ψ`:
- Current: `bra_grad → e^{iφ} · bra_grad`, so `term1 = bra_grad · D_I → e^{2iφ} · term1` (NOT invariant)
- Corrected: `bra_grad = sum_mu conj(C) · dS = ⟨ψ|d_kα⟩`, so `bra_grad → e^{-iφ} · bra_grad`, and `term1 = bra_grad · D_I → e^{-iφ} · e^{iφ} · term1 = term1` (INVARIANT)

**Fix**: change `conj(ds_val) * c_val` to `conj(c_val) * ds_val` in the bra_grad computation.

This fix is **essential** for gauge fixing to work — without it, term1 picks up spurious gauge-dependent phases.

## 3. Architecture

### 3.1 Integration with Existing Module

```
Existing call sequence in compute_atomic_polarization:
  1. compute_real_overlaps
  2. setup_kstring
  3. for each k: compute_S_k, compute_D_I          ← D_I computed here
  4. for each k: compute_berry_connection           ← FD uses D_I here
  5. integrate_polarization

New call sequence:
  1. compute_real_overlaps
  2. setup_kstring
  3. for each k: compute_S_k, compute_D_I
  3.5. gauge_fix_smo_anchored                       ← NEW: compute gauge phases
  4. for each k: compute_berry_connection (modified) ← applies gauge phases
  5. integrate_polarization
```

### 3.2 New File: `deltap_gauge.cpp`

```cpp
void DeltaP::gauge_fix_smo_anchored(int nbands)
{
    // Phase 1: Determine anchor SMO at k_0 (first k on string)
    // For each band n: ref(n) = argmax_{I,lm} |D_I(lm,n,k_0)|
    
    // Phase 2: Compute gauge phase at each k-point
    // For each (n, k_j):
    //   D_anchor = D_I[anchor_iat][anchor_lm][n][k_j]
    //   if |D_anchor| < epsilon: re-select anchor, record Δφ
    //   g = conj(D_anchor) / |D_anchor|
    //   continuous tracking: if Re(g · conj(g_prev)) < 0: g = -g
    //   gauge_phase_[k_j][n] = g
}
```

### 3.3 Modified: `deltap_berry.cpp::compute_berry_connection`

Changes:
1. **Bug fix**: `bra_grad += conj(c_val) * ds_val` (was `conj(ds_val) * c_val`)
2. **Gauge on term1**: `c_val_gauge = c_val * gauge_phase_[ik][n]`
3. **Gauge on term2**: `D_I_gauge = D_I * gauge_phase_[ik][n]`, and use gauge-fixed D_I for FD

### 3.4 New Data Members in `deltap.h`

```cpp
std::vector<std::vector<std::complex<double>>> gauge_phase_;  // [ik][n]
std::vector<int> anchor_iat_;                                 // [n]
std::vector<int> anchor_lm_;                                  // [n]
std::vector<std::complex<double>> phase_corrections_;         // accumulated Δφ
```

### 3.5 New Input Parameter

```
deltap_gauge_mode    smo_anchored    # gauge fixing mode: "none" (default), "smo_anchored"
deltap_anchor_thr    1e-8            # threshold for anchor re-selection
```

## 4. Detailed Algorithm

### 4.1 `gauge_fix_smo_anchored`

```
Input: D_I[iat][lm][n][ik] for all atoms, bands, k-points
Output: gauge_phase_[ik][n], anchor_iat_[n], anchor_lm_[n]

Phase 1: Anchor establishment at k_0
  for n = 0 to nbands-1:
    max_proj = 0
    for iat = 0 to nat-1:
      for lm = 0 to nproj_per_atom_[iat]-1:
        proj = |D_I[iat][lm][n][0]|
        if proj > max_proj:
          max_proj = proj
          anchor_iat_[n] = iat
          anchor_lm_[n] = lm
    # Compute gauge at k_0
    D_anchor = D_I[anchor_iat_[n]][anchor_lm_[n]][n][0]
    gauge_phase_[0][n] = conj(D_anchor) / |D_anchor|

Phase 2: Gauge phases at k_1, ..., k_{nppstr-1}
  for j = 1 to nppstr_-1:
    for n = 0 to nbands-1:
      D_anchor = D_I[anchor_iat_[n]][anchor_lm_[n]][n][j]
      
      if |D_anchor| < deltap_anchor_thr:
        # Anchor jump: re-select
        new_iat, new_lm = argmax_{I,lm} |D_I[I][lm][n][j]|
        # Record phase correction
        D_old = D_I[anchor_iat_[n]][anchor_lm_[n]][n][j]
        D_new = D_I[new_iat][new_lm][n][j]
        delta_phi = arg(D_new) - arg(D_old)
        phase_corrections_[n] *= exp(-i * delta_phi)
        anchor_iat_[n] = new_iat
        anchor_lm_[n] = new_lm
        D_anchor = D_new
      
      g = conj(D_anchor) / |D_anchor|
      
      # Continuous phase tracking
      g_prev = gauge_phase_[j-1][n]
      if Re(g * conj(g_prev)) < 0:
        g = -g
      
      gauge_phase_[j][n] = g
```

### 4.2 Modified `compute_berry_connection`

```
for each iat, n, alpha:
  # term1: <psi|d_k alpha> * <alpha|psi>  (gauge invariant after bug fix)
  bra_grad = 0
  for mu = 0 to s_size-1:
    ds_val = dS_k[iat][alpha][lm][mu]
    c_val = psi_k[mu + n * nrow_local]
    c_val_gauge = c_val * gauge_phase_[ik][n]       # APPLY GAUGE
    bra_grad += conj(c_val_gauge) * ds_val           # BUG FIX: conj(C)*dS, not conj(dS)*C
  
  D_I_gauge = D_I[iat][lm][n] * gauge_phase_[ik][n]  # APPLY GAUGE
  term1 = bra_grad * D_I_gauge
  
  # term2: conj(D_I) * d_k D_I  (needs gauge-fixed FD)
  D_next = D_I_next[iat][lm][n] * gauge_phase_[ik_next][n]  # APPLY GAUGE
  D_prev = D_I_prev[iat][lm][n] * gauge_phase_[ik_prev][n]  # APPLY GAUGE
  d_D = (D_next - D_prev) / (2*dk)
  term2 = conj(D_I_gauge) * d_D
  
  A[iat][ik][n][alpha] = term1 + term2
```

## 5. Test Plan

### 5.1 Unit Test: Gauge Anchoring

Verify that after gauge fixing, `D_anchor` is positive real at every k-point:
```
for each (n, k_j):
  D_gauge = D_I[anchor_iat_[n]][anchor_lm_[n]][n][k_j] * gauge_phase_[k_j][n]
  assert |Im(D_gauge)| < 1e-12
  assert Re(D_gauge) > 0
```

### 5.2 Unit Test: Phase Continuity

Verify no sign flips in gauge_phase along the k-string:
```
for j = 1 to nppstr-1:
  overlap = Re(gauge_phase_[j][n] * conj(gauge_phase_[j-1][n]))
  assert overlap > 0  # no π jumps
```

### 5.3 Integration Test: Si and BaTiO3

Re-run the existing Si and BaTiO3 tests with gauge fixing enabled:
- Si: P_total should remain near zero (centrosymmetric)
- BaTiO3: P_total should be more stable and physically reasonable
- Compare with and without gauge fixing

### 5.4 Cross-Structure Test (Future, Phase B)

Run two nearby BaTiO3 structures (slightly different Ti displacement) and verify P^I changes continuously.

## 6. Future Extensions

- **Method 6 (Projected Wannier Functions)**: Alternative computation path that bypasses Berry phase entirely, computing real-space Wannier centers from SMO projections. Can be added as a `deltap_method` parameter option.
- **Multi-string support**: Currently only the first k-string is processed. Extend to all strings for full BZ integration.
- **Parallel transport fallback**: If SMO-anchored gauge fails (e.g., metallic system), fall back to berryphase's parallel transport method.

## 7. References

- Parent spec: `docs/superpowers/specs/2026-06-24-deltap-nao-foundation-design.md`
- Existing module: `source/source_lcao/module_deltap/`
- Berry connection code: `source/source_lcao/module_deltap/deltap_berry.cpp`
- DeltaSpin SMO pattern: `source/source_lcao/module_operator_lcao/dspin_lcao.cpp::cal_PI_sub`
- User's Method 5 description (this conversation)
