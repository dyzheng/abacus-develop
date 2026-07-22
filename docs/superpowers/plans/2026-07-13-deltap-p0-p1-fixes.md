# DeltaP P0/P1 Bug Fixes Implementation Plan

> **Goal:** Fix 4 bugs: P0 (Hungarian matching for B16), P1 (C2 zeta rescaling, H4 phase_corrections_, H3 per-atom branch spacing)

**Architecture:** All changes in `deltap_wannier.cpp` and `deltap_gauge.cpp`. Each fix is independent, verified by compile + SCF integration test.

**Verification:** `cd /root/abacus-develop/build && make -j$(nproc) && <run BN SCF test>`

---

### Task 1: P0 — Hungarian algorithm for eigenvalue matching

**Files:** `source/source_lcao/module_deltap/deltap_wannier.cpp:682-718`

- Replace greedy nearest-neighbor eigenvalue tracking with Hungarian (Munkres) O(n³) global optimal matching
- Cost matrix: |phase(new_eval[n] / prev_eval[m])| wrapped to (-π,π]
- Guarantees deterministic matching even when eigenvalues are near-degenerate

---

### Task 2: C2 — Zeta rescaling with unwrapped Σγ

**Files:** `source/source_lcao/module_deltap/deltap_wannier.cpp:1002-1009`

- Replace `gamma_correct = arg(zeta_scalar)` → `gamma_unw_sum = Σ_n gamma_unwrapped[n]`
- `arg(det W)` is bounded to (-π,π]; unwrapped sum correctly tracks multiples of 2π

---

### Task 3: H4 — Apply phase_corrections_ retroactively

**Files:** `source/source_lcao/module_deltap/deltap_gauge.cpp:129`

- After anchor switch, apply `phase_corrections_[n]` to all previously computed `gauge_phase_[jj][n]` for jj < j
- 1-line for-loop addition

---

### Task 4: H3 — Per-atom weight in branch selection spacing

**Files:** `source/source_lcao/module_deltap/deltap_wannier.cpp:1037`

- Replace `2.0 * M_PI * scale` with `2.0 * M_PI * w_sum_I[iat]` where `w_sum_I[iat] = Σ_n w_In_matrix[n][iat]`
- Current uniform `scale` causes up to 5x spacing error for light atoms (e.g., H in H₂O)

---
