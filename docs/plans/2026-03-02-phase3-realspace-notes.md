# Phase 3: Real-Space Projector Extension

## Status: Architecture Complete, Real-Space Implementation Deferred

Phase 3 establishes the `ProjectorBase` abstraction layer for nonlocal pseudopotential projectors, enabling future alternative implementations without modifying the core Nonlocal operator.

## Implemented Components

### ProjectorBase (Abstract Interface)

`source/module_hamilt_pw/hamilt_pwdft/projector_base.h`

Template abstract base class `ProjectorBase<T>` defining three pure virtual methods:

- `compute_becp(psi, becp, nbands, npw, max_npw, npol)` - Compute projection coefficients becp = <beta|psi>
- `apply_deeq_and_accumulate(becp, hpsi, nbands, npw, max_npw, npol)` - Apply D matrix and accumulate: hpsi += |beta> * D * becp
- `get_memory_bytes()` - Report current GPU memory usage in bytes

### ReciprocalProjector (Concrete Implementation)

`source/module_hamilt_pw/hamilt_pwdft/reciprocal_projector.h`
`source/module_hamilt_pw/hamilt_pwdft/reciprocal_projector.cpp`

Template class `ReciprocalProjector<T, Device>` implementing `ProjectorBase<T>` using reciprocal-space VKB projectors. Supports two modes:

- **Full mode**: Entire VKB matrix (nkb x npwx) resident on GPU. Standard GEMM operations.
- **Batched mode**: VKB stored on CPU, transferred batch-by-batch to GPU via `VKBBatchManager`. Uses contiguous `becp_batch_` buffer with proper stride handling and MPI reduction.

### Unit Tests

`source/module_hamilt_pw/hamilt_pwdft/test/reciprocal_projector_test.cpp`

9 tests covering: construction, memory reporting, full-mode single/multi-band becp, full-mode D-matrix application (identity and general), batched mode becp, batched mode end-to-end, and polymorphism through base pointer.

## Future Extension: RealSpaceProjector

A `RealSpaceProjector<T, Device>` class would implement `ProjectorBase<T>` using real-space beta(r) projections:

```
becp_i = integral( beta_i(r) * psi(r) dr )
```

This avoids storing the full VKB matrix in reciprocal space at the cost of FFT overhead.

### Implementation Steps (Future Work)

1. Create `RealSpaceProjector` class implementing `ProjectorBase<T>`
2. Implement FFT-based psi(G) -> psi(r) transforms using existing `PW_Basis_K`
3. Generate beta_r from pseudopotential radial functions
4. Implement real-space integration kernel: `becp = sum_r beta(r) * psi(r)`
5. Implement reverse projection: `hpsi(r) += beta(r) * (D * becp)` then FFT back
6. Add `projector_type` INPUT parameter: `"reciprocal"` (default) | `"realspace"`
7. Modify Nonlocal operator to select projector type based on parameter

### Integration Point

The Nonlocal operator (`operator_pw/nonlocal_pw.cpp`) currently has its own inline GEMM + nonlocal_op logic. To use the `ProjectorBase` abstraction:

```cpp
// In Nonlocal::act():
if (projector_ != nullptr) {
    projector_->compute_becp(tmpsi_in, becp, nbands, npw, max_npw, npol);
    Parallel_Reduce::reduce_pool(becp, nkb * nbands);
    projector_->apply_deeq_and_accumulate(becp, tmhpsi, nbands, npw, max_npw, npol);
} else {
    // Existing inline code path
}
```

### When to Implement

- Very large systems (>500 atoms, >50000 plane waves)
- Memory-constrained GPUs where Phase 2 batching is insufficient
- When FFT overhead (~10-15%) is acceptable

## Architecture Diagram

```
                   ProjectorBase<T>
                   (abstract interface)
                         |
            +------------+------------+
            |                         |
   ReciprocalProjector<T,D>   RealSpaceProjector<T,D>
   (implemented, Phase 3)     (future extension)
            |
    +-------+-------+
    |               |
  Full mode    Batched mode
  (GPU VKB)    (CPU VKB + batch transfer)
```

## References

- Design document: `docs/plans/2026-03-02-gpu-memory-optimization-design.md`
- Implementation plan: `docs/plans/2026-03-02-gpu-memory-optimization-implementation.md`
- Phase 1 (Psi paging): `source/module_psi/psi_paging.cpp`
- Phase 2 (VKB batching): `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.cpp`
