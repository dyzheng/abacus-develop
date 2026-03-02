# GPU Memory Optimization for Davidson Diagonalization

**Date:** 2026-03-02
**Author:** Design Session with User
**Status:** Approved Design

## Executive Summary

This design document describes a phased approach to optimize GPU memory usage in ABACUS's Davidson diagonalization solver for plane-wave (PW) basis calculations. The optimization targets medium-large systems (50-200 atoms, 50-200 k-points) and aims to achieve 75-85% GPU memory reduction with less than 15% performance overhead.

**Key Strategy:** K-point paging + atom-batch VKB computation

**Expected Outcomes:**
- Phase 1: 90-95% reduction in wavefunction memory, ~5-8% overhead
- Phase 2: 85-90% reduction in VKB memory, ~7-10% overhead
- Combined: 75-85% total GPU memory reduction, ~12-15% overhead

## Background

### Current Memory Bottlenecks

In GPU-accelerated Davidson diagonalization (`dav_subspace` method), the primary memory consumers are:

1. **Wavefunctions (psi):** `nk × nband × npw × sizeof(complex<double>)`
   - For 100 k-points, 200 bands, 10000 plane waves: ~3.2 GB

2. **Nonlocal pseudopotential projectors (vkb):** `nkb × npw × sizeof(complex<double>)`
   - For 200 atoms, 10 projectors/atom, 10000 plane waves: ~320 MB

3. **Davidson workspace:** `nbase_x × npw` where `nbase_x = nband × pw_diag_ndim`
   - For 200 bands, ndim=4, 10000 plane waves: ~256 MB

### Design Constraints

- **Performance target:** <15% overhead acceptable
- **Memory target:** 70-80% reduction required
- **System scale:** 50-200 atoms, 50-200 k-points, 5000-20000 plane waves
- **Implementation:** Phased approach with early validation
- **Compatibility:** Maintain backward compatibility with existing code

## Architecture Overview

### High-Level Design

The optimization uses a **two-tier memory hierarchy** where CPU RAM acts as a staging area for GPU memory:

```
CPU Memory (Large)          GPU Memory (Limited)
├─ All k-points psi    →    ├─ Current k-point psi only
├─ VKB templates       →    ├─ Current atom batch vkb
└─ Eigenvalues         ←    └─ Davidson workspace (basis, hphi, hcc, vcc)
```

### Key Design Principles

1. **K-point locality:** Davidson solver processes one k-point at a time, so only current k-point data needs to be on GPU
2. **Atom-batch locality:** Nonlocal operator acts on all atoms, but can be computed in batches and accumulated
3. **Minimal code disruption:** Changes isolated to HSolverPW, Psi class, and Nonlocal operator
4. **Backward compatibility:** Original all-GPU path remains available via runtime flag

### Memory Model

- **Before optimization:** `GPU_mem = nk × nband × npw + nkb × npw + workspace`
- **After optimization:** `GPU_mem = 1 × nband × npw + nkb_batch × npw + workspace`
- **Reduction factor:** For nk=100, nkb_batch=nkb/10: ~92% reduction in psi+vkb memory

### Data Flow

1. HSolverPW loops over k-points
2. For each k-point:
   - Transfer psi[ik] from CPU → GPU
   - Initialize vkb batch manager
   - Call Davidson solver (operates on GPU)
   - Transfer eigenvalues and converged psi[ik] from GPU → CPU
3. Davidson solver calls Hamiltonian operator
4. Nonlocal operator computes vkb in batches, accumulates result

### Configuration

New INPUT parameters:
- `device_memory_mode`: `full_gpu` (default, current behavior) or `paged` (new optimization)
- `vkb_batch_atoms`: Number of atoms per batch (default: 0 = auto-tune)

## Phase 1: Psi K-Point Paging

**Timeline:** 2-3 weeks
**Goal:** Reduce wavefunction GPU memory by 90-95%

### Component Changes

#### 1. Psi Class Modifications

**File:** `source/module_psi/psi.h`

Add storage mode enum and dual-storage support:

```cpp
enum class PsiStorageMode {
    ALL_GPU,      // Current behavior: all k-points on GPU
    ALL_CPU,      // All k-points on CPU (for reference)
    PAGED_GPU     // New: CPU storage + single k-point GPU buffer
};

class Psi {
    PsiStorageMode storage_mode_;
    T* psi_cpu_;           // CPU storage for all k-points (nk × nband × nbasis)
    T* psi_gpu_buffer_;    // GPU buffer for current k-point (1 × nband × nbasis)
    int current_k_gpu_;    // Which k-point is currently on GPU (-1 if none)

    void load_k_to_gpu(int ik);      // Transfer k-point from CPU to GPU
    void store_k_from_gpu(int ik);   // Transfer k-point from GPU to CPU
    void ensure_k_on_gpu(int ik);    // Load if not already loaded
};
```

**Memory allocation strategy:**
- In `PAGED_GPU` mode: allocate `psi_cpu_` on CPU, `psi_gpu_buffer_` on GPU (size = 1 k-point)
- `get_pointer()` returns `psi_gpu_buffer_` when in paged mode
- `fix_k(ik)` triggers `ensure_k_on_gpu(ik)` in paged mode

#### 2. HSolverPW Changes

**File:** `source/module_hsolver/hsolver_pw.cpp`

Modify k-point loop in `solve()` method (around lines 328 and 370):

```cpp
for (int ik = 0; ik < nks; ik++) {
    if (psi.get_storage_mode() == PsiStorageMode::PAGED_GPU) {
        psi.load_k_to_gpu(ik);  // CPU → GPU transfer
    }

    psi.fix_k(ik);
    this->update_precondition(...);
    this->hamiltSolvePsiK(pHamilt, psi, precondition, ...);

    if (psi.get_storage_mode() == PsiStorageMode::PAGED_GPU) {
        psi.store_k_from_gpu(ik);  // GPU → CPU transfer
    }
}
```

#### 3. Data Transfer Optimization

Use CUDA streams for overlapping computation and transfer:

- **Stream 0:** Computation (Davidson solver)
- **Stream 1:** Data transfer (next k-point prefetch)

**Double buffering strategy:**

```cpp
// Pseudocode for overlapped transfer
cudaStream_t compute_stream, transfer_stream;

for (int ik = 0; ik < nks; ik++) {
    // Wait for previous transfer to complete
    cudaStreamSynchronize(transfer_stream);

    // Swap buffers: transfer_buffer becomes compute_buffer
    swap(psi_gpu_buffer_, psi_gpu_transfer_buffer_);

    // Start computation on current k-point
    hamiltSolvePsiK(...);  // Uses compute_stream

    // Prefetch next k-point in parallel (if ik+1 < nks)
    if (ik + 1 < nks) {
        cudaMemcpyAsync(psi_gpu_transfer_buffer_,
                        psi_cpu_ + (ik+1)*nband*nbasis,
                        size, cudaMemcpyHostToDevice, transfer_stream);
    }
}
```

**Expected overhead:** ~5-8% from transfers, reduced to ~3-5% with prefetching.

#### 4. Davidson Solver Compatibility

No changes needed to `DiagoDavid` or `Diago_DavSubspace` - they already operate on single k-point data via `psi.get_pointer()`. The paging is transparent to the solver.

### Memory Savings

For a typical system (100 k-points, 200 bands, 10000 plane waves, complex double):
- **Before:** 100 × 200 × 10000 × 16 bytes = 3.2 GB
- **After:** 1 × 200 × 10000 × 16 bytes = 32 MB (with double buffering: 64 MB)
- **Reduction:** 98% for psi component

### Implementation Tasks

1. Add `PsiStorageMode` enum and member variables to Psi class
2. Implement `load_k_to_gpu()` and `store_k_from_gpu()` methods
3. Modify Psi constructor to support PAGED_GPU mode
4. Update HSolverPW k-point loop with transfer calls
5. Implement double buffering with CUDA streams
6. Add unit tests for paging functionality
7. Run integration tests with paged mode

### Acceptance Criteria

- ✓ All unit tests pass
- ✓ All integration tests pass with paged mode
- ✓ GPU memory reduction ≥90% for psi component
- ✓ Performance overhead ≤8%
- ✓ Energy differences <1e-6 eV vs baseline

## Phase 2: VKB Atom Batching

**Timeline:** 2-3 weeks (after Phase 1)
**Goal:** Reduce VKB GPU memory by 85-90%

### Component Changes

#### 1. Batch Strategy

Divide atoms into batches based on GPU memory budget:

```cpp
// Auto-tuning logic
int compute_optimal_batch_size(int total_atoms, int nproj_per_atom,
                                int npw, size_t available_gpu_mem) {
    size_t vkb_size_per_atom = nproj_per_atom * npw * sizeof(T);
    size_t target_batch_mem = available_gpu_mem * 0.15;  // Use 15% for vkb
    int batch_size = target_batch_mem / vkb_size_per_atom;
    return std::max(1, std::min(batch_size, total_atoms));
}
```

**Batching by atom type** (more efficient than arbitrary batches):
- Group atoms by type (same number of projectors)
- Process all atoms of one type, then next type
- Reduces kernel launch overhead and improves cache locality

#### 2. VKB Batch Manager

**New file:** `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h`

```cpp
class VKBBatchManager {
    int nbatch_;                    // Number of batches
    std::vector<int> batch_start_;  // Starting atom index for each batch
    std::vector<int> batch_size_;   // Number of atoms in each batch
    std::vector<int> batch_nkb_;    // Number of projectors in each batch
    T* vkb_batch_gpu_;              // GPU buffer for current batch

    void init(const UnitCell* ucell, int npw, size_t gpu_mem_budget);
    void get_batch_info(int ibatch, int& atom_start, int& atom_end, int& nkb_batch);
    T* get_vkb_batch_buffer() { return vkb_batch_gpu_; }
};
```

#### 3. Nonlocal Operator Modifications

**File:** `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

Modify `act()` method to loop over batches:

```cpp
void Nonlocal::act(...) {
    if (use_vkb_batching) {
        // Batched path
        for (int ibatch = 0; ibatch < vkb_manager_.get_nbatch(); ibatch++) {
            int atom_start, atom_end, nkb_batch;
            vkb_manager_.get_batch_info(ibatch, atom_start, atom_end, nkb_batch);

            // Compute vkb for this batch
            this->ppcell->getvnl_batch(this->ctx, *this->ucell, this->ik,
                                       atom_start, atom_end, vkb_batch_gpu_);

            // Compute becp_batch = vkb_batch^† × psi
            gemm_op()(this->ctx, 'C', 'N', nkb_batch, nbands, this->npw,
                     &this->one, vkb_batch_gpu_, this->npw,
                     tmpsi_in, max_npw,
                     &this->zero, becp_batch, nkb_batch);

            // Accumulate: hpsi += vkb_batch × ps_batch
            compute_ps_batch(becp_batch, nkb_batch, atom_start, atom_end);
            gemm_op()(this->ctx, 'N', 'T', this->npw, nbands, nkb_batch,
                     &this->one, vkb_batch_gpu_, this->npw,
                     ps_batch, nbands,
                     &this->one, tmhpsi, max_npw);  // Note: alpha=1 for accumulation
        }
    } else {
        // Original non-batched path (unchanged)
        ...
    }
}
```

#### 4. VNL Batch Computation

**File:** `source/module_hamilt_pw/hamilt_pwdft/VNL_in_pw.cpp`

Add batched version of `getvnl()`:

```cpp
void pseudopot_cell_vnl::getvnl_batch(Device* ctx, const UnitCell& ucell,
                                      const int ik, int atom_start, int atom_end,
                                      T* vkb_batch) {
    // Similar to getvnl() but only process atoms [atom_start, atom_end)
    int jkb = 0;
    for (int iat = atom_start; iat < atom_end; iat++) {
        int it = ucell.iat2it[iat];  // atom type
        int ia = ucell.iat2ia[iat];  // atom index within type

        // Compute vkb for this atom (same logic as original getvnl)
        for (int ih = 0; ih < ucell.atoms[it].ncpp.nh; ih++) {
            // vkb_batch[jkb, :] = ylm * vq * structure_factor * phase
            ...
            jkb++;
        }
    }
}
```

### Memory Savings

For a typical system (200 atoms, 10 projectors/atom, 10000 plane waves, batch_size=20):
- **Before:** 2000 × 10000 × 16 bytes = 320 MB
- **After:** 200 × 10000 × 16 bytes = 32 MB (10 batches)
- **Reduction:** 90% for vkb component

**Performance overhead:** ~7-10% from:
- Multiple `getvnl_batch` calls: ~3-4%
- Smaller GEMM operations: ~4-6%
- Can be optimized with cuBLAS batched GEMM in future

### Implementation Tasks

1. Create VKBBatchManager class with auto-tuning logic
2. Implement `getvnl_batch()` method
3. Modify Nonlocal::act() to support batching
4. Add `compute_ps_batch()` helper function
5. Implement batch-by-atom-type optimization
6. Add unit tests for batch computation
7. Validate numerical correctness vs non-batched

### Acceptance Criteria

- ✓ All Phase 1 criteria still met
- ✓ VKB batching tests pass
- ✓ Combined memory reduction ≥75%
- ✓ Combined performance overhead ≤15%
- ✓ Numerical validation passes (energy diff <1e-6 eV)

## Phase 3: Real-Space Extension (Optional)

**Timeline:** 4-6 weeks (optional future work)
**Goal:** Further optimize memory for very large systems

### Overview

Phase 3 is an **optional future enhancement** that replaces reciprocal-space vkb computation with real-space projections. This phase is designed to be added later without disrupting Phases 1-2.

### Architectural Hooks in Phase 2

To enable Phase 3 without refactoring, Phase 2 includes these extension points:

#### Abstract Projector Interface

**New file:** `source/module_hamilt_pw/hamilt_pwdft/projector_base.h`

```cpp
class ProjectorBase {
public:
    virtual void compute_becp(const T* psi, T* becp, int nbands) = 0;
    virtual void apply_deeq(const T* becp, T* hpsi, int nbands) = 0;
    virtual size_t get_memory_size() const = 0;
};

// Phase 2 implementation
class ReciprocalProjector : public ProjectorBase {
    // Uses vkb in reciprocal space (current batching approach)
};

// Phase 3 implementation (future)
class RealSpaceProjector : public ProjectorBase {
    // Uses beta(r) in real space
};
```

#### Nonlocal Operator Abstraction

```cpp
// In Nonlocal::init()
if (PARAM.inp.projector_type == "reciprocal") {
    projector_ = std::make_unique<ReciprocalProjector>(...);
} else if (PARAM.inp.projector_type == "realspace") {
    projector_ = std::make_unique<RealSpaceProjector>(...);
}

// In Nonlocal::act()
projector_->compute_becp(tmpsi_in, becp, nbands);
projector_->apply_deeq(becp, tmhpsi, nbands);
```

### Real-Space Method Design

#### Real-Space Projector Storage

Instead of storing `vkb(nkb, npw)`, store localized `beta_r(natom, nproj, nr_local)`:

```cpp
class RealSpaceProjector : public ProjectorBase {
    std::vector<T*> beta_r_;  // beta_r[iat] points to real-space projector
    int* nr_local_;           // Number of grid points in local region per atom
    int* grid_offset_;        // Starting grid index for each atom's region

    // Typical nr_local ~ 1000-5000 points per atom (vs npw ~ 10000-50000)
};
```

**Memory advantage:** For 200 atoms with 10 projectors each, 2000 points/atom:
- Real-space: 200 × 10 × 2000 × 16 bytes = 64 MB
- Reciprocal-space (batched): 200 × 10000 × 16 bytes = 32 MB (per batch)
- Real-space scales with natom, reciprocal scales with npw

#### Computation Flow

```cpp
void RealSpaceProjector::compute_becp(const T* psi_G, T* becp, int nbands) {
    // 1. FFT: psi(G) → psi(r)
    for (int ib = 0; ib < nbands; ib++) {
        fft_->forward(psi_G + ib*npw, psi_r_buffer_);

        // 2. Local integration: becp[ib,jkb] = ∫ beta_r[jkb](r) * psi(r) dr
        for (int iat = 0; iat < natom; iat++) {
            for (int iproj = 0; iproj < nproj[iat]; iproj++) {
                int jkb = atom_to_jkb(iat, iproj);
                becp[ib*nkb + jkb] = integrate_local(
                    beta_r_[iat] + iproj*nr_local_[iat],
                    psi_r_buffer_ + grid_offset_[iat],
                    nr_local_[iat]
                );
            }
        }
    }
}

void RealSpaceProjector::apply_deeq(const T* becp, T* hpsi_G, int nbands) {
    // 1. Compute ps = deeq × becp (same as reciprocal space)
    compute_ps(becp, ps, nbands);

    // 2. Real-space accumulation: hpsi(r) += Σ_jkb ps[jkb] * beta_r[jkb](r)
    for (int ib = 0; ib < nbands; ib++) {
        set_zero(hpsi_r_buffer_);
        for (int iat = 0; iat < natom; iat++) {
            for (int iproj = 0; iproj < nproj[iat]; iproj++) {
                int jkb = atom_to_jkb(iat, iproj);
                accumulate_local(
                    hpsi_r_buffer_ + grid_offset_[iat],
                    beta_r_[iat] + iproj*nr_local_[iat],
                    ps[ib*nkb + jkb],
                    nr_local_[iat]
                );
            }
        }

        // 3. FFT: hpsi(r) → hpsi(G), accumulate to output
        fft_->backward(hpsi_r_buffer_, hpsi_G_temp_);
        accumulate(hpsi_G + ib*npw, hpsi_G_temp_, npw);
    }
}
```

#### FFT Integration

Reuse existing FFT infrastructure:
- `ModulePW::PW_Basis` already has FFT methods
- Use `wfcpw->recip2real()` and `wfcpw->real2recip()`
- Leverage cuFFT/rocFFT for GPU acceleration

### Performance Trade-offs

**Pros:**
- Better memory scaling for large systems (O(natom) vs O(npw))
- No vkb recomputation overhead
- Naturally localized (good for domain decomposition)

**Cons:**
- Requires 2 FFTs per band (forward + backward): ~10-15% overhead
- Real-space integration adds computation
- More complex implementation

**When to use:**
- Very large systems (>500 atoms, >50000 plane waves)
- Memory-constrained GPUs
- When FFT overhead is acceptable

### Implementation Tasks

1. Implement `RealSpaceProjector` class (2 weeks)
2. Generate `beta_r` from pseudopotential files (1 week)
3. Optimize real-space integration kernels (1-2 weeks)
4. Validation and performance tuning (1 week)

### Acceptance Criteria

- ✓ Real-space projector tests pass
- ✓ Energy differences <1e-6 eV vs Phase 2
- ✓ Memory scaling better for large systems
- ✓ Performance competitive with Phase 2

## Testing and Validation Strategy

### Unit Tests

#### Phase 1: Psi K-Point Paging

**File:** `source/module_psi/test/psi_paging_test.cpp`

```cpp
TEST_F(PsiPagingTest, LoadStoreKPoint) {
    // Test basic load/store operations
    Psi<complex<double>, DEVICE_GPU> psi(nk, nband, nbasis, PsiStorageMode::PAGED_GPU);

    psi.load_k_to_gpu(5);
    EXPECT_EQ(psi.get_current_k_gpu(), 5);

    modify_gpu_data(psi.get_pointer());
    psi.store_k_from_gpu(5);

    verify_cpu_data(psi.get_cpu_pointer(5));
}

TEST_F(PsiPagingTest, DoubleBuffering) {
    // Test overlapped transfer with double buffering
    // Verify no data corruption when prefetching
}

TEST_F(PsiPagingTest, MemoryFootprint) {
    // Verify GPU memory usage is 1-2 k-points, not nk k-points
    size_t mem_before = get_gpu_memory_usage();
    Psi<complex<double>, DEVICE_GPU> psi(100, 200, 10000, PsiStorageMode::PAGED_GPU);
    size_t mem_after = get_gpu_memory_usage();

    size_t expected = 2 * 200 * 10000 * sizeof(complex<double>);
    EXPECT_NEAR(mem_after - mem_before, expected, expected * 0.1);
}
```

#### Phase 2: VKB Atom Batching

**File:** `source/module_hamilt_pw/hamilt_pwdft/test/vkb_batch_test.cpp`

```cpp
TEST_F(VKBBatchTest, BatchComputation) {
    // Compare batched vs non-batched results
    compute_nonlocal_batched(psi, hpsi_batched);
    compute_nonlocal_original(psi, hpsi_original);

    EXPECT_ARRAY_NEAR(hpsi_batched, hpsi_original, 1e-12);
}

TEST_F(VKBBatchTest, BatchSizeAutoTuning) {
    // Test auto-tuning logic
    int batch_size = compute_optimal_batch_size(200, 10, 10000, 8*1024*1024*1024);
    EXPECT_GT(batch_size, 0);
    EXPECT_LE(batch_size, 200);
}

TEST_F(VKBBatchTest, GetvnlBatchCorrectness) {
    // Verify getvnl_batch produces same results as getvnl
    getvnl_batch(ik, 0, 20, vkb_batch);
    getvnl(ik, vkb_full);

    EXPECT_ARRAY_NEAR(vkb_batch, vkb_full, 1e-12);
}
```

### Integration Tests

#### Existing Test Suite Validation

```bash
# Phase 1 validation
cd tests/integrate
export ABACUS_DEVICE_MEMORY_MODE=paged
./Autotest.sh -r "11_PW_GPU.*"

# Phase 2 validation
export ABACUS_VKB_BATCH_ATOMS=20
./Autotest.sh -r "11_PW_GPU.*"
```

**Acceptance criteria:**
- All tests pass with identical results (energy diff < 1e-6 eV)
- Forces and stress match within 1e-5 eV/Å and 1e-3 GPa
- Eigenvalues match within 1e-8 eV

#### New Test Cases

**Test:** `tests/integrate/11_PW_GPU/GPU_101_paged_memory/`

Test with many k-points (100+) that wouldn't fit in GPU without paging:

```
INPUT:
    calculation  scf
    basis_type   pw
    device       gpu
    device_memory_mode  paged
    ntype        1
    nbands       200
    ecutwfc      50
    ks_solver    dav

KPT:
    0
    Gamma
    10 10 10 0 0 0  # 1000 k-points
```

### Performance Benchmarks

#### Benchmark Systems

1. **Small:** Si2 (2 atoms, 8×8×8 k-points, 50 Ry cutoff)
2. **Medium:** GaAs (8 atoms, 6×6×6 k-points, 60 Ry cutoff)
3. **Large:** TiO2 supercell (96 atoms, 4×4×4 k-points, 50 Ry cutoff)

#### Metrics to Track

```bash
# Run benchmark script
./benchmark_memory_optimization.sh

# Outputs:
# 1. GPU memory usage (peak, average)
# 2. Wall time per SCF iteration
# 3. Davidson iterations per k-point
# 4. CPU-GPU transfer time
# 5. Total SCF convergence time
```

#### Target Performance

| Phase | Memory Reduction | Time Overhead | Status |
|-------|------------------|---------------|--------|
| Phase 1 | 90-95% (psi) | <8% | Must meet |
| Phase 2 | 85-90% (vkb) | <10% | Must meet |
| Combined | 75-85% (total) | <15% | Must meet |

### Memory Profiling

#### Tools

- `nvidia-smi` / `rocm-smi` for GPU memory monitoring
- CUDA/ROCm profiler for detailed memory traces
- Custom memory tracker in ABACUS

#### Profiling Script

```bash
#!/bin/bash
# profile_memory.sh

# Baseline (full GPU)
export ABACUS_DEVICE_MEMORY_MODE=full_gpu
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem_baseline.log &
PROFILER_PID=$!
mpirun -np 1 abacus > output_baseline.log
kill $PROFILER_PID

# Phase 1 (paged)
export ABACUS_DEVICE_MEMORY_MODE=paged
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem_phase1.log &
PROFILER_PID=$!
mpirun -np 1 abacus > output_phase1.log
kill $PROFILER_PID

# Compare
python analyze_memory.py mem_baseline.log mem_phase1.log
```

### Numerical Validation

#### Energy Conservation

```python
# Verify total energy matches across modes
E_baseline = parse_energy("output_baseline.log")
E_paged = parse_energy("output_phase1.log")
assert abs(E_baseline - E_paged) < 1e-6, "Energy mismatch"
```

#### Wavefunction Overlap

```python
# Verify wavefunctions are identical
psi_baseline = read_wavefunction("OUT.baseline/wfc_k1.dat")
psi_paged = read_wavefunction("OUT.paged/wfc_k1.dat")
overlap = compute_overlap(psi_baseline, psi_paged)
assert overlap > 0.999999, "Wavefunction mismatch"
```

### Continuous Integration

**File:** `.github/workflows/gpu_memory_test.yml`

```yaml
name: GPU Memory Optimization Tests

on: [push, pull_request]

jobs:
  test-phase1:
    runs-on: gpu-runner
    steps:
      - name: Build with GPU support
        run: cmake -B build -DUSE_CUDA=ON && cmake --build build

      - name: Run paged memory tests
        run: |
          cd tests/integrate
          export ABACUS_DEVICE_MEMORY_MODE=paged
          ./Autotest.sh -r "11_PW_GPU_101.*"

      - name: Check memory usage
        run: python scripts/check_memory_reduction.py --threshold 0.90
```

## Error Handling and Edge Cases

### Memory Allocation Failures

#### GPU Out-of-Memory Handling

```cpp
// In Psi::init() for PAGED_GPU mode
try {
    cudaMalloc(&psi_gpu_buffer_, nband * nbasis * sizeof(T));
} catch (std::bad_alloc& e) {
    // Fallback to CPU-only mode
    ModuleBase::WARNING("Psi", "GPU allocation failed, falling back to CPU mode");
    storage_mode_ = PsiStorageMode::ALL_CPU;
    // Continue execution on CPU (slower but functional)
}

// In VKBBatchManager::init()
int batch_size = compute_optimal_batch_size(...);
while (batch_size > 0) {
    try {
        cudaMalloc(&vkb_batch_gpu_, batch_size * nproj * npw * sizeof(T));
        break;  // Success
    } catch (std::bad_alloc& e) {
        batch_size /= 2;  // Try smaller batch
    }
}
if (batch_size == 0) {
    ModuleBase::WARNING_QUIT("VKBBatchManager",
        "Cannot allocate even 1 atom batch on GPU");
}
```

**Graceful Degradation:**
- If paged mode fails → fall back to CPU-only execution
- If batching fails → try smaller batch sizes down to 1 atom
- If all fails → clear error message and exit

### Data Transfer Failures

#### CUDA Transfer Error Handling

```cpp
void Psi::load_k_to_gpu(int ik) {
    cudaError_t err = cudaMemcpy(psi_gpu_buffer_,
                                  psi_cpu_ + ik * nband * nbasis,
                                  nband * nbasis * sizeof(T),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::string msg = "Failed to transfer k-point " + std::to_string(ik)
                        + " to GPU: " + cudaGetErrorString(err);
        ModuleBase::WARNING_QUIT("Psi::load_k_to_gpu", msg);
    }
    current_k_gpu_ = ik;
}
```

#### Asynchronous Transfer Validation

```cpp
// After async transfer, check for errors before using data
cudaMemcpyAsync(..., stream);
cudaError_t err = cudaStreamSynchronize(stream);
if (err != cudaSuccess) {
    // Handle error
}
```

### Edge Cases

#### 1. Single K-Point Systems

```cpp
// In HSolverPW::solve()
if (nks == 1 && psi.get_storage_mode() == PsiStorageMode::PAGED_GPU) {
    // No benefit from paging with single k-point
    // Could auto-switch to ALL_GPU mode
    if (PARAM.inp.device_memory_mode == "auto") {
        psi.set_storage_mode(PsiStorageMode::ALL_GPU);
    }
}
```

#### 2. Very Small Systems (nkb < batch_size)

```cpp
// In VKBBatchManager::init()
if (total_nkb < target_batch_size) {
    // System is small enough to fit in one batch
    nbatch_ = 1;
    batch_size_[0] = total_atoms;
    // Effectively no batching overhead
}
```

#### 3. Extremely Large Bands (nband > npw)

```cpp
// Sanity check in DiagoDavid constructor
if (nband > dim) {
    ModuleBase::WARNING_QUIT("DiagoDavid",
        "Number of bands exceeds basis size");
}
```

#### 4. Mixed Precision Edge Cases

```cpp
// Ensure consistent precision across CPU and GPU
static_assert(sizeof(T) == sizeof(device_T),
              "CPU and GPU data types must have same size");
```

### Backward Compatibility

#### Default Behavior

```cpp
// In read_input_item_elec_stru.cpp
if (PARAM.inp.device_memory_mode == "") {
    // Auto-detect: use paged mode if many k-points
    if (nks > 10 && device == "gpu") {
        PARAM.inp.device_memory_mode = "paged";
    } else {
        PARAM.inp.device_memory_mode = "full_gpu";  // Original behavior
    }
}
```

#### Legacy Code Path

```cpp
// All original code paths remain functional
if (psi.get_storage_mode() == PsiStorageMode::ALL_GPU) {
    // Original behavior - no paging overhead
    // Existing tests continue to work unchanged
}
```

#### INPUT File Compatibility

```
# Old INPUT files work without modification
device  gpu
ks_solver  dav
# Uses auto-detection for memory mode

# New INPUT files can explicitly control
device  gpu
device_memory_mode  paged  # or "full_gpu"
vkb_batch_atoms  20
```

### Validation and Assertions

#### Runtime Checks

```cpp
// In Psi::load_k_to_gpu()
assert(ik >= 0 && ik < nk_);
assert(psi_cpu_ != nullptr);
assert(psi_gpu_buffer_ != nullptr);

// In VKBBatchManager::get_batch_info()
assert(ibatch >= 0 && ibatch < nbatch_);
assert(atom_start < atom_end);
assert(atom_end <= total_atoms_);
```

#### Debug Mode

```cpp
#ifdef DEBUG
    // Extra validation in debug builds
    void Psi::validate_k_data(int ik) {
        // Check for NaN/Inf
        // Verify normalization
        // Check orthogonality
    }
#endif
```

### Error Messages

#### User-Friendly Messages

```cpp
// Bad: "cudaMalloc failed"
// Good:
ModuleBase::WARNING("GPU Memory",
    "Insufficient GPU memory for full calculation.\n"
    "  Required: " + std::to_string(required_mem/1e9) + " GB\n"
    "  Available: " + std::to_string(available_mem/1e9) + " GB\n"
    "  Suggestion: Using paged memory mode (device_memory_mode=paged)\n"
    "  or reduce system size / k-points");
```

#### Diagnostic Information

```cpp
// On failure, print useful debug info
void print_memory_diagnostic() {
    std::cout << "=== GPU Memory Diagnostic ===" << std::endl;
    std::cout << "Total GPU memory: " << get_total_gpu_mem()/1e9 << " GB" << std::endl;
    std::cout << "Used GPU memory: " << get_used_gpu_mem()/1e9 << " GB" << std::endl;
    std::cout << "Free GPU memory: " << get_free_gpu_mem()/1e9 << " GB" << std::endl;
    std::cout << "Psi memory required: " << psi_mem_required/1e9 << " GB" << std::endl;
    std::cout << "VKB memory required: " << vkb_mem_required/1e9 << " GB" << std::endl;
    std::cout << "Storage mode: " << storage_mode_to_string() << std::endl;
}
```

### Recovery Mechanisms

#### Checkpoint and Restart

```cpp
// If transfer fails mid-calculation, save state
void save_checkpoint(int ik_failed) {
    // Save converged k-points to disk
    for (int ik = 0; ik < ik_failed; ik++) {
        write_wavefunction(ik, psi.get_cpu_pointer(ik));
    }
    // User can restart from checkpoint
}
```

#### Automatic Retry

```cpp
// Retry failed transfers with exponential backoff
int retry_count = 0;
const int max_retries = 3;
while (retry_count < max_retries) {
    cudaError_t err = cudaMemcpy(...);
    if (err == cudaSuccess) break;

    retry_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100 * retry_count));
}
```

## Implementation Timeline

### Phase 1: Psi K-Point Paging (Weeks 1-3)

**Week 1:**
- Day 1-2: Add PsiStorageMode enum and member variables
- Day 3-4: Implement load_k_to_gpu() and store_k_from_gpu()
- Day 5: Modify Psi constructor for PAGED_GPU mode

**Week 2:**
- Day 1-2: Update HSolverPW k-point loop
- Day 3-4: Implement double buffering with CUDA streams
- Day 5: Write unit tests

**Week 3:**
- Day 1-3: Run integration tests, fix bugs
- Day 4-5: Performance profiling and optimization

**Deliverable:** Working psi paging with 90%+ memory reduction, <8% overhead

### Phase 2: VKB Atom Batching (Weeks 4-6)

**Week 4:**
- Day 1-2: Create VKBBatchManager class
- Day 3-4: Implement auto-tuning logic
- Day 5: Implement getvnl_batch()

**Week 5:**
- Day 1-2: Modify Nonlocal::act() for batching
- Day 3-4: Add compute_ps_batch() helper
- Day 5: Batch-by-atom-type optimization

**Week 6:**
- Day 1-2: Write unit tests for batching
- Day 3-4: Integration testing and validation
- Day 5: Performance tuning

**Deliverable:** Combined 75%+ memory reduction, <15% overhead

### Phase 3: Real-Space Extension (Weeks 7-12, Optional)

**Weeks 7-8:**
- Implement RealSpaceProjector class
- Design beta_r storage and generation

**Weeks 9-10:**
- Implement real-space integration kernels
- FFT integration and optimization

**Weeks 11-12:**
- Validation against Phase 2
- Performance tuning and documentation

**Deliverable:** Real-space projector option for very large systems

## Code Structure

```
source/module_hsolver/
├── memory_manager/           # New: Memory management module
│   ├── psi_pager.h/cpp      # Wavefunction paging management
│   └── vkb_cache.h/cpp      # VKB caching strategy
├── diago_david.cpp          # Modified: Support paged psi
├── diago_dav_subspace.cpp   # Modified: Support paged psi
└── hsolver_pw.cpp           # Modified: K-point loop with transfers

source/module_hamilt_pw/hamilt_pwdft/
├── operator_pw/
│   └── nonlocal_pw.cpp      # Modified: Support vkb batching
├── VNL_in_pw.cpp            # Modified: getvnl_batch support
├── vkb_batch_manager.h/cpp  # New: VKB batch management
├── projector_base.h         # New: Abstract projector interface
└── realspace_nonlocal/      # New: Real-space projectors (Phase 3)
    ├── beta_realspace.h/cpp
    └── becp_realspace.h/cpp

source/module_psi/
└── psi.h                    # Modified: Add PAGED_GPU mode

source/module_parameter/
└── input_parameter.h        # Modified: Add new parameters
```

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-GPU transfer bottleneck | Performance >20% | Use CUDA streams for async transfer; prefetch next k-point |
| Real-space precision issues | Numerical errors | Use dense real-space grid; validate against reciprocal space |
| Batching reduces GEMM efficiency | Performance >30% | Use cuBLAS batched GEMM; optimize batch size |
| Code complexity increases | Maintenance burden | Comprehensive unit tests; clear documentation; keep legacy path |

## References

1. **VASP Real-Space Projectors:**
   - G. Kresse, J. Furthmüller, PRB 54, 11169 (1996)

2. **QE On-the-Fly Computation:**
   - P. Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009)

3. **GPU Memory Optimization:**
   - NVIDIA CUDA Best Practices Guide
   - Unified Memory and Stream Concurrency

4. **Davidson Diagonalization:**
   - E.R. Davidson, J. Comput. Phys. 17, 87 (1975)

## Appendix: Configuration Examples

### Example 1: Auto-Detect Mode (Recommended)

```
INPUT:
    calculation  scf
    basis_type   pw
    device       gpu
    # device_memory_mode auto-detected based on nks
    ks_solver    dav
```

### Example 2: Explicit Paged Mode

```
INPUT:
    calculation  scf
    basis_type   pw
    device       gpu
    device_memory_mode  paged
    vkb_batch_atoms  20  # or 0 for auto-tune
    ks_solver    dav
```

### Example 3: Force Full GPU Mode

```
INPUT:
    calculation  scf
    basis_type   pw
    device       gpu
    device_memory_mode  full_gpu
    ks_solver    dav
```

### Example 4: Real-Space Projectors (Phase 3)

```
INPUT:
    calculation  scf
    basis_type   pw
    device       gpu
    device_memory_mode  paged
    projector_type  realspace  # or "reciprocal"
    ks_solver    dav
```

---

**End of Design Document**
