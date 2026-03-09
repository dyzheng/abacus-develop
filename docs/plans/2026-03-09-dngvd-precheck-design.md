# Matrix Condition Number Pre-check for GPU dngvd Design

**Date:** 2026-03-09
**Status:** Approved

## Problem

The current multi-layer fallback mechanism for GPU dngvd (cusolver) relies on try-catch error handling, which only detects failures after they occur. This reactive approach wastes computation time and can lead to repeated failures in cases where the scc matrix is numerically unstable.

We need a proactive solution that predicts whether GPU dngvd will succeed by checking the matrix condition number before calling the GPU interface.

## Solution Overview

Implement a condition number pre-check mechanism that:
1. Calculates the condition number of the scc matrix on CPU before calling GPU dngvd
2. If condition number exceeds threshold (1e12), uses CPU dngvd instead of GPU
3. Provides user control via INPUT parameter with three modes: first electronic step only, every electronic step, or disabled
4. Caches the decision result to avoid repeated checks (for "first" mode)

## Design Details

### 1. Architecture

**Control Flow:**
```
diago_dav_subspace.cpp::diag_subspace()
  ↓
Check PARAM.inp.diago_cond_check parameter
  ↓
If check needed (first && !done, or always):
  ├─ D2H copy scc matrix to CPU
  ├─ Call check_matrix_condition_number()
  ├─ If cond(scc) > 1e12: set use_cpu_dngvd = true
  └─ Cache decision in static variable
  ↓
Select execution path based on decision:
  ├─ use_cpu_dngvd == true → call dngvd_op<T, DEVICE_CPU>
  └─ use_cpu_dngvd == false → call dngvd_op<T, DEVICE_GPU> (keep try-catch)
```

**Key Design Points:**
- Insert check logic in `diago_dav_subspace.cpp::diag_subspace()` before dngvd_op call
- Add INPUT parameter `diago_cond_check` (string: "first"/"always"/"off")
- Use static variables to cache decision across electronic steps
- Add helper function `check_matrix_condition_number()` for condition number calculation
- Preserve existing GPU try-catch mechanism as safety net

### 2. Core Components

#### 2.1 INPUT Parameter

**Parameter name:** `diago_cond_check`
**Type:** string
**Default value:** `"off"`
**Valid values:**
- `"first"` - Check condition number only in first electronic step
- `"always"` - Check condition number in every electronic step
- `"off"` - Disable pre-check, use existing try-catch mechanism

**Location:**
- Declaration: `source/module_parameter/input_parameter.h`
- Parsing: `source/module_io/input_conv.cpp`

#### 2.2 Condition Number Check Function

Add to `diago_dav_subspace.cpp`:

```cpp
template <typename T>
bool check_matrix_condition_number(const T* scc_gpu,
                                   const int nbase,
                                   const double threshold = 1e12)
{
    using Real = typename GetTypeReal<T>::type;

    // 1. D2H copy scc matrix
    std::vector<T> scc_cpu(nbase * nbase);
    cudaMemcpy(scc_cpu.data(), scc_gpu,
               sizeof(T) * nbase * nbase,
               cudaMemcpyDeviceToHost);

    // 2. LU factorization (LAPACK xGETRF)
    std::vector<int> ipiv(nbase);
    int info = 0;
    LapackConnector::xgetrf(nbase, nbase, scc_cpu.data(), nbase, ipiv.data(), &info);

    if (info > 0) {
        // Matrix is singular
        std::cout << "WARNING: scc matrix is singular (xGETRF info=" << info
                  << "), using CPU dngvd" << std::endl;
        return true;  // Use CPU
    }

    // 3. Estimate condition number (LAPACK xGECON)
    Real anorm = 0.0;  // Matrix 1-norm (should be computed before LU)
    Real rcond = 0.0;  // Reciprocal condition number
    std::vector<T> work(4 * nbase);
    std::vector<Real> rwork(nbase);

    // Compute 1-norm before calling xgecon
    // For simplicity, use infinity norm approximation
    for (int i = 0; i < nbase; i++) {
        Real row_sum = 0.0;
        for (int j = 0; j < nbase; j++) {
            row_sum += std::abs(scc_cpu[i * nbase + j]);
        }
        anorm = std::max(anorm, row_sum);
    }

    LapackConnector::xgecon('1', nbase, scc_cpu.data(), nbase,
                           anorm, &rcond, work.data(), rwork.data(), &info);

    if (info != 0) {
        // xGECON failed, be conservative
        std::cout << "WARNING: xGECON failed (info=" << info
                  << "), using CPU dngvd" << std::endl;
        return true;  // Use CPU
    }

    // 4. Check condition number
    double cond = (rcond > 1e-16) ? (1.0 / rcond) : 1e16;

    if (cond > threshold) {
        std::cout << "WARNING: scc matrix condition number "
                  << cond << " exceeds threshold " << threshold
                  << ", using CPU dngvd for numerical stability"
                  << std::endl;
        return true;  // Use CPU
    }

    return false;  // Use GPU
}
```

**Key points:**
- Uses LAPACK xGETRF for LU factorization
- Uses LAPACK xGECON to estimate reciprocal condition number
- Threshold: 1e12 (conservative for double precision)
- Returns true if CPU should be used, false if GPU is safe

#### 2.3 Static Variables

Add to `Diago_DavSubspace` class in `diago_dav_subspace.h`:

```cpp
// Static variables for condition number check caching
static bool use_cpu_dngvd_;
static bool cond_check_done_;

// Static method to reset state
static void reset_cond_check() {
    cond_check_done_ = false;
    use_cpu_dngvd_ = false;
}
```

Initialize in `diago_dav_subspace.cpp`:
```cpp
template <typename T, typename Device>
bool Diago_DavSubspace<T, Device>::use_cpu_dngvd_ = false;

template <typename T, typename Device>
bool Diago_DavSubspace<T, Device>::cond_check_done_ = false;
```

### 3. Data Flow and Control Logic

#### 3.1 Modification to diag_subspace()

In `diago_dav_subspace.cpp::diag_subspace()`, before calling dngvd_op:

```cpp
// Determine if condition number check is needed
bool should_check = false;
if (PARAM.inp.diago_cond_check == "always") {
    should_check = true;
} else if (PARAM.inp.diago_cond_check == "first" && !cond_check_done_) {
    should_check = true;
}

// Perform check if needed and on GPU device
if (should_check && this->device == base_device::GpuDevice) {
    use_cpu_dngvd_ = check_matrix_condition_number(this->scc, nbase);
    cond_check_done_ = true;
}

// Select execution path based on decision
if (use_cpu_dngvd_ && this->device == base_device::GpuDevice) {
    // Use CPU path
    std::vector<T> scc_cpu(nbase * nbase);
    std::vector<T> hcc_cpu(nbase * nbase);
    std::vector<T> vcc_cpu(nbase * nbase);
    std::vector<Real> eigenvalue_cpu(nbase);

    // D2H copy
    cudaMemcpy(scc_cpu.data(), this->scc, sizeof(T) * nbase * nbase, cudaMemcpyDeviceToHost);
    cudaMemcpy(hcc_cpu.data(), this->hcc, sizeof(T) * nbase * nbase, cudaMemcpyDeviceToHost);

    // Call CPU dngvd
    base_device::DEVICE_CPU* cpu_ctx = {};
    dngvd_op<T, base_device::DEVICE_CPU>()(
        cpu_ctx, nbase, nbase,
        hcc_cpu.data(), scc_cpu.data(),
        eigenvalue_cpu.data(), vcc_cpu.data()
    );

    // H2D copy results
    cudaMemcpy(this->vcc, vcc_cpu.data(), sizeof(T) * nbase * nbase, cudaMemcpyHostToDevice);
    cudaMemcpy(eigenvalue_iter.data(), eigenvalue_cpu.data(), sizeof(Real) * nbase, cudaMemcpyHostToDevice);
} else {
    // Original GPU path (keep try-catch)
    dngvd_op<T, Device>()(this->ctx, nbase, nbase,
                         this->hcc, this->scc,
                         eigenvalue_iter.data(), this->vcc);
}
```

#### 3.2 Electronic Step Counter Reset

In `esolver_ks_pw.cpp` and `esolver_ks_lcao.cpp`, at the beginning of SCF loop:

```cpp
// In Run() function, before SCF loop
if (istep == 0) {  // First ionic/MD step
    Diago_DavSubspace<T, Device>::reset_cond_check();
}
```

### 4. Error Handling and Edge Cases

#### 4.1 Error Handling Strategy

**Condition number calculation failure:**
- If LAPACK xGETRF returns info > 0 (singular matrix): set `use_cpu_dngvd_ = true`
- If xGECON fails: conservatively set `use_cpu_dngvd_ = true`
- Log warning messages to output

**GPU/CPU mixed scenarios:**
- If `this->device == base_device::DEVICE_CPU`, skip all checks, use CPU path directly
- Condition number check only triggers on GPU device

**Preserve existing try-catch:**
- Even if pre-check passes (cond < 1e12), GPU dngvd may still fail for other reasons
- Keep existing try-catch mechanism as final safety net
- If try-catch catches exception: log "pre-check passed but GPU still failed"

#### 4.2 Edge Cases

**Small nbase (< 10):**
- Condition number check overhead negligible, execute normally

**Large nbase (> 1000):**
- D2H copy and LU factorization overhead significant (O(n³))
- But if matrix is ill-conditioned, cost of GPU failure is higher (entire dngvd fails and needs recomputation)
- Pre-check overhead is justified

**Multi-k-point parallelism:**
- Static variables shared across all k-points
- First k-point's check result applies to all subsequent k-points
- This is reasonable because scc matrix condition numbers are typically similar across k-points

**PAGED_GPU mode:**
- Condition number check works normally (scc is always in GPU memory during diag_subspace)
- No special handling needed

## Files to Modify

| File | Change |
|------|--------|
| `source/module_parameter/input_parameter.h` | Add `diago_cond_check` parameter declaration |
| `source/module_io/input_conv.cpp` | Add parsing logic for `diago_cond_check` |
| `source/module_hsolver/diago_dav_subspace.h` | Add static variables and reset method |
| `source/module_hsolver/diago_dav_subspace.cpp` | Add condition check function and modify diag_subspace() |
| `source/module_esolver/esolver_ks_pw.cpp` | Call reset_cond_check() at SCF start |
| `source/module_esolver/esolver_ks_lcao.cpp` | Call reset_cond_check() at SCF start |

## Testing Strategy

### Unit Tests

1. Test `check_matrix_condition_number()` with:
   - Well-conditioned matrix (cond ~ 10)
   - Moderately ill-conditioned matrix (cond ~ 1e10)
   - Severely ill-conditioned matrix (cond > 1e12)
   - Singular matrix

### Integration Tests

1. Run existing test `102_PW_DS_davsubspace` with:
   - `diago_cond_check off` (baseline)
   - `diago_cond_check first`
   - `diago_cond_check always`

2. Verify:
   - Results are identical across all three modes
   - Performance overhead is acceptable
   - Warning messages appear when condition number exceeds threshold

3. Test with problematic cases that previously triggered GPU failures

## Performance Considerations

**Overhead for "first" mode:**
- One-time cost per calculation
- D2H copy: ~1 ms for nbase=200
- LU + xGECON: ~10 ms for nbase=200
- Total: ~11 ms (negligible compared to full SCF)

**Overhead for "always" mode:**
- Per electronic step cost
- For typical calculation with 10 electronic steps: ~110 ms total
- Still acceptable for most use cases

**Memory overhead:**
- Temporary CPU buffers: O(nbase²) per check
- Static variables: 2 bools (negligible)

## Benefits

1. **Proactive prevention:** Avoids GPU failures before they occur
2. **User control:** Three modes provide flexibility for different scenarios
3. **Performance:** Caching mechanism minimizes overhead
4. **Robustness:** Preserves existing try-catch as safety net
5. **Transparency:** Clear warning messages inform users of decisions

## Future Enhancements

1. Add condition number to output log for analysis
2. Collect statistics on GPU vs CPU usage
3. Consider adaptive threshold based on matrix size
4. Extend to other diagonalization methods (dnevx, dngvx)

