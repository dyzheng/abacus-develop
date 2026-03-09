# Matrix Condition Number Pre-check for GPU dngvd Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement proactive condition number checking before GPU dngvd calls to prevent numerical instability failures.

**Architecture:** Add INPUT parameter `diago_cond_check` with three modes (first/always/off). Before calling GPU dngvd in `diag_subspace()`, check scc matrix condition number using LAPACK on CPU. If cond > 1e12, use CPU dngvd instead. Cache decision in static variables to minimize overhead.

**Tech Stack:** C++, CUDA, LAPACK (xGETRF, xGECON), CMake

---

## Task 1: Add INPUT Parameter

**Files:**
- Modify: `source/module_parameter/input_parameter.h`
- Modify: `source/module_io/input_conv.cpp`

**Step 1: Add parameter declaration**

In `source/module_parameter/input_parameter.h`, find the diagonalization parameters section (around line 64 where `diago_proc` is defined) and add:

```cpp
std::string diago_cond_check = "off";  ///< condition number check for GPU dngvd: "first", "always", "off"
```

**Step 2: Add parameter parsing**

In `source/module_io/input_conv.cpp`, find the function that reads diagonalization parameters and add:

```cpp
// Read diago_cond_check parameter
param.input.diago_cond_check = INPUT.read_string("diago_cond_check");

// Validate the value
if (param.input.diago_cond_check != "first" &&
    param.input.diago_cond_check != "always" &&
    param.input.diago_cond_check != "off")
{
    ModuleBase::WARNING_QUIT("Input_Conv",
        "diago_cond_check must be 'first', 'always', or 'off'");
}
```

**Step 3: Build to verify syntax**

```bash
cmake --build build -j$(nproc)
```

Expected: Compilation succeeds

**Step 4: Commit**

```bash
git add source/module_parameter/input_parameter.h source/module_io/input_conv.cpp
git commit -m "Feature(hsolver): add diago_cond_check INPUT parameter

Add new parameter to control condition number pre-check for GPU dngvd:
- 'first': check only in first electronic step
- 'always': check in every electronic step
- 'off': disable pre-check (default)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Add Static Variables to Diago_DavSubspace

**Files:**
- Modify: `source/module_hsolver/diago_dav_subspace.h`
- Modify: `source/module_hsolver/diago_dav_subspace.cpp`

**Step 1: Add static variable declarations**

In `source/module_hsolver/diago_dav_subspace.h`, in the `Diago_DavSubspace` class private section (after line 80), add:

```cpp
// Static variables for condition number check caching
static bool use_cpu_dngvd_;
static bool cond_check_done_;

public:
// Static method to reset state
static void reset_cond_check() {
    cond_check_done_ = false;
    use_cpu_dngvd_ = false;
}
```

**Step 2: Initialize static variables**

In `source/module_hsolver/diago_dav_subspace.cpp`, after the namespace declaration (around line 13), add:

```cpp
// Initialize static variables
template <typename T, typename Device>
bool Diago_DavSubspace<T, Device>::use_cpu_dngvd_ = false;

template <typename T, typename Device>
bool Diago_DavSubspace<T, Device>::cond_check_done_ = false;
```

**Step 3: Build to verify syntax**

```bash
cmake --build build -j$(nproc)
```

Expected: Compilation succeeds

**Step 4: Commit**

```bash
git add source/module_hsolver/diago_dav_subspace.h source/module_hsolver/diago_dav_subspace.cpp
git commit -m "Feature(hsolver): add static variables for dngvd condition check caching

Add static variables to cache condition number check decision:
- use_cpu_dngvd_: whether to use CPU instead of GPU
- cond_check_done_: whether check has been performed
- reset_cond_check(): method to reset state between calculations

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Implement Condition Number Check Function

**Files:**
- Modify: `source/module_hsolver/diago_dav_subspace.cpp`

**Step 1: Add helper function before class methods**

In `source/module_hsolver/diago_dav_subspace.cpp`, after the includes and before the namespace (around line 13), add:

```cpp
#include "module_base/lapack_connector.h"

namespace hsolver
{

// Helper function to check matrix condition number
template <typename T>
bool check_matrix_condition_number(const T* scc_gpu,
                                   const int nbase,
                                   const double threshold = 1e12)
{
    using Real = typename GetTypeReal<T>::type;

    // 1. D2H copy scc matrix
    std::vector<T> scc_cpu(nbase * nbase);
#if defined(__CUDA) || defined(__ROCM)
    cudaMemcpy(scc_cpu.data(), scc_gpu,
               sizeof(T) * nbase * nbase,
               cudaMemcpyDeviceToHost);
#else
    // If not GPU, scc_gpu is already on CPU
    std::copy(scc_gpu, scc_gpu + nbase * nbase, scc_cpu.begin());
#endif

    // 2. Compute 1-norm before LU factorization
    Real anorm = 0.0;
    for (int j = 0; j < nbase; j++) {
        Real col_sum = 0.0;
        for (int i = 0; i < nbase; i++) {
            col_sum += std::abs(scc_cpu[i + j * nbase]);
        }
        anorm = std::max(anorm, col_sum);
    }

    // 3. LU factorization (LAPACK xGETRF)
    std::vector<int> ipiv(nbase);
    int info = 0;
    LapackConnector::xgetrf(nbase, nbase, scc_cpu.data(), nbase, ipiv.data(), &info);

    if (info > 0) {
        // Matrix is singular
        std::cout << "WARNING: scc matrix is singular (xGETRF info=" << info
                  << "), using CPU dngvd" << std::endl;
        return true;  // Use CPU
    }

    // 4. Estimate condition number (LAPACK xGECON)
    Real rcond = 0.0;  // Reciprocal condition number
    std::vector<T> work(2 * nbase);
    std::vector<Real> rwork(2 * nbase);

    LapackConnector::xgecon('1', nbase, scc_cpu.data(), nbase,
                           anorm, &rcond, work.data(), rwork.data(), &info);

    if (info != 0) {
        // xGECON failed, be conservative
        std::cout << "WARNING: xGECON failed (info=" << info
                  << "), using CPU dngvd" << std::endl;
        return true;  // Use CPU
    }

    // 5. Check condition number
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

} // namespace hsolver
```

**Step 2: Build to verify syntax**

```bash
cmake --build build -j$(nproc)
```

Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add source/module_hsolver/diago_dav_subspace.cpp
git commit -m "Feature(hsolver): add condition number check function

Implement check_matrix_condition_number() using LAPACK:
- D2H copy scc matrix from GPU to CPU
- Compute matrix 1-norm
- LU factorization with xGETRF
- Estimate condition number with xGECON
- Return true if cond > 1e12 (use CPU), false otherwise

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Modify diag_subspace() to Use Condition Check

**Files:**
- Modify: `source/module_hsolver/diago_iter_assist.cpp`

**Step 1: Find diag_subspace() function**

Search for the function that calls dngvd_op in the diagonalization code:

```bash
grep -n "dngvd_op" source/module_hsolver/diago_iter_assist.cpp
```

**Step 2: Add condition check logic before dngvd_op call**

In `source/module_hsolver/diago_iter_assist.cpp`, in the `diag_subspace()` function, before the dngvd_op call, insert:

```cpp
// Determine if condition number check is needed
bool should_check = false;
if (PARAM.inp.diago_cond_check == "always") {
    should_check = true;
} else if (PARAM.inp.diago_cond_check == "first" && !Diago_DavSubspace<T, Device>::cond_check_done_) {
    should_check = true;
}

// Perform check if needed and on GPU device
if (should_check && device == base_device::GpuDevice) {
    Diago_DavSubspace<T, Device>::use_cpu_dngvd_ =
        check_matrix_condition_number(scc, nbase);
    Diago_DavSubspace<T, Device>::cond_check_done_ = true;
}

// Select execution path based on decision
if (Diago_DavSubspace<T, Device>::use_cpu_dngvd_ && device == base_device::GpuDevice) {
    // Use CPU path
    std::vector<T> scc_cpu(nbase * nbase);
    std::vector<T> hcc_cpu(nbase * nbase);
    std::vector<T> vcc_cpu(nbase * nbase);
    std::vector<Real> eigenvalue_cpu(nbase);

    // D2H copy
    cudaMemcpy(scc_cpu.data(), scc, sizeof(T) * nbase * nbase, cudaMemcpyDeviceToHost);
    cudaMemcpy(hcc_cpu.data(), hcc, sizeof(T) * nbase * nbase, cudaMemcpyDeviceToHost);

    // Call CPU dngvd
    base_device::DEVICE_CPU* cpu_ctx = {};
    dngvd_op<T, base_device::DEVICE_CPU>()(
        cpu_ctx, nbase, nbase,
        hcc_cpu.data(), scc_cpu.data(),
        eigenvalue_cpu.data(), vcc_cpu.data()
    );

    // H2D copy results
    cudaMemcpy(vcc, vcc_cpu.data(), sizeof(T) * nbase * nbase, cudaMemcpyHostToDevice);
    cudaMemcpy(eigenvalue, eigenvalue_cpu.data(), sizeof(Real) * nbase, cudaMemcpyHostToDevice);
} else {
    // Original GPU path (keep existing code)
    dngvd_op<T, Device>()(ctx, nbase, nbase, hcc, scc, eigenvalue, vcc);
}
```

**Step 3: Build to verify syntax**

```bash
cmake --build build -j$(nproc)
```

Expected: Compilation succeeds (may have warnings about unused variables)

**Step 4: Commit**

```bash
git add source/module_hsolver/diago_iter_assist.cpp
git commit -m "Feature(hsolver): integrate condition check into diag_subspace

Add condition number pre-check before GPU dngvd call:
- Check if pre-check is needed based on diago_cond_check parameter
- If cond > 1e12, use CPU dngvd with D2H/H2D copies
- Otherwise use original GPU path
- Cache decision in static variables

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Add Reset Calls in ESolver

**Files:**
- Modify: `source/module_esolver/esolver_ks_pw.cpp`
- Modify: `source/module_esolver/esolver_ks_lcao.cpp`

**Step 1: Add reset call in esolver_ks_pw.cpp**

In `source/module_esolver/esolver_ks_pw.cpp`, find the `Run()` function and add at the beginning of the SCF loop (before the first k-point loop):

```cpp
// Reset condition number check state at the start of each ionic/MD step
if (istep == 0) {
    hsolver::Diago_DavSubspace<std::complex<double>, base_device::DEVICE_GPU>::reset_cond_check();
    hsolver::Diago_DavSubspace<std::complex<float>, base_device::DEVICE_GPU>::reset_cond_check();
}
```

**Step 2: Add reset call in esolver_ks_lcao.cpp**

In `source/module_esolver/esolver_ks_lcao.cpp`, find the `Run()` function and add at the beginning of the SCF loop:

```cpp
// Reset condition number check state at the start of each ionic/MD step
if (istep == 0) {
    hsolver::Diago_DavSubspace<double, base_device::DEVICE_GPU>::reset_cond_check();
    hsolver::Diago_DavSubspace<std::complex<double>, base_device::DEVICE_GPU>::reset_cond_check();
}
```

**Step 3: Build to verify syntax**

```bash
cmake --build build -j$(nproc)
```

Expected: Compilation succeeds

**Step 4: Commit**

```bash
git add source/module_esolver/esolver_ks_pw.cpp source/module_esolver/esolver_ks_lcao.cpp
git commit -m "Feature(hsolver): reset condition check state in esolvers

Call reset_cond_check() at the start of each ionic/MD step to ensure
condition number check is performed for the first electronic step when
diago_cond_check='first' mode is enabled.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Integration Testing

**Files:**
- Test: `tests/integrate/102_PW_DS_davsubspace/`

**Step 1: Test with diago_cond_check off (baseline)**

```bash
cd tests/integrate/102_PW_DS_davsubspace
cp INPUT INPUT.backup
# Ensure diago_cond_check is not set (defaults to "off")
OMP_NUM_THREADS=2 mpirun -np 4 abacus > log-off 2>&1
```

Expected: Test completes successfully

**Step 2: Test with diago_cond_check first**

```bash
echo "diago_cond_check first" >> INPUT
OMP_NUM_THREADS=2 mpirun -np 4 abacus > log-first 2>&1
```

Expected: Test completes successfully, results identical to baseline

**Step 3: Test with diago_cond_check always**

```bash
sed -i 's/diago_cond_check first/diago_cond_check always/' INPUT
OMP_NUM_THREADS=2 mpirun -np 4 abacus > log-always 2>&1
```

Expected: Test completes successfully, results identical to baseline

**Step 4: Compare results**

```bash
# Compare final energies
grep "final etot" log-off log-first log-always
```

Expected: All three runs produce identical final energies

**Step 5: Check for warning messages**

```bash
grep "WARNING.*condition number" log-first log-always
```

Expected: If matrix is well-conditioned, no warnings. If ill-conditioned, warning appears.

**Step 6: Restore INPUT file**

```bash
mv INPUT.backup INPUT
```

**Step 7: Document test results**

Create a summary of test results showing that all three modes produce identical results.

---

## Task 7: Final Build and Verification

**Step 1: Clean build**

```bash
rm -rf build
cmake -B build -DUSE_CUDA=ON
cmake --build build -j$(nproc)
cmake --install build
```

Expected: Clean build succeeds

**Step 2: Run full test suite**

```bash
cd tests/integrate
./Autotest.sh -r "102_PW_DS_davsubspace"
```

Expected: Test passes

**Step 3: Final commit**

```bash
git add -A
git commit -m "Feature(hsolver): complete matrix condition number pre-check implementation

Implement proactive condition number checking for GPU dngvd to prevent
numerical instability failures. Key features:

- New INPUT parameter diago_cond_check (first/always/off)
- Check scc matrix condition number using LAPACK xGECON
- Conservative threshold 1e12 for double precision
- Static caching minimizes overhead
- Preserves existing try-catch as safety net

Tested with 102_PW_DS_davsubspace, all modes produce identical results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

This implementation adds proactive condition number checking before GPU dngvd calls:

1. **INPUT parameter** `diago_cond_check` controls check frequency
2. **Condition check function** uses LAPACK to estimate matrix condition number
3. **Static caching** minimizes overhead for "first" mode
4. **Fallback to CPU** when condition number exceeds 1e12
5. **Integration tested** with existing test cases

The implementation is conservative, transparent (warning messages), and preserves existing error handling as a safety net.

