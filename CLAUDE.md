# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABACUS (Atomic-orbital Based Ab-initio Computation at UStc) is an open-source electronic structure package based on density functional theory (DFT). It supports both plane wave (PW) and linear combination of atomic orbitals (LCAO) basis sets for materials simulations.

## Build System

ABACUS uses CMake as its primary build system (minimum version 3.16).

### Basic Build Commands

```bash
# Configure with default options (MPI + OpenMP + LCAO)
cmake -B build

# Build with parallel compilation
cmake --build build -j$(nproc)

# Install (required before running tests)
cmake --install build

# The executable name depends on configuration:
# - abacus: LCAO + MPI (default)
# - abacus_pw: PW only + MPI
# - abacus_serial: LCAO + no MPI
# - abacus_pw_serial: PW only + no MPI
```

### Key CMake Options

```bash
# Core features
-DENABLE_LCAO=ON          # Enable LCAO algorithm (default: ON)
-DENABLE_MPI=ON           # Enable MPI parallelization (default: ON)
-DUSE_OPENMP=ON           # Enable OpenMP (default: ON)
-DUSE_CUDA=OFF            # Enable CUDA for GPU acceleration
-DUSE_ROCM=OFF            # Enable ROCm for AMD GPUs

# Math libraries
-DMKLROOT=/path/to/mkl    # Use Intel MKL (recommended)
# Without MKL, requires: FFTW3, LAPACK, BLAS, ScaLAPACK

# LCAO-specific
-DUSE_ELPA=ON             # Enable ELPA diagonalization (default: ON with LCAO)
-DENABLE_LIBRI=OFF        # Enable LibRI for hybrid functionals
-DENABLE_PEXSI=OFF        # Enable PEXSI for large-scale LCAO

# Optional features
-DENABLE_LIBXC=OFF        # Enable LibXC for additional XC functionals
-DENABLE_MLALGO=OFF       # Enable ML algorithms (DeePKS, etc.)
-DDeePMD_DIR=/path        # Enable DeePMD-kit integration
-DENABLE_RAPIDJSON=OFF    # Enable JSON output

# Development
-DBUILD_TESTING=ON        # Build unit tests (default: OFF)
-DENABLE_COVERAGE=ON      # Enable code coverage (requires GCC)
-DENABLE_ASAN=ON          # Enable AddressSanitizer for debugging
-DCMAKE_BUILD_TYPE=Debug  # Debug build (default: Release-like with -O3 -g)
```

### Example Build Configurations

```bash
# Standard build with MKL
cmake -B build -DMKLROOT=$MKLROOT

# GPU build with CUDA (for modern GPUs with CUDA 11+)
cmake -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86"

# GPU build with CUDA 10.1 (older CUDA versions)
# CUDA 10.1 requires GCC 8 or earlier and compute capability 3.0-7.5
cmake -B build -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="30" \
  -DCUDAToolkit_INCLUDE_DIR=/usr/include \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-8

# Development build with tests
cmake -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cmake --install build
```

## Testing

### Unit Tests

Unit tests use GoogleTest framework and are located in `source/*/test` directories.

```bash
# Build with unit tests enabled
cmake -B build -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
cmake --install build  # Required before running tests

# Run all unit tests
cd build
ctest -V

# Run specific test pattern
ctest -R cell  # Run tests matching "cell"

# Run single test
ctest -R cell_unitcell_test_readpp

# Run tests directly
cd build/source/source_cell/test
./cell_unitcell_test
```

### Integration Tests

Integration tests are end-to-end tests located in `tests/` directory.

```bash
# Run integration tests from test directory
cd tests/integrate
./Autotest.sh

# Run specific test case
cd tests/integrate
./Autotest.sh -r "101_PW_.*"  # Run tests matching pattern

# Run single test manually
cd tests/01_PW/101_PW_OU_PL
OMP_NUM_THREADS=2 mpirun -np 4 abacus

# Generate reference results (for new tests)
cd tests/integrate
./Autotest.sh -g
```

Test categories:
- `01_PW`: Plane wave basis tests
- `02_NAO_Gamma`: LCAO with gamma-only k-points
- `03_NAO_multik`: LCAO with multiple k-points
- `04_FF`: Force fields (LJ, DeePMD, NEP)
- `05_rtTDDFT`: Real-time TDDFT
- `06_SDFT`: Stochastic DFT
- `07_OFDFT`: Orbital-free DFT
- `08_EXX`: Hybrid functionals and LR-TDDFT
- `09_DeePKS`: DeePKS tests
- `11_PW_GPU`, `12_NAO_Gamma_GPU`, etc.: GPU versions

## Code Architecture

### Module Structure

```
source/
├── source_base/          # Foundation: math libs, containers, MPI, utilities
│   ├── module_container/ # Data containers for CPU/GPU
│   ├── module_device/    # Device abstraction (CPU/CUDA/ROCm)
│   ├── module_mixing/    # Charge mixing algorithms
│   └── module_grid/      # Grid operations
├── source_basis/         # Basis set implementations
│   ├── module_nao/       # Numerical atomic orbitals (LCAO)
│   └── module_pw/        # Plane wave basis
├── source_cell/          # Unit cell, pseudopotentials, symmetry
│   ├── module_neighbor/  # Neighbor list construction
│   └── module_symmetry/  # Symmetry operations
├── source_estate/        # Electronic state management
│   ├── module_charge/    # Charge density and mixing
│   └── potentials/       # Hartree, XC, local pseudopotential
├── source_hamilt/        # General Hamiltonian (PW + LCAO)
│   ├── module_xc/        # Exchange-correlation functionals
│   ├── module_vdw/       # Van der Waals corrections
│   └── module_ewald/     # Ewald summation
├── source_pw/            # PW-specific Hamiltonian
│   ├── module_pwdft/     # PW-DFT operators
│   ├── module_ofdft/     # Orbital-free DFT
│   └── module_stodft/    # Stochastic DFT
├── source_lcao/          # LCAO-specific Hamiltonian
│   ├── module_gint/      # Grid integration for LCAO
│   ├── module_hcontainer/# Hamiltonian matrix storage
│   ├── module_dftu/      # DFT+U
│   ├── module_deepks/    # DeePKS integration
│   ├── module_ri/        # Resolution of identity (RI)
│   └── module_rt/        # Real-time TDDFT
├── source_hsolver/       # Hamiltonian solvers/diagonalization
│   ├── diago_david.cpp   # Davidson (PW)
│   ├── diago_cg.cpp      # Conjugate gradient (PW)
│   ├── diago_scalapack.cpp # ScaLAPACK (LCAO)
│   └── module_genelpa/   # ELPA wrapper (LCAO)
├── source_esolver/       # Energy solvers (workflow drivers)
│   ├── esolver_ks_pw.*   # Kohn-Sham PW solver
│   ├── esolver_ks_lcao.* # Kohn-Sham LCAO solver
│   ├── esolver_of.*      # Orbital-free DFT solver
│   ├── esolver_sdft_pw.* # Stochastic DFT solver
│   ├── esolver_dp.*      # DeePMD solver
│   └── esolver_lj.*      # Lennard-Jones solver
├── source_psi/           # Wavefunction representation
├── source_io/            # Input/output operations
│   └── module_parameter/ # Input parameter handling
├── source_md/            # Molecular dynamics
└── source_relax/         # Structural optimization
```

### Key Design Patterns

1. **ESolver Pattern**: Each calculation type (KS-PW, KS-LCAO, OFDFT, etc.) has an ESolver class that orchestrates the workflow (initialization → SCF iteration → output).

2. **Operator Pattern**: Hamiltonian terms are implemented as operators that can be composed. See `source_hamilt/operator.h` and implementations in `source_pw/module_pwdft/operator_pw/` and `source_lcao/module_operator_lcao/`.

3. **Device Abstraction**: Code can run on CPU, CUDA, or ROCm through the device abstraction layer in `source_base/module_device/` and container classes in `source_base/module_container/`.

4. **HSolver Abstraction**: Different diagonalization methods (Davidson, CG, ScaLAPACK, ELPA) implement a common HSolver interface.

## Code Style

### Formatting

- Use `clang-format` with the provided `.clang-format` configuration (Microsoft style base)
- Indentation: 4 spaces
- Pointer alignment: left (`int* ptr`)
- Pre-commit hooks run `clang-tidy` automatically

### Documentation

- Use Doxygen Javadoc style for public APIs in `.h` files:
  ```cpp
  /**
   * @brief Brief description
   *
   * Detailed description
   *
   * @param[in] input Input parameter description
   * @param[out] output Output parameter description
   * @return Return value description
   */
  ```

### Naming Conventions

- Classes: PascalCase (e.g., `ESolver_KS_PW`)
- Functions: snake_case (e.g., `cal_energy()`)
- Member variables: snake_case with trailing underscore for private (e.g., `energy_`)
- Namespaces: lowercase (e.g., `namespace hamilt`)

## Development Workflow

### Adding New Features

1. Read existing code in the relevant module before making changes
2. For significant features, consider using plan mode to design the approach
3. Add unit tests in `source/*/test/` directories
4. Add integration tests in `tests/` if needed
5. Update documentation if adding new input parameters

### Common Tasks

**Add a unit test:**
```cmake
# In source/module_name/test/CMakeLists.txt
AddTest(
  TARGET module_name_feature_test
  SOURCES feature_test.cpp
  LIBS math_libs  # if math libraries needed
)
```

**Add an integration test:**
1. Create test directory in appropriate `tests/XX_*/` folder
2. Prepare input files (keep runtime < 20 seconds)
3. Set `pseudo_dir` and `orbital_dir` relative to `tests/PP_ORB`
4. Run test and generate reference: `bash ../tools/catch_properties.sh result.ref`
5. Add test to `tests/integrate/CASES_CPU.txt` or `CASES_GPU.txt`

**Debug with GDB:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cmake --install build
cd /path/to/test/case
gdb abacus
# In GDB: run, where, print var_name
```

**Check code coverage:**
```bash
cmake -B build -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON
cmake --build build -j$(nproc)
cmake --install build
cmake --build build --target test ARGS="-V --timeout 21600"
cd build && make lcov
# View build/lcov/html/all_targets/index.html
```

**Debug GPU diagonalization issues (PW basis):**

When encountering cusolver failures or numerical issues in GPU diagonalization (e.g., `cusolver Zhegvd failed`), you can add diagnostic output to check the hcc (Hamiltonian) and scc (overlap) matrices in `source/module_hsolver/kernels/cuda/dngvd_op.cu`:

```cpp
// In dngvd_op<T, base_device::DEVICE_GPU>::operator()
// Add before xhegvd_wrapper call:

const int mat_size = nstart * ldh;
std::vector<T> h_A_debug(mat_size);
std::vector<T> h_B_debug(mat_size);
cudaErrcheck(cudaMemcpy(h_A_debug.data(), A, sizeof(T) * mat_size, cudaMemcpyDeviceToHost));
cudaErrcheck(cudaMemcpy(h_B_debug.data(), B, sizeof(T) * mat_size, cudaMemcpyDeviceToHost));

// Calculate matrix norms and check for NaN/Inf
Real norm_A = 0.0, norm_B = 0.0;
bool has_nan_A = false, has_nan_B = false;
for (int i = 0; i < mat_size; i++) {
    Real abs_A = std::abs(h_A_debug[i]);
    Real abs_B = std::abs(h_B_debug[i]);
    if (std::isnan(abs_A)) has_nan_A = true;
    if (std::isnan(abs_B)) has_nan_B = true;
    if (abs_A > norm_A) norm_A = abs_A;
    if (abs_B > norm_B) norm_B = abs_B;
}

std::cout << "DEBUG dngvd_op: n=" << nstart
          << ", ||hcc||_max=" << norm_A
          << ", ||scc||_max=" << norm_B;
if (has_nan_A) std::cout << " [hcc has NaN]";
if (has_nan_B) std::cout << " [scc has NaN]";
std::cout << std::endl;
```

This diagnostic output helps identify:
- Zero or uninitialized matrices (norm = 0)
- NaN/Inf values indicating numerical instability
- Matrix conditioning issues before they cause cusolver failures

Common root causes revealed by this diagnostic:
- Uninitialized wavefunctions (all zeros) → check wavefunction initialization
- PAGED_GPU mode data transfer issues → verify load/store sequence
- K-point continuity propagation failures → check propagate_psi implementation

## Python Interface

The `python/pyabacus/` directory contains Python bindings built with pybind11:

```bash
cd python/pyabacus
pip install -v .  # or pip install .[test] for test dependencies
pytest -v  # run Python tests
```

Modules: `io`, `Cell`, `ModuleBase`, `ModuleNAO`, `hsolver`

## Important Notes

- **C++ Standard**: Minimum C++11, but C++14 required for PEXSI/LibRI, C++17 for CUDA 13+
- **MPI**: Most features require MPI; serial builds have limited functionality
- **ELPA**: Automatically disabled for serial builds
- **Math Libraries**: MKL is strongly recommended for performance; otherwise need FFTW3 + LAPACK + BLAS + ScaLAPACK
- **Pseudopotentials**: Store in `tests/PP_ORB/` for tests; use relative paths in INPUT files
- **Git Hooks**: Pre-commit runs clang-tidy; may need to fix issues before committing

### CUDA Version Compatibility

ABACUS supports CUDA 10.1+ with the following considerations:

**CUDA 10.1 (older systems):**
- Requires GCC 8 or earlier (GCC 9+ not supported)
- Supports compute capability 3.0-7.5
- Use architecture `30` for compatibility
- Must explicitly set `CUDAToolkit_INCLUDE_DIR` if headers are in non-standard location
- Example build command:
  ```bash
  cmake -B build -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="30" \
    -DCUDAToolkit_INCLUDE_DIR=/usr/include \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-8
  ```

**CUDA 11.0+ (modern systems):**
- Supports GCC 9+
- Supports compute capability 3.5-9.0
- Use architecture matching your GPU (e.g., `80` for A100, `86` for RTX 3090)
- Example build command:
  ```bash
  cmake -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86"
  ```

**CUDA 13.0+:**
- Requires C++17 standard
- Dropped support for compute capability < 7.5
- Use architectures 75+ only

**Common CUDA Build Issues:**

1. **GCC version mismatch**: CUDA 10.1 requires GCC ≤ 8
   - Solution: Explicitly set `CMAKE_CXX_COMPILER` and `CMAKE_C_COMPILER` to GCC-8

2. **Missing CUDA headers**: CMake cannot find `cuda_runtime.h`
   - Solution: Set `-DCUDAToolkit_INCLUDE_DIR=/usr/include` (or appropriate path)

3. **Unsupported architecture**: Architecture not supported by CUDA version
   - Solution: Use architecture 30 for CUDA 10.1, or check CUDA documentation for supported architectures

4. **Code compatibility**: Some CUDA features require specific versions
   - `CUDA_R_64I` data type requires CUDA 11.0+ (handled automatically in code)
   - `__ldg()` intrinsic requires compute capability 3.5+ (handled automatically in code)

## Commit Message Format

Follow Conventional Commits:
```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

Types: `Feature`, `Fix`, `Docs`, `Style`, `Refactor`, `Perf`, `Test`, `Build`, `CI`, `Revert`

Example:
```
Fix(lcao): correct scalapack interface for complex matrices

Use complex* instead of double* for alpha/beta parameters in pzgemv_
and pzgemm_ calls to fix compilation with GNU compiler.

Fix #753
```

## Claude Code 开发行为规范

以下规范基于历史会话分析总结，用于约束 Claude Code 的行为模式，提高协作效率。

### 构建失败处理

- 同一构建错误最多重试 **2 次**，第 2 次仍失败时必须停下来，输出结构化的根因分析报告：(1) 已解决的错误，(2) 当前阻塞错误及分析，(3) 2-3 个按可能性排序的假设
- 每次修复只针对 **第一个** 错误，不要追级联错误
- 修复前必须先 **阅读相关源文件/头文件**，不要盲目猜测
- **绝对不要** 为了绕过构建错误而禁用用户需要的功能（如 CUDA、MPI、ELPA 等），除非用户明确要求

### 探索与执行的平衡

- 代码探索阶段限制在 **10 次工具调用** 以内，之后必须输出简短的行动计划并开始编辑代码
- 不要在一个会话中只做探索和写计划而不产出任何代码变更
- 优先使用 Grep/Glob 进行定向搜索，避免大范围逐文件阅读
- 如果用户指定了要修改的文件，直接在这些文件上工作，不要扩大探索范围

### 增量式开发

- 每完成一个逻辑变更后立即编译验证：`cmake --build build -j$(nproc)`
- 不要积攒多个变更后再一起编译，这会导致错误难以定位
- 编辑顺序：头文件声明 → 源文件实现 → CMakeLists.txt 更新 → 编译验证 → 测试

### TDD 工作流

- 实现新物理模块时遵循 TDD：先写测试，再写实现
- 测试文件放在对应模块的 `test/` 目录下，使用 GoogleTest 框架
- 确保测试能独立编译通过（可用 stub/mock），再实现功能代码
- 每轮迭代：编辑 → 编译 → 运行测试 → 修复，最多 **8 轮**迭代，超过则停下报告状态

### CUDA/GPU 相关

- 修改 CUDA 相关代码前，先确认目标 CUDA 版本和对应 API 可用性
- 使用 `#if CUDA_VERSION >= XXYY` 做版本守卫时，务必查阅官方文档确认正确的版本阈值
- 不要假设所有 CUDA API 在所有版本中都可用，特别注意 cusolver、cublas 的版本差异

### 沟通与中断

- 如果对用户意图不确定，**立即询问**，不要假设后执行
- 遇到无法解决的问题时，输出当前进展和阻塞点，而不是继续循环尝试
- 每次会话开始时，确认当前任务目标和约束条件

## Resources

- Documentation: https://abacus.deepmodeling.com/
- GitHub: https://github.com/deepmodeling/abacus-develop
- Issue Tracker: https://github.com/deepmodeling/abacus-develop/issues
