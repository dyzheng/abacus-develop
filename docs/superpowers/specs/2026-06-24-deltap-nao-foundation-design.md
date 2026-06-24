# DeltaP NAO Foundation: SMO-Projected Atomic Polarization

> **Spec date**: 2026-06-24  
> **Parent design**: `DeltaP_Incremental_Design.md` (Phase A logic, Phase B prerequisite)  
> **Scope**: Post-processing computation of atomic polarization P^I_alpha via SMO-projected Berry connection, NAO basis only. No SCF integration (H^lambda) yet.  
> **Test systems**: Si (T0, centrosymmetric P=0), BaTiO3 (T1, ferroelectric ~0.26 C/m^2)

---

## 1. Goal

Implement and verify the core physics of DeltaP: decomposing total crystal
polarization into atomic contributions P^I_alpha using Smooth Modulation Orbital
(SMO) projection, reusing DeltaSpin's SMO construction and two-center overlap
infrastructure.

**Success criteria**:
- T0: Analytic `|d_k alpha>` vs finite-difference `d_k<alpha|psi>` agree to <1e-8
- T1: `|sum_I P^I - P_total| / |P_total| < 0.01` (1% sum rule) against ABACUS `berry_phase` output, stable across SMO modulation radii (rm = 2.0, 3.0, 4.0 Bohr)

This is the prerequisite for Phase B (SCF-integrated DeltaP with H^lambda and
inner loop). Once P^I can be computed and verified, the inner loop can constrain it.

---

## 2. Physics Background

### 2.1 Total Polarization (Berry Phase)

The electronic polarization along direction alpha (lattice vector a_alpha) is:

```
P_alpha = -(e / 2*pi*a_alpha) * sum_n Im[ integral dk_alpha * <u_nk | d_k_alpha u_nk> ]
```

In the discretized k-string form used by ABACUS:

```
P_alpha = -(e / 2*pi*a_alpha) * sum_n Im[ sum_j <u_{n,k_j} | u_{n,k_{j+1}}> * dk ]
```

(Wilson loop product; ABACUS `berryphase::Berry_Phase` computes this.)

### 2.2 Atomic Decomposition via SMO Projection

Insert the SMO identity `sum_I Pi^I = sum_I sum_{lm} |alpha^I_{lmk}><alpha^I_{lmk}| approx= 1`
(where Pi^I is the SMO projection operator, distinct from P^I_alpha the polarization)
into the Berry connection:

```
<psi_nk | d_k_alpha psi_nk> approx= sum_I sum_{lm} [
    <psi_nk | d_k_alpha alpha^I_{lmk}> * <alpha^I_{lmk} | psi_nk>
  + <psi_nk | alpha^I_{lmk}> * d_k_alpha <alpha^I_{lmk} | psi_nk>
]
```

The atomic Berry connection is:

```
A^I_n(k, alpha) = sum_{lm} [
    <psi_nk | d_k_alpha alpha^I_{lmk}> * <alpha^I_{lmk} | psi_nk>
  + <psi_nk | alpha^I_{lmk}> * d_k_alpha <alpha^I_{lmk} | psi_nk>
]
```

And the atomic polarization:

```
P^I_alpha = -(e / 2*pi*a_alpha) * sum_n Im[ integral dk_alpha * A^I_n(k, alpha) ]
```

Discretized on the k-string:

```
P^I_alpha = -(e / 2*pi*a_alpha) * dk * sum_n sum_j Im[ A^I_n(k_j, alpha) ]
```

### 2.3 SMO k-Space Overlaps

The SMO Bloch sum (centered on atom I):

```
|alpha^I_{lmk}> = (1/sqrt(N)) sum_R e^{ikR} |alpha^I_{lm}(r - R - tau_I)>
```

The NAO-SMO overlap (real-space, two-center integral):

```
<phi^0_mu | alpha^I_{lm}(R)> = int dr phi^0_mu(r) * alpha^I_{lm}(r - R - tau_I)
```

This is computed by `TwoCenterIntegrator::snap()` — the same function DeltaSpin uses
in `dspin_lcao.cpp::cal_pre_HR()` (line 351).

The k-space overlap:

```
S_{mu,Ilm}(k) = sum_R e^{ikR} <phi^0_mu | alpha^I_{lm}(R)>
```

Its k-derivative (analytic):

```
d_{k_alpha} S_{mu,Ilm}(k) = sum_R (i*R_alpha) * e^{ikR} * <phi^0_mu | alpha^I_{lm}(R)>
```

This uses the **same** real-space overlaps with a different phase weight (`i*R_alpha*e^{ikR}`
instead of `e^{ikR}`). Zero new overlap code needed.

### 2.4 SMO-Wavefunction Overlaps

The SMO-wavefunction overlap:

```
<alpha^I_{lmk} | psi_nk> = sum_mu S*_{mu,Ilm}(k) * C_{nmu}(k)
```

Its k-derivative (finite difference for verification):

```
d_k <alpha^I_{lmk} | psi_nk> approx [D_I(k+dk) - D_I(k-dk)] / (2*dk)
```

where `D_I(lm, n, k) = <alpha^I_{lmk} | psi_nk>`.

And the analytic bra-ket:

```
<psi_nk | d_k_alpha alpha^I_{lmk}> = sum_mu [d_{k_alpha} S_{mu,Ilm}(k)]* * C_{nmu}(k)
```

---

## 3. Architecture

### 3.1 Module Location

```
source/source_lcao/module_deltap/       [NEW]
    deltap.h                             # Main class header
    deltap.cpp                            # Main class implementation
    deltap_orbital.cpp                    # SMO construction (reuses DeltaSpin's TwoCenterBundle)
    deltap_overlap.cpp                    # Real-space <phi|alpha> overlaps (reuses snap())
    deltap_berry.cpp                      # k-space Berry connection & polarization
    deltap_io.cpp                         # Input/output, results reporting
    CMakeLists.txt                        # Build configuration
    test/                                 # Unit tests (T0, T1)
        CMakeLists.txt
        deltap_overlap_test.cpp           # T0: overlap & grad validation
        deltap_sumrule_test.cpp           # T1: sum rule integration test
```

### 3.2 Integration Point

Post-processing entry, triggered after SCF convergence in `ctrl_scf_lcao.cpp`,
following the same pattern as `berryphase::Macroscopic_polarization`:

```
// In ctrl_scf_lcao.cpp, after the berry_phase block (step 12):
if (inp.calculation == "nscf" && inp.deltap_switch)
{
    deltap::DeltaP dp(&pv);
    dp.init(ucell, gd, kv, orb, ...);
    dp.compute_atomic_polarization(psi, ...);
}
```

### 3.3 Input Parameters

New parameters added to `read_input_item_other.cpp`, category "DeltaP":

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deltap_switch` | bool | false | Enable DeltaP atomic polarization decomposition |
| `deltap_rm` | real | 3.0 | SMO modulation radius (Bohr), reuses onsite_radius if 0 |
| `deltap_gdir` | int | 3 | Polarization direction (1=x, 2=y, 3=z) |
| `deltap_dk_fd` | real | 1e-6 | Finite-difference delta-k for T0 validation |
| `deltap_npk_string` | int | 0 | Override k-string density (0 = use KPT mesh) |

### 3.4 Reuse from DeltaSpin

| DeltaSpin component | Reuse method |
|---------------------|-------------|
| `TwoCenterBundle::build_orb_onsite(radius)` | Direct call — SMO construction from NAO zeta functions |
| `TwoCenterIntegrator::snap()` | Direct call — compute `<phi\|alpha>` two-center overlaps |
| `B_I_data` overlap storage pattern | Replicate structure — store individual `<phi\|alpha(R)>` per atom |
| `cal_PI_sub` phase-summing logic | Replicate algorithm — `S(k) = sum_R e^{ikR} <phi\|alpha(R)>` |
| `berryphase::set_kpoints` | Call or replicate — k-string setup along gdir |
| `Grid_Driver::Find_atom` | Direct call — neighbor list for overlap pairs |

**No modification to DeltaSpin source files.** DeltaP is a pure consumer of the
same infrastructure (TwoCenterIntegrator, Grid_Driver, HContainer patterns).

---

## 4. Data Structures

### 4.1 Real-Space Overlaps (Per Atom)

```cpp
// Stored per atom I (all atoms, not just "constrained" — DeltaP decomposes all atoms)
struct OverlapData {
    int iat_adj;                    // global index of adjacent atom
    ModuleBase::Vector3<int> R_index;  // cell offset
    // iw_global -> vector<double> of <phi_mu | alpha^I_lm(R)> for each lm
    std::unordered_map<int, std::vector<double>> nlm;
};
std::vector<std::vector<OverlapData>> overlap_R_;  // [iat][adj_index]
std::vector<int> nproj_per_atom_;                   // max_l_plus_1^2 per atom
```

### 4.2 k-Space Quantities (Per k-point on String)

```cpp
// For each k-point k_j on the string:
struct KSpaceData {
    ModuleBase::Vector3<double> kvec_d;     // direct coords
    // S_{mu,Ilm}(k): [iat][lm][mu_local] = sum_R e^{ikR} <phi|alpha(R)>
    std::vector<std::vector<std::vector<std::complex<double>>>> S_k;
    // dS_{mu,Ilm}(k,alpha): [iat][alpha][lm][mu_local] = sum_R i*R_alpha*e^{ikR} <phi|alpha(R)>
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> dS_k;
    // D_I(lm, n) = <alpha^I_lmk | psi_nk>: [iat][lm][nband]
    std::vector<std::vector<std::vector<std::complex<double>>>> D_I;
};
std::vector<KSpaceData> kstring_data_;  // size = nppstr
```

### 4.3 Results

```cpp
struct AtomicPolarization {
    std::vector<ModuleBase::Vector3<double>> P_I;     // [nat] polarization per atom
    std::vector<ModuleBase::Vector3<double>> gamma_I;  // [nat] Berry phase per atom
    ModuleBase::Vector3<double> P_total;               // sum of P_I
    ModuleBase::Vector3<double> P_abacus;              // from berry_phase (for T1)
};
```

---

## 5. Algorithm

### 5.1 Main Workflow (`compute_atomic_polarization`)

```
1. build_smo_orbitals()
   - Call TwoCenterBundle::build_orb_onsite(deltap_rm)
   - Tabulate two-center integrator (overlap_orb_onsite)

2. compute_real_overlaps()
   - For each atom I:
     - Find adjacent atoms via Grid_Driver::Find_atom
     - For each adjacent atom (T1, I1, R):
       - For each NAO basis function mu on atom (T1, I1):
         - intor->snap(T1, L1, N1, M1, T0, dtau*lat0, 0, nlm)
         - Store <phi_mu | alpha^I_lm(R)> in overlap_R_[I][adj]

3. setup_kstring()
   - Reuse berryphase::set_kpoints(kv, gdir) to get k_index strings
   - Or replicate the logic for the gdir direction

4. For each k-point k_j on the string:
   a. compute_S_k(j)
      - For each atom I, each (lm, mu):
        S_{mu,Ilm}(k_j) = sum_R e^{ik_j*R} * <phi_mu | alpha^I_lm(R)>
        dS_{mu,Ilm}(k_j,alpha) = sum_R i*R_alpha*e^{ik_j*R} * <phi_mu | alpha^I_lm(R)>

   b. compute_D_I(j)
      - D_I(lm, n, k_j) = sum_mu S*_{mu,Ilm}(k_j) * C_{n,mu}(k_j)
      (Same algorithm as dspin_lcao::cal_PI_sub, but storing D_I per k)

   c. compute_berry_connection(j)
      - For each atom I, band n:
        term1 = sum_{lm} <psi_nk | d_k_alpha alpha^I_lmk> * <alpha^I_lmk | psi_nk>
              = sum_{lm} [sum_mu dS*_{mu,Ilm}(k,alpha) * C_{n,mu}] * D_I(lm, n)
        term2 = sum_{lm} <psi_nk | alpha^I_lmk> * d_k_alpha <alpha^I_lmk | psi_nk>
              = sum_{lm} D_I*(lm, n) * d_k_alpha D_I(lm, n)
        A^I_n(k_j, alpha) = term1 + term2

      d_k D_I via finite difference: [D_I(k_{j+1}) - D_I(k_{j-1})] / (2*dk)
      (Both T0 and T1 use FD for term2. An analytic d_k D_I would require
      d_k C_{n,mu} via Sternheimer response — deferred to Phase B.)

5. integrate_polarization()
   - dk = 1/nppstr (reciprocal lattice vector fraction)
   - P^I_alpha = -(e / 2*pi*a_alpha) * dk * sum_n sum_j Im[A^I_n(k_j, alpha)]
   - P_total = sum_I P^I
   - Compare with P_abacus from berry_phase output

6. verify_sum_rule()
   - |P_total - P_abacus| / |P_abacus| < 0.01
   - Report per-atom P^I to output file
```

### 5.2 Finite-Difference d_k D_I (for T0 and T1)

```
d_k_alpha D_I(lm, n, k_j) approx [D_I(lm, n, k_{j+1}) - D_I(lm, n, k_{j-1})] / (2*dk)

where dk = |k_{j+1} - k_j| in the gdir direction
```

This requires computing D_I at all k-points on the string first (step 4b for all j),
then computing the finite difference in step 4c.

### 5.3 Analytic d_k S (always used)

```
d_{k_alpha} S_{mu,Ilm}(k) = sum_R (i * R_alpha) * e^{ikR} * <phi_mu | alpha^I_lm(R)>
```

This is the same sum as S but with the `i*R_alpha` factor. Since the real-space
overlaps `<phi|alpha(R)>` are already stored, this is a weighted re-sum.

---

## 6. Test Plan

### 6.1 T0: Single-k Berry Connection Element Validation

**System**: Si (diamond structure, centrosymmetric, P=0)
- 2-atom primitive cell, nspin=1
- Gamma-center 4x4x4 k-mesh (for string density)
- DZP orbital, SG15 pseudopotential

**What to validate**:
1. `S_{mu,Ilm}(k)` computed by phase-summing real-space overlaps matches
   direct evaluation at a single k-point (consistency check)
2. `d_{k_alpha} S_{mu,Ilm}(k)` analytic (i*R_alpha weight) matches finite
   difference of S at neighboring k-points (dk=1e-6): relative error < 1e-8
3. `A^I_n(k, alpha)` from analytic term1 + finite-diff term2 is gauge-invariant
   (i.e., same result under phase rotation of psi)

**Pass criteria**: relative error < 1e-8 for analytic vs finite-difference

**Implementation**: GoogleTest unit test `deltap_overlap_test.cpp`, using a
minimal Si cell setup with hardcoded expected values or self-consistent FD checks.

### 6.2 T1: Sum Rule Verification

**System**: BaTiO3 (tetragonal ferroelectric phase)
- 5-atom primitive cell, nspin=1
- Gamma-center 8x8x8 k-mesh (dense along gdir=3/z)
- DZP orbital, SG15 pseudopotential
- Displaced Ti (ferroelectric distortion ~0.2 Bohr along z)

**What to validate**:
1. Run ABACUS `berry_phase=1` to get `P_total` (Wilson loop)
2. Run DeltaP post-processing to get `P^I` per atom
3. Check `sum_I P^I_z approx= P_total_z`
4. Check stability across rm = 2.0, 3.0, 4.0 Bohr

**Pass criteria**:
- `|sum_I P^I - P_total| / |P_total| < 0.01` (1%)
- Per-atom P^I physically reasonable: Ti and O(apical) dominate, Ba small
- Variation across rm < 5% of P_total

**Implementation**: Integration test `deltap_sumrule_test.cpp` or shell-script
test in `tests/` directory, comparing DeltaP output against `result.ref`.

### 6.3 Additional Checks

- **Si P=0 check**: For centrosymmetric Si, `sum_I P^I = 0` and each `P^I` should
  be small (gauge-dependent but < 0.01 C/m^2)
- **BaTiO3 individual atoms**: Ti P^z > 0 (displaced upward), O1/O2 P^z < 0
  (displaced downward relative to center), Ba P^z small

---

## 7. File-by-File Implementation Plan

### 7.1 `deltap.h` — Main Class

```cpp
namespace deltap {

class DeltaP {
public:
    DeltaP(const Parallel_Orbitals* paraV_in);
    ~DeltaP();

    // Initialization
    void init(const UnitCell& ucell, const Grid_Driver& gd,
              const K_Vectors& kv, const TwoCenterIntegrator* intor,
              const std::vector<double>& orb_cutoff,
              double rm, int gdir, double dk_fd);

    // Main entry point
    void compute_atomic_polarization(
        const psi::Psi<std::complex<double>>* psi,
        const elecstate::ElecState* pelec);

    // Results access
    const AtomicPolarization& get_results() const { return results_; }

private:
    // Step 1: SMO construction (reuse TwoCenterBundle)
    void build_smo_orbitals(const UnitCell& ucell);

    // Step 2: Real-space <phi|alpha(R)> overlaps (reuse snap())
    void compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd);

    // Step 3: k-string setup (reuse berryphase pattern)
    void setup_kstring(const K_Vectors& kv);

    // Step 4a: k-space S and dS
    void compute_S_k(int ik);

    // Step 4b: SMO-wavefunction overlap D_I
    void compute_D_I(int ik, const std::complex<double>* psi_k, int nbands);

    // Step 4c: Berry connection
    void compute_berry_connection(int ik, int nbands, const double* wg);

    // Step 5: Integrate to polarization
    void integrate_polarization(const UnitCell& ucell, int nbands);

    // Step 6: Verify sum rule
    void verify_sum_rule();

    // ... data members (see Section 4)
};

} // namespace deltap
```

### 7.2 `deltap_orbital.cpp` — SMO Construction

Thin wrapper around `TwoCenterBundle::build_orb_onsite(rm)` and `tabulate()`.
No new physics — just the DeltaSpin SMO construction called from DeltaP context.

### 7.3 `deltap_overlap.cpp` — Real-Space Overlaps

Replicates `dspin_lcao.cpp::cal_pre_HR()` lines 252-371 (the overlap computation
part), but stores individual `<phi_mu | alpha^I_lm(R)>` overlaps instead of the
sandwich product. Uses the same:
- `Grid_Driver::Find_atom` for neighbor search
- `TwoCenterIntegrator::snap(T1, L1, N1, M1, T0, dtau*lat0, 0, nlm)` for overlap
- First-zeta-per-l filtering (only first zeta of each l channel)

### 7.4 `deltap_berry.cpp` — Berry Connection & Polarization

Core new physics:
- `compute_S_k`: Phase-sum `S(k) = sum_R e^{ikR} <phi|alpha(R)>` and
  `dS(k,alpha) = sum_R i*R_alpha*e^{ikR} <phi|alpha(R)>`
- `compute_D_I`: `D_I(lm,n) = sum_mu S*_{mu,Ilm}(k) * C_{n,mu}(k)` (same as
  `cal_PI_sub` but storing per-k-point)
- `compute_berry_connection`: `A^I_n = term1 + term2` (see Section 5.1)
- `integrate_polarization`: k-string sum to get P^I

### 7.5 `deltap_io.cpp` — Input/Output

- Read `deltap_*` parameters from `PARAM.inp`
- Write per-atom P^I to `OUT.{suffix}/deltap_results.dat`
- Write sum-rule comparison to log

### 7.6 `CMakeLists.txt`

```cmake
list(APPEND objects
    deltap.cpp
    deltap_orbital.cpp
    deltap_overlap.cpp
    deltap_berry.cpp
    deltap_io.cpp
)
add_library(deltap OBJECT ${objects})
if(BUILD_TESTING)
    if(ENABLE_MPI)
        add_subdirectory(test)
    endif()
endif()
```

Also add `module_deltap` to the parent `source_lcao/CMakeLists.txt` include path
and `Makefile.Objects`.

### 7.7 Input Parameter Changes

In `source/source_io/module_parameter/`:
- `input.h`: Add `bool deltap_switch`, `double deltap_rm`, `int deltap_gdir`,
  `double deltap_dk_fd`, `int deltap_npk_string` to `Input` struct
- `read_input_item_other.cpp`: Add parameter parsing (category "DeltaP")
- `input_conv.cpp`: Add defaults if needed

### 7.8 Entry Point Changes

In `source/source_io/module_ctrl/ctrl_scf_lcao.cpp`:
- Add `#include "source_lcao/module_deltap/deltap.h"`
- After step 12 (berry_phase), add step 12b: if `deltap_switch`, run DeltaP

---

## 8. Risk Points and Mitigations

### 8.1 SMO Completeness (HIGH)

**Risk**: The sum rule `sum_I P^I = P_total` only holds if the SMO set is complete.
If the SMO only captures part of the Hilbert space, the decomposition will be
incomplete.

**Mitigation**:
- Test with multiple rm values (2.0, 3.0, 4.0 Bohr)
- If sum rule fails at 1% for all rm, investigate whether more l-channels are needed
- Compare with Wannier-based decomposition as a cross-check (future work)

### 8.2 Finite-Difference Accuracy (MEDIUM)

**Risk**: The finite difference `d_k <alpha|psi>` may have numerical noise if dk
is too small (round-off) or too large (discretization error).

**Mitigation**:
- T0 explicitly validates the FD step size (dk=1e-6 vs 1e-4 vs 1e-8)
- Use the k-string spacing from the actual KPT mesh (not an arbitrary dk)
- For T1, the k-string spacing is `1/nppstr` which is typically ~0.05-0.1 (1/8 to 1/16)

### 8.3 Phase Convention / Gauge (MEDIUM)

**Risk**: Berry connection is gauge-dependent. The SMO overlaps introduce a
gauge choice. If the wavefunction phases are inconsistent across k-points
(different random phases after diagonalization), the finite difference will fail.

**Mitigation**:
- The finite difference `d_k D_I = [D_I(k+dk) - D_I(k-dk)] / (2*dk)` requires
  consistent wavefunction phases at neighboring k-points. If each k-point is
  independently diagonalized with a random phase, the FD will fail.
- ABACUS's `berryphase::stringPhase` solves this via parallel transport: it
  computes overlaps `<psi_{k_j}|psi_{k_{j+1}}>` and rotates phases to maximize
  real part. DeltaP must apply the same phase-fixing before computing D_I at
  each k-point on the string.
- The Berry connection integral (unlike the connection at a single k-point)
  is gauge-invariant, so the final P^I is robust once phases are consistent.
- T0 includes a gauge-invariance check: compute A^I_n at a single k with and
  without an artificial phase rotation; the integral over the full string
  should be invariant.

### 8.4 MPI Parallelization (LOW)

**Risk**: The k-string points are distributed across MPI ranks. Need to gather
D_I from all ranks for the finite difference.

**Mitigation**:
- Follow `cal_PI_sub` pattern: compute local D_I, then `MPI_Allreduce`
- The k-string is 1D, so parallelization is straightforward

---

## 9. Future Todos (Phase B and Beyond)

After this foundation deliverable is verified, the following phases build on it:

### Phase B: SCF-Integrated DeltaP (NAO)

| Step | Content | Dependencies |
|------|---------|-------------|
| B.1 | `HContainer_grad` prestorage: store `dS_{mu,Ilm}(k,alpha)` for all k | This spec |
| B.2 | H^lambda construction: `\|d_k alpha><alpha\|psi> + \|alpha>d_k<alpha\|psi>` operator | B.1 |
| B.3 | Inner loop adaptation: reuse DeltaSpin `lambda_loop.cpp` CG framework, change M->P | B.2 |
| B.4 | Force/stress correction: adapt `dspin_force_stress.hpp` Pulay terms | B.2 |
| B.5 | Finite-difference verification of force/stress/lambda (T4) | B.3+B.4 |
| B.6 | Full system convergence testing (T2, T3) | B.5 |

### Phase C: PW Basis DeltaP

| Step | Content | Dependencies |
|------|---------|-------------|
| C.1 | PW `d_k alpha` analytic: `i*(G+k)_alpha * alpha(G+k)` | Phase B verified |
| C.2 | PW H^lambda via `OnsiteProjector` becp infrastructure | C.1 |

### Phase D: Advanced Features

| Step | Content | Dependencies |
|------|---------|-------------|
| D.1 | Polarization quantum tracking (mod eR/Omega branch tracking) | Phase B |
| D.2 | E(P) curve benchmark vs Dieguez-Vanderbilt (2006) (T5) | Phase B |
| D.3 | Wannier interpolation for k-point reduction | Phase B |
| D.4 | Nested relaxation loop (atoms + DeltaP SCF) | Phase B |

---

## 10. References

- `DeltaP_Incremental_Design.md` — Parent design document
- DeltaSpin source: `source/source_lcao/module_deltaspin/`
- DeltaSpin LCAO operator: `source/source_lcao/module_operator_lcao/dspin_lcao.cpp`
  - `cal_pre_HR()` (line 238): SMO overlap computation pattern
  - `cal_PI_sub()` (line 568): k-space SMO-wavefunction overlap pattern
- Berry phase: `source/source_io/module_unk/berryphase.cpp`
  - `set_kpoints()` (line 57): k-string setup
  - `Berry_Phase()` (line 345): Wilson loop computation
- SMO construction: `source/source_basis/module_nao/two_center_bundle.cpp`
  - `build_orb_onsite()` (line 53): SMO from NAO zeta functions
  - `tabulate()`: Two-center integration tables
