# DeltaQS: Charge and Spin Constrained DFT

## Overview

DeltaQS extends DeltaSpin to enable **joint charge and spin constraints** on individual atoms. It unifies three constraint modes within the same Lagrange multiplier framework:

| Mode | Charge (μ) | Spin (λ) | Description |
|------|------------|----------|-------------|
| **DeltaSpin** | ✗ | ✓ | Spin constraint only |
| **DeltaQ** | ✓ | ✗ | Charge constraint only |
| **DeltaQS** | ✓ | ✓ | Joint charge-spin constraint |

DeltaQS enables:
- Constraining projected charge and magnetic moment simultaneously
- Computing energy landscapes E(N, M) where N = charge, M = magnetic moment
- Identifying natural charge states via chemical potential μ → 0
- Gradient-based optimization with envelope theorem: ∂E/∂N = -μ, ∂E/∂M = -λ

## Theoretical Foundation

The constrained DFT energy functional:

$$E_{\text{total}} = E_{KS} + \sum_I \mu_I (N_I - N_I^{\text{target}}) + \sum_I \lambda_I (M_I - M_I^{\text{target}})$$

The effective potential becomes:

$$v_{\text{eff}}^\alpha = v_{KS}^\alpha + \sum_I (\mu_I + \lambda_I) w_I$$
$$v_{\text{eff}}^\beta = v_{KS}^\beta + \sum_I (\mu_I - \lambda_I) w_I$$

where $w_I$ is the CSZ projection weight for atom $I$.

Reference: Cai Z, et al., *Quantum Frontiers* 2.1 (2023): 21.

## Enabling DeltaQS

### INPUT File Parameters

```bash
# Core switches
sc_mag_switch      1        # Enable spin constraint (required)
sc_charge_switch   1        # Enable charge constraint
sc_qs_mode         deltaqs  # Mode: deltaspin / deltaq / deltaqs / auto

# Convergence parameters
nsc                50       # Max outer iterations
sc_thr             1e-4     # Spin convergence threshold (μB)
sc_charge_thr      1e-3     # Charge convergence threshold (e)
sc_scf_thr         1e-3     # SCF threshold before entering constraint loop

# Optimization parameters
sc_charge_alpha    0.1      # Charge step size (Ry/e²)
alpha_trial        0.01     # Initial trial step (eV/μB²)
sccut              3.0      # Max step size (eV/μB)

# Advanced features
sc_gradient_output 1        # Write gradient files for analysis
sc_ground_state    0        # Enable ground state search (0/1)
sc_outer_max_iter  50       # Max outer loop iterations
sc_outer_thr       0.01     # Outer loop convergence (Ry)
```

### STRU File Format

Specify target charge and magnetic moment per atom:

```
ATOMIC_POSITIONS
Direct

Fe
0.0
2
0.00  0.00  0.00  mag  2.0   sc 1 0 1  tc 13.5  cq 1
0.51  0.51  0.51  mag  -2.0  sc 1 0 1  tc 13.5  cq 1

O
0.0
3
0.25  0.25  0.25  mag  0.0
0.75  0.75  0.25  mag  0.0
0.75  0.25  0.75  mag  0.0
```

**Keywords:**
- `mag 2.0`: target magnetic moment (μB)
- `sc 1 0 1`: spin constraint flags (x, y, z components; 1=constrained, 0=free)
- `tc 13.5`: target charge (electrons)
- `cq 1`: charge constraint flag (1=constrained, 0=free)

## Usage Examples

### Example 1: DeltaSpin (Spin Constraint Only)

Constrain magnetic moments without charge constraint:

**INPUT:**
```bash
sc_mag_switch      1
sc_charge_switch   0
sc_qs_mode         deltaspin
nspin              2
```

**STRU:**
```
Fe
0.0
2
0.00  0.00  0.00  mag  2.0   sc 1
0.51  0.51  0.51  mag  -2.0  sc 1
```

### Example 2: DeltaQ (Charge Constraint Only)

Constrain projected charge without spin constraint:

**INPUT:**
```bash
sc_mag_switch      1
sc_charge_switch   1
sc_qs_mode         deltaq
nspin              2
sc_charge_thr      1e-3
```

**STRU:**
```
Fe
0.0
2
0.00  0.00  0.00  mag  0.0  sc 0  tc 13.5  cq 1
0.51  0.51  0.51  mag  0.0  sc 0  tc 13.5  cq 1
```

### Example 3: DeltaQS (Joint Constraint)

Constrain both charge and magnetic moment:

**INPUT:**
```bash
sc_mag_switch      1
sc_charge_switch   1
sc_qs_mode         deltaqs
nspin              2
sc_thr             1e-4
sc_charge_thr      1e-3
sc_gradient_output 1
```

**STRU:**
```
Fe
0.0
2
0.00  0.00  0.00  mag  2.0   sc 1 0 1  tc 13.5  cq 1
0.51  0.51  0.51  mag  -2.0  sc 1 0 1  tc 13.5  cq 1
```

### Example 4: Charge Scan

Compute E(N) curve by scanning target charge:

```bash
#!/bin/bash
for N in 12.0 12.5 13.0 13.5 14.0 14.5 15.0; do
    cat > STRU <<EOF
ATOMIC_POSITIONS
Direct
Fe
0.0
1
0.00  0.00  0.00  mag  0.0  sc 0  tc $N  cq 1
EOF
    
    cat > INPUT <<EOF
INPUT_PARAMETERS
suffix    fe_scan
sc_mag_switch      1
sc_charge_switch   1
sc_qs_mode         deltaqs
sc_charge_thr      1e-3
sc_gradient_output 1
EOF
    
    abacus > output_${N}.log 2>&1
    
    # Extract results
    E=$(grep "E_KohnSham" OUT.fe_scan/running_scf.log | tail -1 | awk '{print $2}')
    MU=$(grep "Fe_0" deltaqs_gradient_*.dat | tail -1 | awk '{print $6}')
    echo "$N  $E  $MU" >> charge_scan.dat
done
```

## Output Files

### Standard Output

```
===============================================================================
[DeltaQS] Joint charge-spin constraint enabled
[DeltaQS] Constrained atoms: 2/5
[DeltaQS] Charge RMS: 0.523 e (threshold: 0.001)
[DeltaQS] mu step 0: charge RMS = 0.412
[DeltaQS] mu step 1: charge RMS = 0.287
...
[DeltaQS] mu step 15: charge RMS = 0.0008
[DeltaQS] Charge constraint converged in 15 steps
===============================================================================
```

### Gradient Files

When `sc_gradient_output = 1`, DeltaQS writes `deltaqs_gradient_N.dat`:

```
# DeltaQS Gradient Output (step 15)
# Atom  Ni  Mi_z  target_N  target_M  mu(Ry)  lambda_z(Ry)  mu(eV)  lambda_z(eV)
Fe_0  1.35023e+01  2.00015e+00  1.350000e+01  2.000000e+00  3.21e-04  1.52e-03  4.37e-03  2.07e-02
Fe_1  1.34977e+01  -2.00012e+00  1.350000e+01  -2.000000e+00  -3.18e-04  -1.49e-03  -4.33e-03  -2.03e-02
O_2   0.00000e+00  0.00000e+00  0.000000e+00  0.000000e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
```

**Columns:**
- `Ni`: actual projected charge (electrons)
- `Mi_z`: actual magnetic moment (μB)
- `target_N`, `target_M`: constraint targets
- `mu`: charge chemical potential (∂E/∂N)
- `lambda_z`: spin chemical potential (∂E/∂M)

### Interpreting μ and λ

- **μ ≈ 0**: target charge matches natural charge state (equilibrium)
- **μ < 0**: system wants more charge (energy decreases if N increases)
- **μ > 0**: system wants less charge (energy decreases if N decreases)
- **λ ≈ 0**: target magnetic moment matches natural moment
- **|μ|, |λ|**: magnitude of constraint force (Ry/e or Ry/μB)

## Advanced Features

### Ground State Search

Enable automatic ground state search in E(N,M) space:

```bash
sc_ground_state    1
sc_outer_max_iter  50
sc_outer_thr       0.01
```

DeltaQS will iteratively update targets to minimize E until |∇E| < sc_outer_thr.

### Multi-Start Optimization

For complex landscapes with multiple local minima, use the programmatic API:

```cpp
SpinConstrain& sc = SpinConstrain::get_instance();
sc.run_qs_multistart(
    10,                              // n_starts
    {12.0, 15.0},                    // N_range
    {-5.0, 5.0},                     // M_range
    "lbfgs",                         // optimizer
    50,                              // max_steps
    0.01                             // conv_thr
);
```

### Dataset Generation

Generate training data for machine learning:

```cpp
sc.run_qs_dataset_generation(
    "deltaqs_dataset.dat",           // output_file
    100,                             // n_samples
    {12.0, 15.0},                    // N_range
    {-5.0, 5.0},                     // M_range
    "uniform"                        // sampling_method
);
```

## Troubleshooting

### Charge Constraint Not Converging

**Symptom:** `[DeltaQS] Charge RMS` stays above threshold after 50 steps.

**Solutions:**
1. Increase `sc_charge_alpha` (e.g., 0.1 → 0.3)
2. Increase `nsc` (e.g., 50 → 100)
3. Relax `sc_charge_thr` (e.g., 1e-3 → 1e-2)
4. Check if target charge is physically reasonable (run DeltaSpin first to find natural charge)

### Large μ Values

**Symptom:** |μ| > 1 Ry/e, indicating target is far from natural charge.

**Solutions:**
1. Run unconstrained calculation first to estimate natural charge
2. Use charge scan to map E(N) and find minimum
3. Adjust target charge to be closer to natural value

### Energy Higher Than Unconstrained

**Expected:** Constrained energy is always ≥ unconstrained energy.

**Interpretation:** Energy difference ΔE = E_constrained - E_unconstrained represents the thermodynamic cost of the constraint. Small ΔE (< 0.1 eV) indicates target is close to natural state.

## Best Practices

1. **Start with DeltaSpin**: Understand magnetic structure before adding charge constraints
2. **Scan first**: Map E(N) or E(M) to identify reasonable target ranges
3. **Check gradients**: Use `sc_gradient_output 1` to monitor convergence
4. **Validate physics**: Ensure constrained states make chemical sense
5. **Use natural charges**: Target charges near μ ≈ 0 for stable convergence

## See Also

- [DeltaSpin documentation](spin.md#deltaspin-spin-constrained-dft)
- [STRU file format](../input_files/stru.md)
- [SCF convergence](converge.md)
- Implementation details: `docs/deltaqs_implementation.md`
