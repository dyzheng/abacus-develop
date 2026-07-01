# T5 Thermodynamic Scan Results

## Data

| δN   | E(Ry)           | μ₀(eV)  | μ₁(eV)  | Ni₀     | Ni₁     |
|------|-----------------|---------|---------|---------|---------|
| -0.2 | -6773.04897     | 2.463   | -2.046  | 13.5551 | 13.9564 |
| -0.1 | -6773.39654     | 1.345   | -0.816  | 13.6591 | 13.8519 |
| 0.0  | -6773.50435     | 0.157   | 0.203   | 13.7655 | 13.7614 |
| +0.1 | -6773.39656     | -0.816  | 1.345   | 13.8518 | 13.6591 |
| +0.2 | -6773.05031     | -2.043  | 2.459   | 13.9561 | 13.5555 |

## Observations

1. **Energy symmetry**: E(δN) ≈ E(-δN) as expected for symmetric BCC Fe₂
2. **Energy minimum**: E(0) < E(±0.1) < E(±0.2), natural charge is energy minimum
3. **μ antisymmetry**: μ₀(δN) ≈ -μ₁(-δN), consistent with symmetric system
4. **μ linearity**: μ₀(δN) is approximately linear in δN, suggesting quadratic E(δN)

## Thermodynamic Consistency Check

Expected: dE/d(δN) = μ₁ - μ₀ (from E' = E + μ₀(N₀-N₀*) + μ₁(N₁-N₁*))

| δN   | dE/d(δN) num (eV/e) | μ₁ - μ₀ (eV/e) | Discrepancy |
|------|---------------------|-----------------|-------------|
| 0.0  | ~0                  | 0.046           | -           |
| +0.1 | 30.89               | 2.161           | 14x         |
| -0.1 | -30.89              | -2.161          | 14x         |

**ISSUE**: Numerical derivative is ~14x larger than μ₁ - μ₀. Possible causes:
1. Unit conversion: μ might be in Ry internally, print might be wrong
2. Energy includes SCF error that dominates small ΔE
3. Sign convention mismatch in energy functional definition
4. The constrained energy E_scon = -Σ(λ·M + μ·N) not included in total E

## Quadratic Fit

E(δN) = E₀ + a·δN² (by symmetry, linear term = 0)

Using E(0) and E(+0.2):
a = (E(+0.2) - E(0)) / 0.04 = 0.45404 / 0.04 = 11.351 Ry/e²

dE/d(δN) = 2a·δN = 22.702·δN Ry/e

At δN=+0.1: dE/d(δN) = 2.270 Ry/e = 30.89 eV/e

If μ = -dE/dN, then μ₀ = -30.89/2 = -15.4 eV/e (half because perturbation splits between two atoms)

But measured μ₀ = -0.816 eV/e. Factor of ~19 discrepancy.

## TODO
- Verify μ units in code (Ry vs eV)
- Check if E_total includes E_scon contribution
- Compare with PW DeltaSpin results for same system
