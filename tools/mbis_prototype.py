#!/usr/bin/env python3
"""MBIS (Minimal Basis Iterative Stockholder) prototype for ABACUS cube densities.

Read-only analysis: partitions an ABACUS charge-density cube with an iterative
Slater-s pro-density (population N_A and exponent zeta_A updated to match the
atomic density's population and mean radius), and prints MBIS charges, local
atomic dipoles, and the molecular-dipole reconstruction.

This is a validation prototype for aligning the ABACUS allocation/Hirshfeld
backend with ORCA's MBIS (see docs/superpowers/specs/2026-10-30-mbis-alignment-plan.md).
It is NOT the production C++ implementation.

Usage:
    python3 tools/mbis_prototype.py OUT.*/chg.cube [max_iter] [damping]

Notes:
  * Cube units are Bohr (ABACUS `out_chg`); density is in Bohr^-3.
  * Atom line format is `Z_atomic Z_valence/charge x y z`; we use the second
    field as the valence (pseudo) charge for net charges.
  * For a neutral periodic cell the molecular dipole is well defined; for
    charged cells use a Berry-phase/sawtooth approach instead.
"""
import sys
import numpy as np


def read_cube(path):
    lines = open(path).read().splitlines()
    natom = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]])
    n, dv = [], []
    for i in range(3):
        p = lines[3 + i].split()
        n.append(int(p[0]))
        dv.append([float(x) for x in p[1:4]])
    n = np.array(n)
    dv = np.array(dv)
    atoms = []
    for i in range(natom):
        p = lines[6 + i].split()
        z_eff = float(p[1])  # valence/pseudo charge
        atoms.append((z_eff, np.array([float(x) for x in p[2:5]])))
    data = np.array(" ".join(lines[6 + natom:]).split(), dtype=float)
    rho = data.reshape(n[0], n[1], n[2])
    return origin, dv, n, atoms, rho


def main():
    cube = sys.argv[1] if len(sys.argv) > 1 else "OUT.*/chg.cube"
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    damp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    origin, dv, n, atoms, rho = read_cube(cube)
    dV = abs(np.linalg.det(dv))
    X = origin[0] + np.arange(n[0]) * dv[0, 0]
    Y = origin[1] + np.arange(n[1]) * dv[1, 1]
    Z = origin[2] + np.arange(n[2]) * dv[2, 2]
    Ne = rho.sum() * dV
    print(f"grid={n.tolist()} spacing={np.diag(dv).tolist()} Bohr  N_elec(cube)={Ne:.6f}")
    R = [a[1] for a in atoms]
    Zv = [a[0] for a in atoms]

    def dist(Ra):
        return np.sqrt((X - Ra[0])[:, None, None] ** 2
                       + (Y - Ra[1])[None, :, None] ** 2
                       + (Z - Ra[2])[None, None, :] ** 2)

    r = [dist(Ra) for Ra in R]
    N = [z for z in Zv]
    zeta = [1.5] * len(R)
    for it in range(maxit):
        pro = [N[i] * zeta[i] ** 3 / (8 * np.pi) * np.exp(-zeta[i] * r[i])
               for i in range(len(R))]
        S = sum(pro)
        Nn, zn, res = [], [], 0.0
        for i in range(len(R)):
            w = np.where(S > 1e-30, pro[i] / S, 0.0)
            rhoA = w * rho
            Ni = rhoA.sum() * dV
            mr = (r[i] * rhoA).sum() * dV
            Nn.append(Ni)
            zn.append(3.0 * Ni / mr if (Ni > 1e-12 and mr > 1e-12) else zeta[i])
            res = max(res, abs(Ni - N[i]))
        N = [(1 - damp) * N[i] + damp * Nn[i] for i in range(len(N))]
        zeta = [(1 - damp) * zeta[i] + damp * zn[i] for i in range(len(zeta))]
        if it % 20 == 0 or res < 1e-9:
            print(f"  it{it:3d} N={['%.6f' % x for x in N]} "
                  f"zeta={['%.4f' % x for x in zeta]} dN={res:.2e}")
        if res < 1e-9:
            break

    pro = [N[i] * zeta[i] ** 3 / (8 * np.pi) * np.exp(-zeta[i] * r[i])
           for i in range(len(R))]
    S = sum(pro)
    mumol = np.zeros(3)
    print("\n=== MBIS ===")
    for i in range(len(R)):
        w = np.where(S > 1e-30, pro[i] / S, 0.0)
        rhoA = w * rho
        Ni = rhoA.sum() * dV
        Rx = (X - R[i][0])[:, None, None]
        Ry = (Y - R[i][1])[None, :, None]
        Rz = (Z - R[i][2])[None, None, :]
        mu = -dV * np.array([(Rx * rhoA).sum(), (Ry * rhoA).sum(),
                             (Rz * rhoA).sum()])
        Q = Zv[i] - Ni
        mumol += Q * np.asarray(R[i]) + mu
        print(f"atom {i}: N={Ni:.5f}  netQ={Q:+.5f}  zeta={zeta[i]:.5f}  "
              f"mu_local={mu} au  |mu|={np.linalg.norm(mu) * 2.541746:.4f} D")
    print(f"sum N = {sum(N):.6f}")
    print(f"mu_mol = {mumol} au  |mu_mol| = {np.linalg.norm(mumol) * 2.541746:.4f} D")


if __name__ == "__main__":
    main()
