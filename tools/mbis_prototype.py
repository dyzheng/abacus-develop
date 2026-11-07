#!/usr/bin/env python3
"""MBIS (Minimal Basis Iterative Stockholder) prototype for ABACUS cube densities.

Read-only analysis: partitions an ABACUS charge-density cube with an iterative
Slater-s pro-density (population N_A and exponent zeta_A updated to match the
atomic density's population and mean radius), and prints MBIS charges, local
atomic dipoles, the molecular-dipole reconstruction, and (for nspin=2) a spatial
atomic spin moment.

Usage:
    python3 tools/mbis_prototype.py chg.cube
    python3 tools/mbis_prototype.py chgs1.cube chgs2.cube      # nspin=2
    python3 tools/mbis_prototype.py cube --images 1            # periodic images
Options: --maxit N, --damp x, --images {0,1}.

Notes:
  * Cube units are Bohr (ABACUS `out_chg`); density is in Bohr^-3.
  * Voxel vectors may be non-diagonal (e.g. bcc cells); the full coordinate
    grid is built from them.
  * Atom line format is `Z_atomic Z_valence/charge x y z`; the second field is
    used as the valence (pseudo) charge for net charges.
  * --images 1 sums each pro-density over the 26 nearest periodic images
    (needed for small periodic cells); --images 0 uses direct positions only.
  * For a neutral cell the molecular dipole is well defined; charged cells need
    a Berry-phase/sawtooth approach instead.
"""
import argparse
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
    dv = np.array(dv)  # rows are voxel vectors
    atoms = []
    for i in range(natom):
        p = lines[6 + i].split()
        atoms.append((float(p[1]), np.array([float(x) for x in p[2:5]])))
    data = np.array(" ".join(lines[6 + natom:]).split(), dtype=float)
    rho = data.reshape(n[0], n[1], n[2])
    return origin, dv, n, atoms, rho


def coords(origin, dv, n):
    ix = np.arange(n[0])
    iy = np.arange(n[1])
    iz = np.arange(n[2])
    X = (origin[0] + ix[:, None, None] * dv[0, 0]
         + iy[None, :, None] * dv[1, 0] + iz[None, None, :] * dv[2, 0])
    Y = (origin[1] + ix[:, None, None] * dv[0, 1]
         + iy[None, :, None] * dv[1, 1] + iz[None, None, :] * dv[2, 1])
    Z = (origin[2] + ix[:, None, None] * dv[0, 2]
         + iy[None, :, None] * dv[1, 2] + iz[None, None, :] * dv[2, 2])
    return X, Y, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubes", nargs="+", help="chg.cube or chgs1.cube chgs2.cube")
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--images", type=int, default=0,
                    help="0=direct atoms only, N=sum images with |L|<=N")
    ap.add_argument("--symmetrize", action="store_true",
                    help="average N,zeta over atoms of the same species (equivalent sites)")
    a = ap.parse_args()

    origin, dv, n, atoms, rho1 = read_cube(a.cubes[0])
    if len(a.cubes) > 1:
        _, _, _, _, rho2 = read_cube(a.cubes[1])
        rho = rho1 + rho2
        spin = rho1 - rho2
    else:
        rho = rho1
        spin = np.zeros_like(rho)
    dV = abs(np.linalg.det(dv))
    X, Y, Z = coords(origin, dv, n)
    Ne = rho.sum() * dV
    print(f"grid={n.tolist()} dV={dV:.6e} Bohr^3  N_elec(cube)={Ne:.6f}")
    R = [at[1] for at in atoms]
    Zv = [at[0] for at in atoms]

    # periodic image translations (cell vectors = n_i * voxel_i)
    cell = n[:, None] * dv
    images = [np.zeros(3)]
    if a.images >= 1:
        rng = range(-a.images, a.images + 1)
        images = [Lx * cell[0] + Ly * cell[1] + Lz * cell[2]
                  for Lx in rng for Ly in rng for Lz in rng]

    def dist(Ra):
        # direct distance to the atom (used for moments)
        return np.sqrt((X - Ra[0]) ** 2 + (Y - Ra[1]) ** 2 + (Z - Ra[2]) ** 2)

    def pro_density(Nv, zv, Ra):
        # sum the Slater-s pro-density over periodic images
        acc = np.zeros_like(X)
        for T in images:
            r = np.sqrt((X - Ra[0] - T[0]) ** 2 + (Y - Ra[1] - T[1]) ** 2
                        + (Z - Ra[2] - T[2]) ** 2)
            acc += np.exp(-zv * r)
        return Nv * zv ** 3 / (8 * np.pi) * acc

    r = [dist(Ra) for Ra in R]
    N = [z for z in Zv]
    zeta = [1.5] * len(R)
    for it in range(a.maxit):
        pro = [pro_density(N[i], zeta[i], R[i]) for i in range(len(R))]
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
        if a.symmetrize:
            groups = {}
            for ii, zz in enumerate(Zv):
                groups.setdefault(zz, []).append(ii)
            for zz, idx in groups.items():
                nm = sum(Nn[i] for i in idx) / len(idx)
                zm = sum(zn[i] for i in idx) / len(idx)
                for i in idx:
                    Nn[i], zn[i] = nm, zm
        N = [(1 - a.damp) * N[i] + a.damp * Nn[i] for i in range(len(N))]
        zeta = [(1 - a.damp) * zeta[i] + a.damp * zn[i]
                for i in range(len(zeta))]
        if it % 20 == 0 or res < 1e-9:
            print(f"  it{it:3d} N={['%.6f' % x for x in N]} "
                  f"zeta={['%.4f' % x for x in zeta]} dN={res:.2e}")
        if res < 1e-9:
            break

    pro = [pro_density(N[i], zeta[i], R[i]) for i in range(len(R))]
    S = sum(pro)
    mumol = np.zeros(3)
    sum_m = 0.0
    print("\n=== MBIS ===")
    for i in range(len(R)):
        w = np.where(S > 1e-30, pro[i] / S, 0.0)
        rhoA = w * rho
        Ni = rhoA.sum() * dV
        mA = (w * spin).sum() * dV
        sum_m += mA
        mu = -dV * np.array([((X - R[i][0]) * rhoA).sum(),
                             ((Y - R[i][1]) * rhoA).sum(),
                             ((Z - R[i][2]) * rhoA).sum()])
        Q = Zv[i] - Ni
        mumol += Q * np.asarray(R[i]) + mu
        print(f"atom {i}: N={Ni:.5f}  netQ={Q:+.5f}  zeta={zeta[i]:.5f}  "
              f"m_spin={mA:+.5f}  mu_local={mu} au  "
              f"|mu|={np.linalg.norm(mu) * 2.541746:.4f} D")
    print(f"sum N = {sum(N):.6f}")
    if len(a.cubes) > 1:
        print(f"sum m_spin = {sum_m:.6f}")
    print(f"mu_mol = {mumol} au  |mu_mol| = {np.linalg.norm(mumol) * 2.541746:.4f} D")


if __name__ == "__main__":
    main()
