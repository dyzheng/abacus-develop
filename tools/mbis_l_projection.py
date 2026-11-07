#!/usr/bin/env python3
"""Approximate in-basin l-projection from an ABACUS gamma-only DMR + NAO radial functions.

Computes, for each atom A and angular momentum l,
    n_{A,l} = sum_{n,n'} I^R_{nl,n'l} * Tr_m[ D^{(A,l)}_{n,n'} ]
where I^R = int_0^R chi_{nl}(r) chi_{n'l}(r) r^2 dr and Tr_m sums the
same-m diagonal of the (l,n),(l,n') DMR block.

This is a *spherical-basin* ("Löwdin-like within the atomic basin") l-population:
it uses only same-atom, same-l DMR blocks and is invariant under any rotation of
the real-spherical-harmonic basis (it only needs the (l,zeta) block structure).
It is an approximation to the true MBIS weight-basin l-projection, which needs
the full weighted overlap  int w_A(r) phi_mu phi_nu  in ABACUS's exact real-SH
basis (C++ / weighted-overlap route).

Usage:
    python3 tools/mbis_l_projection.py dms1g1_nao.txt dms2g1_nao.txt Fe.orb R_Bohr
Assumptions: 1 atom type, nat atoms, orbital ordering s(zeta) p(zeta) d(zeta) f
(as in ABACUS / mulliken.txt), gamma-only dense DMR.
"""
import sys
import numpy as np


def parse_orb(path):
    toks = open(path).read().split()
    tpos = [i for i, t in enumerate(toks) if t == "Type"]
    chi = {}
    mesh = None
    for a, tp in enumerate(tpos):
        t, l, n = int(toks[tp + 3]), int(toks[tp + 4]), int(toks[tp + 5])
        end = tpos[a + 1] if a + 1 < len(tpos) else len(toks)
        vals = np.array([float(x) for x in toks[tp + 6:end]])
        if mesh is None:
            mesh = len(vals)
        if t == 0:
            chi[(l, n)] = vals
    return chi, mesh


def parse_dmr(path, norb):
    lines = open(path).read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if "size of this matrix" in l)
    toks = []
    for l in lines[i0 + 1:]:
        for x in l.split():
            try:
                toks.append(float(x))
            except ValueError:
                pass
    data = np.array(toks[-norb * norb:])
    return data.reshape(norb, norb)


def l_blocks(nat, norb_atom):
    # per atom, (l, zeta) -> list of AO indices (m contiguous)
    # counts per l for Fe 4s2p2d1f: s:4 zeta x1, p:2x3, d:2x5, f:1x7
    layout = {0: [1, 1, 1, 1], 1: [3, 3], 2: [5, 5], 3: [7]}
    blocks = {}
    for a in range(nat):
        base = a * norb_atom
        off = 0
        for l in (0, 1, 2, 3):
            for z, nm in enumerate(layout[l]):
                blocks[(a, l, z)] = [base + off + m for m in range(nm)]
                off += nm
    return blocks


def main():
    up = parse_dmr(sys.argv[1], 54)
    dn = parse_dmr(sys.argv[2], 54)
    chi, mesh = parse_orb(sys.argv[3])
    R = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    dr = 0.01
    r = np.arange(mesh) * dr
    # radial integral matrix I^R[(l,n),(l,n')]
    rad = {}
    for l in (0, 1, 2, 3):
        ns = sorted(n for (ll, n) in chi if ll == l)
        for n in ns:
            for n2 in ns:
                integ = np.sum(chi[(l, n)] * chi[(l, n2)] * r * r) * dr
                mask = r <= R
                integR = np.sum((chi[(l, n)] * chi[(l, n2)] * r * r)[mask]) * dr
                rad[(l, n, n2)] = integR
        # report normalization of the first function
        if ns:
            nn = ns[0]
            norm = np.sum(chi[(l, nn)] ** 2 * r * r) * dr
            print(f"  l={l}: {len(ns)} zeta, int chi^2 r^2 dr = {norm:.4f}")
    blocks = l_blocks(2, 27)
    print(f"\n=== In-basin (sphere R={R} Bohr) l-population, Fe1 ===")
    print(f"{'l':>2} {'charge':>10} {'spin(up-dn)':>12}")
    totc = tots = 0.0
    for l in (0, 1, 2, 3):
        ns = sorted(n for (ll, n) in chi if ll == l)
        c = s = 0.0
        for n in ns:
            for n2 in ns:
                I = rad[(l, n, n2)]
                if abs(I) < 1e-14:
                    continue
                cu = sum(up[blocks[(0, l, n)][m], blocks[(0, l, n2)][m]]
                         for m in range(len(blocks[(0, l, n)])))
                cd = sum(dn[blocks[(0, l, n)][m], blocks[(0, l, n2)][m]]
                         for m in range(len(blocks[(0, l, n)])))
                c += I * (cu + cd)
                s += I * (cu - cd)
        print(f"{l:>2} {c:>10.5f} {s:>12.5f}")
        totc += c
        tots += s
    print(f"{'sum':>2} {totc:>10.5f} {tots:>12.5f}")


if __name__ == "__main__":
    main()
