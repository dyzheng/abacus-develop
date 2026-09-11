#!/usr/bin/env python3
"""Build the DeltaSpin control STRU (S5) from the plain FeO STRU.

Usage: make_deltaspin_stru.py <src-STRU> <dst-STRU> <M2> <M3>

The DeltaSpin STRU syntax is `<pos> mag <target> sc <x> <y> <z>`: the `mag`
value becomes the target moment (UnitCell::get_target_mag reads m_loc_, which
the single-number `mag` form puts on z) and the three `sc` flags select the
constrained Cartesian components.  Only Fe atom 2 is constrained here, matching
the constraint-run setup where the second Fe is left free.
"""
import sys


def main():
    src, dst, m2, m3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    txt = open(src).read()
    txt = txt.replace("0.00   0.00   0.00   mag  2.0",
                      f"0.00   0.00   0.00   mag  {m2}   sc 0 0 1")
    txt = txt.replace("0.50   0.50   0.50   mag  -2.0",
                      f"0.50   0.50   0.50   mag  {m3}")
    open(dst, "w").write(txt)


if __name__ == "__main__":
    main()
