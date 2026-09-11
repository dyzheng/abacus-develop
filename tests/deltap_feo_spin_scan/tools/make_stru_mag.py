#!/usr/bin/env python3
"""Rewrite the Fe `mag` starting guesses in the II-1 FeO STRU.

Usage: make_stru_mag.py <src-STRU> <dst-STRU> <mag-Fe2> <mag-Fe3>

The baseline triage starts the SCF from several magnetic guesses in order to
map the self-consistent solution set of the DFT+U FeO cell.  The two Fe lines
carry the only `mag` keys in the inherited STRU, so the substitution is
unambiguous.
"""
import sys


def main():
    src, dst, m2, m3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    txt = open(src).read()
    txt = txt.replace("0.00   0.00   0.00   mag  2.0",
                      "0.00   0.00   0.00   mag  " + m2)
    txt = txt.replace("0.50   0.50   0.50   mag  -2.0",
                      "0.50   0.50   0.50   mag  " + m3)
    open(dst, "w").write(txt)


if __name__ == "__main__":
    main()
