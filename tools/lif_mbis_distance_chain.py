#!/usr/bin/env python3
"""LiF distance chain with MBIS target-Q constraints."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lif_mbis_q_homotopy as h


def write_stru(template_dir, dst, distance):
    text = (template_dir / "STRU").read_text()
    fz = 5.0 + distance
    text = text.replace("10.000 10.000 8.000", f"10.000 10.000 {fz:.3f}")
    dst.write_text(text)


def write_input(path, suffix, args, read_dir):
    lines = [
        "INPUT_PARAMETERS",
        f"suffix               {suffix}",
        "calculation          scf",
        "basis_type           pw",
        "gamma_only           1",
        f"nspin                {args.nspin}",
        f"nbands               {args.nbands}",
        f"ecutwfc              {args.ecutwfc}",
        f"ecutrho              {args.ecutrho}",
        f"scf_thr              {args.scf_thr}",
        f"scf_nmax             {args.scf_nmax}",
        "symmetry             0",
        f"pseudo_dir           {args.pseudo_dir}",
        f"orbital_dir          {args.orbital_dir}",
        f"nelec                {args.nelec}",
        "smearing_method      gaussian",
        "smearing_sigma       0.005",
        "mixing_type          broyden",
        f"mixing_beta          {args.mixing_beta}",
        "device               gpu",
        "ks_solver            dav_subspace",
        "pw_diag_ndim         2",
        f"dft_functional       {args.functional}",
        "constraint                 true",
        "constraint_type            charge",
        "constraint_weight_type     mbis",
        "constraint_target_file     constraint_target.json",
        "constraint_target_mode     absolute",
        "constraint_mbis_images     1",
        f"constraint_mbis_cutoff     {args.cutoff}",
        f"constraint_kappa_min       {args.kappa_min}",
        f"constraint_step_max        {args.step_max}",
        f"constraint_step_probe      {args.step_probe}",
        f"constraint_mu_max          {args.mu_max}",
        f"constraint_thr             {args.constraint_thr}",
        "constraint_branch_tol      0",
        "constraint_start_constrained true",
        "init_chg             file",
        f"read_file_dir        {read_dir}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_target(path, target, mu):
    obj = {
        "constraints": [
            {"type": "charge", "target": target[0], "atoms": [0],
             "initial_mu": mu[0]},
            {"type": "charge", "target": target[1], "atoms": [1],
             "initial_mu": mu[1]},
        ]
    }
    path.write_text(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/opencode/lif_chain_ionic")
    ap.add_argument("--template-dir", default="/tmp/opencode/lif_homotopy/lif_homo_free")
    ap.add_argument("--seed-dir", default="/tmp/opencode/lif_homotopy/step_00_q2.04")
    ap.add_argument("--seed-suffix", default="step_00_q2.04")
    ap.add_argument("--mu-start", default="0,0")
    ap.add_argument("--target", default="2.0,8.0")
    ap.add_argument("--r-start", type=float, default=3.0)
    ap.add_argument("--r-end", type=float, default=10.0)
    ap.add_argument("--dr", type=float, default=0.25)
    ap.add_argument("--functional", default="MGGA_X_R2SCAN+MGGA_C_R2SCAN")
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--nbands", type=int, default=6)
    ap.add_argument("--nelec", type=float, default=10.0)
    ap.add_argument("--ecutwfc", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=160.0)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--kappa-min", type=float, default=0.05)
    ap.add_argument("--step-max", type=float, default=0.1)
    ap.add_argument("--step-probe", type=float, default=0.05)
    ap.add_argument("--mu-max", type=float, default=5.0)
    ap.add_argument("--constraint-thr", type=float, default=1.0e-3)
    ap.add_argument("--scf-nmax", type=int, default=250)
    ap.add_argument("--scf-thr", type=float, default=1.0e-6)
    ap.add_argument("--mixing-beta", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--orbital-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--abacus", default="/root/abacus-develop/build_fix_cuda/abacus_basic_gpu")
    args = ap.parse_args()

    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    template_dir = Path(args.template_dir)
    target = tuple(float(x) for x in args.target.split(","))
    mu = tuple(float(x) for x in args.mu_start.split(","))
    src_dir = Path(args.seed_dir)
    src_suffix = args.seed_suffix
    if h.find_restart(src_dir, src_suffix, "CHARGE-DENSITY") is None:
        raise SystemExit(f"seed restart not found in {src_dir}")

    distances = []
    r = args.r_start
    while r <= args.r_end + 1.0e-9:
        distances.append(round(r, 2))
        r += args.dr

    summary = []
    for r in distances:
        suffix = f"lif_r{r:.2f}"
        d = root / f"r{r:.2f}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copy2(template_dir / "KPT", d / "KPT")
        write_stru(template_dir, d / "STRU", r)
        h.copy_restart(src_dir, src_suffix, d, suffix)
        write_target(d / "constraint_target.json", target, mu)
        write_input(d / "INPUT", suffix, args, d)
        print(f"[r] {r:.2f} target={target} mu={mu}", flush=True)
        try:
            log = h.run_abacus(d, args, suffix)
        except subprocess.TimeoutExpired:
            res = {"status": "TIMEOUT", "audits": [], "etot": None,
                   "converged": False}
        else:
            res = h.parse_log(log)
        res["dir"] = str(d)
        res["suffix"] = suffix
        res["r"] = r
        res["mu_in"] = mu
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n")
        summary.append(res)
        if res.get("status") != "CONVERGED":
            print(f"[stop] r={r:.2f} status={res.get('status')} "
                  f"etot={res.get('etot')}", flush=True)
            break
        q = {a["index"]: a for a in res["audits"]}
        if 0 not in q or 1 not in q:
            print(f"[stop] r={r:.2f} no full audit", flush=True)
            break
        mu = (q[0]["mu"], q[1]["mu"])
        src_dir, src_suffix = d, suffix
        print(f"[ok] r={r:.2f} Q=({q[0]['q']:.6f},{q[1]['q']:.6f}) "
              f"mu=({mu[0]:.4f},{mu[1]:.4f}) E={res.get('etot')}", flush=True)
        (root / "chain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    (root / "chain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] summary: {root / 'chain_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
