#!/usr/bin/env python3
"""MBIS fixed-mu scan driver (small mu steps, no ocp_set)."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lif_mbis_q_homotopy as h


def write_fixed_input(path, suffix, args, read_dir):
    lines = [
        "INPUT_PARAMETERS",
        f"suffix               {suffix}",
        "calculation          scf",
        "basis_type           pw",
        "gamma_only           1",
        f"nspin                {args.nspin}",
        f"nbands               {args.nbands}",
        "ecutwfc              40",
        "ecutrho              160",
        "scf_thr              1.0e-6",
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
        f"constraint_mu_max          {args.mu_max}",
        "constraint_branch_tol      0",
        "constraint_start_constrained true",
        "init_chg             file",
        f"read_file_dir        {read_dir}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_fixed_target(path, target, mu_li, mu_f):
    obj = {
        "constraints": [
            {"type": "charge", "target": target[0], "atoms": [0], "fixed_mu": mu_li},
            {"type": "charge", "target": target[1], "atoms": [1], "fixed_mu": mu_f},
        ]
    }
    path.write_text(json.dumps(obj) + "\n")


def run_step(root, name, template_dir, args, src_dir, src_suffix, mu_common, target):
    d = h.prepare_dir(root, name, template_dir)
    h.copy_restart(src_dir, src_suffix, d, name)
    mu_li = -mu_common
    mu_f = mu_common
    write_fixed_target(d / "constraint_target.json", target, mu_li, mu_f)
    write_fixed_input(d / "INPUT", name, args, d)
    print(f"[mu] {name} mu=({mu_li:.4f},{mu_f:.4f})", flush=True)
    try:
        log = h.run_abacus(d, args, name)
    except subprocess.TimeoutExpired:
        return d, {"status": "TIMEOUT", "audits": [], "etot": None, "converged": False}
    res = h.parse_log(log)
    res["dir"] = str(d)
    res["suffix"] = name
    (d / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"[mu] status={res['status']} etot={res['etot']} ok={res['converged']}", flush=True)
    return d, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/opencode/lif_fixed_mu")
    ap.add_argument("--template-dir", default="/tmp/opencode/lif_guess/direct_neutral_r3")
    ap.add_argument("--abacus", default="/root/abacus-develop/build_fix_cuda/abacus_basic_gpu")
    ap.add_argument("--pseudo-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--orbital-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--functional", default="MGGA_X_R2SCAN+MGGA_C_R2SCAN")
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--nbands", type=int, default=6)
    ap.add_argument("--nelec", type=float, default=10.0)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--mu-max", type=float, default=5.0)
    ap.add_argument("--scf-nmax", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--mixing-beta", type=float, default=0.2)
    ap.add_argument("--dmu", type=float, default=0.02)
    ap.add_argument("--nsteps", type=int, default=25)
    ap.add_argument("--target", default="3.0,7.0")
    ap.add_argument("--seed-dir", default="/tmp/opencode/lif_homotopy/lif_homo_free")
    ap.add_argument("--seed-suffix", default="lif_homo_free")
    ap.add_argument("--mu-start", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    template_dir = Path(args.template_dir)
    target = tuple(float(x) for x in args.target.split(","))
    src_dir = Path(args.seed_dir)
    src_suffix = args.seed_suffix
    if not src_suffix:
        raise SystemExit("--seed-suffix is required")
    seed_ok = h.find_restart(src_dir, src_suffix, "CHARGE-DENSITY") is not None
    if not seed_ok:
        raise SystemExit(f"seed restart not found in {src_dir}")

    summary = []
    dmu = args.dmu
    last_good_mu = args.mu_start
    k = 0
    failures = 0
    while k < args.nsteps:
        mu_common = last_good_mu + dmu
        name = f"mu_{k:03d}_{mu_common:.3f}"
        d, res = run_step(root, name, template_dir, args, src_dir, src_suffix, mu_common, target)
        summary.append({"step": k, "mu": mu_common, "result": res})
        if not res["converged"]:
            failures += 1
            print(f"[retry] mu={mu_common:.4f} failed, halving dmu to {dmu/2:.4f}", flush=True)
            dmu /= 2.0
            if failures >= 5 or dmu < 0.001:
                print("[stop] too many failures or dmu too small", flush=True)
                break
            continue
        failures = 0
        q = {}
        for a in res["audits"]:
            q[a["index"]] = a
        if 0 not in q or 1 not in q:
            print(f"[stop] mu={mu_common:.4f} has no full audit", flush=True)
            break
        print(f"[ok] mu={mu_common:.4f} Q=({q[0]['q']:.6f},{q[1]['q']:.6f})", flush=True)
        last_good_mu = mu_common
        src_dir, src_suffix = d, name
        if q[0]["q"] >= target[0] - 0.02:
            print(f"[stop] reached target Q_Li={q[0]['q']:.6f}", flush=True)
            break
        k += 1
    (root / "fixed_mu_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] summary: {root / 'fixed_mu_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
