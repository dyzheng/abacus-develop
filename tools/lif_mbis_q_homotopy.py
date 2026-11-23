#!/usr/bin/env python3
"""MBIS target-Q homotopy driver (direct constrained start, no ocp_set)."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AUDIT_RE = re.compile(
    r"CONSTRAINT_AUDIT c\[(\d+)\] kind=charge q=([-+0-9.eE]+) "
    r"t=([-+0-9.eE]+) mu=([-+0-9.eE]+)"
)
STATUS_RE = re.compile(r"\[constraint\] final status:\s*(\w+)")
ETOT_RE = re.compile(r"FINAL_ETOT_IS\s+([-+0-9.eE]+)")


def write_input(path, suffix, args, constrained, read_dir=None, init_mu=None):
    lines = [
        "INPUT_PARAMETERS",
        f"suffix               {suffix}",
        "calculation          scf",
        "basis_type           pw",
        "gamma_only           1",
        f"nspin                {args.nspin}",
        f"nupdown              {args.nupdown}",
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
    ]
    if constrained:
        lines += [
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
    else:
        lines += ["constraint           false", "init_chg             atomic"]
    path.write_text("\n".join(lines) + "\n")


def write_target(path, q_li, q_f, mu_li, mu_f):
    obj = {
        "constraints": [
            {"type": "charge", "target": q_li, "atoms": [0], "initial_mu": mu_li},
            {"type": "charge", "target": q_f, "atoms": [1], "initial_mu": mu_f},
        ]
    }
    path.write_text(json.dumps(obj) + "\n")


def parse_log(text):
    status = None
    m = STATUS_RE.search(text)
    if m:
        status = m.group(1)
    audits = []
    for m in AUDIT_RE.finditer(text):
        audits.append(
            {
                "index": int(m.group(1)),
                "q": float(m.group(2)),
                "t": float(m.group(3)),
                "mu": float(m.group(4)),
            }
        )
    etot = None
    m = ETOT_RE.search(text)
    if m:
        etot = float(m.group(1))
    converged = (status == "CONVERGED") or ("#SCF IS CONVERGED#" in text)
    if "SCF IS NOT CONVERGED" in text:
        converged = False
    return {
        "status": status,
        "audits": audits,
        "etot": etot,
        "converged": converged,
    }


def find_restart(directory, suffix, kind):
    name = f"{suffix}-{kind}.restart"
    candidates = [directory / name, directory / f"OUT.{suffix}" / name]
    for p in candidates:
        if p.exists():
            return p
    return None


def copy_restart(src_dir, src_suffix, dst_dir, dst_suffix, required=True):
    got = []
    for kind in ("CHARGE-DENSITY", "TAU-DENSITY"):
        p = find_restart(src_dir, src_suffix, kind)
        if p is None:
            if required and kind == "CHARGE-DENSITY":
                raise FileNotFoundError(f"missing {src_suffix}-{kind}.restart in {src_dir}")
            continue
        shutil.copy2(p, dst_dir / f"{dst_suffix}-{kind}.restart")
        got.append(kind)
    return got


def run_abacus(run_dir, args, suffix):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp)
    cmd = [args.abacus]
    with open(run_dir / "run.log", "w") as out:
        subprocess.run(cmd, cwd=run_dir, env=env, stdout=out, stderr=subprocess.STDOUT,
                       timeout=args.timeout)
    parts = [(run_dir / "run.log").read_text(errors="replace")]
    detailed = run_dir / f"OUT.{suffix}" / "running_scf.log"
    if detailed.exists():
        parts.append(detailed.read_text(errors="replace"))
    return "\n".join(parts)


def prepare_dir(root, name, template_dir):
    d = root / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy2(template_dir / "STRU", d / "STRU")
    shutil.copy2(template_dir / "KPT", d / "KPT")
    return d


def run_free_seed(root, seed_name, template_dir, args):
    d = prepare_dir(root, seed_name, template_dir)
    write_input(d / "INPUT", seed_name, args, constrained=False)
    print(f"[seed] running free SCF in {d}", flush=True)
    log = run_abacus(d, args, seed_name)
    res = parse_log(log)
    res["dir"] = str(d)
    res["suffix"] = seed_name
    (root / "seed_result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"[seed] status={res['status']} etot={res['etot']} ok={res['converged']}", flush=True)
    return d, seed_name, res


def run_step(root, name, template_dir, args, src_dir, src_suffix, q, mu):
    d = prepare_dir(root, name, template_dir)
    copy_restart(src_dir, src_suffix, d, name)
    write_target(d / "constraint_target.json", q[0], q[1], mu[0], mu[1])
    write_input(d / "INPUT", name, args, constrained=True, read_dir=d)
    print(f"[step] {name} target=({q[0]:.4f},{q[1]:.4f}) mu=({mu[0]:.4f},{mu[1]:.4f})", flush=True)
    try:
        log = run_abacus(d, args, name)
    except subprocess.TimeoutExpired:
        return d, {"status": "TIMEOUT", "audits": [], "etot": None, "converged": False}
    res = parse_log(log)
    res["dir"] = str(d)
    res["suffix"] = name
    (d / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"[step] status={res['status']} etot={res['etot']} ok={res['converged']}", flush=True)
    return d, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/opencode/lif_homotopy")
    ap.add_argument("--template-dir", default="/tmp/opencode/lif_guess/direct_neutral_r3")
    ap.add_argument("--abacus", default="/root/abacus-develop/build_fix_cuda/abacus_basic_gpu")
    ap.add_argument("--pseudo-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--orbital-dir", default="/root/abacus-develop/tests/PP_ORB")
    ap.add_argument("--functional", default="MGGA_X_R2SCAN+MGGA_C_R2SCAN")
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--nupdown", type=float, default=0.0)
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
    ap.add_argument("--scf-nmax", type=int, default=200)
    ap.add_argument("--scf-thr", type=float, default=1.0e-6)
    ap.add_argument("--mixing-beta", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--q-start", default="2.0,8.0")
    ap.add_argument("--q-target", default="3.0,7.0")
    ap.add_argument("--mu-start", default="0,0")
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--seed-dir", default=None)
    ap.add_argument("--seed-suffix", default=None)
    ap.add_argument("--resume-step", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    template_dir = Path(args.template_dir)
    q_start = tuple(float(x) for x in args.q_start.split(","))
    q_target = tuple(float(x) for x in args.q_target.split(","))
    mu = tuple(float(x) for x in args.mu_start.split(","))

    if args.seed_dir:
        src_dir = Path(args.seed_dir)
        src_suffix = args.seed_suffix
        if not src_suffix:
            raise SystemExit("--seed-suffix is required with --seed-dir")
        seed_ok = find_restart(src_dir, src_suffix, "CHARGE-DENSITY") is not None
        if not seed_ok:
            raise SystemExit(f"seed restart not found in {src_dir}")
        print(f"[seed] using {src_dir} suffix={src_suffix}", flush=True)
    else:
        src_dir, src_suffix, seed_res = run_free_seed(root, "lif_homo_free", template_dir, args)
        if not seed_res["converged"]:
            raise SystemExit("free seed SCF did not converge")
        mu = tuple(float(x) for x in args.mu_start.split(","))

    summary = []
    for k in range(args.resume_step, args.nsteps + 1):
        frac = k / args.nsteps
        q = (
            q_start[0] + frac * (q_target[0] - q_start[0]),
            q_start[1] + frac * (q_target[1] - q_start[1]),
        )
        name = f"step_{k:02d}_q{q[0]:.2f}"
        d, res = run_step(root, name, template_dir, args, src_dir, src_suffix, q, mu)
        summary.append({"step": k, "q": q, "mu_in": mu, "result": res})
        if not res["converged"]:
            print(f"[stop] step {k} failed: {res['status']}", flush=True)
            break
        q_audit = {}
        for a in res["audits"]:
            q_audit[a["index"]] = a
        if 0 not in q_audit or 1 not in q_audit:
            print(f"[stop] step {k} has no full audit", flush=True)
            break
        mu = (q_audit[0]["mu"], q_audit[1]["mu"])
        src_dir, src_suffix = d, name
        print(f"[ok] step {k} Q=({q_audit[0]['q']:.4f},{q_audit[1]['q']:.4f}) "
              f"mu=({mu[0]:.4f},{mu[1]:.4f})", flush=True)
    (root / "homotopy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] summary: {root / 'homotopy_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
