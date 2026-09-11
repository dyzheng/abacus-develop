#!/bin/bash
# I-1 MgO bulk wide-charge scan (plan: docs/superpowers/plans/2026-09-09-p0-cases-feomgo.md).
#
# Core question (R4 verdict): is the linear reachable domain of the charge
# constraint on an ionic solid significantly wider than the +-0.3 e window
# measured on H2O?
#
# Steps (all runs: LCAO, symmetry 0, 2x2x2 MP, KPAR=1, constraint framework):
#   S1  grid calibration: Q_ref(O sublattice) at ecutwfc/ecutrho 60/240 vs
#       80/320 with delta=0 (the reference phase is the mu=0 observation)
#   S2  reference run at the calibrated grid: Q_ref(O)/Q_ref(Mg) + sum rule
#   S3  single-O scan +-0.3/0.5/0.8/1.0 e (hot start)
#   S4  single-Mg scan +-1.0 e (Mg -> Mg3+ expected UNREACHABLE)
#   S5  summary (kappa, linear window)
#
# Usage: bash run_mgo_scan.sh <S1|S2|S3|S4|S5> [nproc]
set -uo pipefail

STEP="${1:?usage: run_mgo_scan.sh <S1|S2|S3|S4|S5> [nproc]}"
NPROC="${2:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
MPIRUN="${MPIRUN:-mpirun}"
CASEDIR="$(cd "$(dirname "$0")" && pwd)"
PPORB="$(cd "$CASEDIR/.." && pwd)/PP_ORB"
WORKROOT="${WORKROOT:-/tmp/mgo_scan}"
RESDIR="$CASEDIR/results"
export OMP_NUM_THREADS=1
mkdir -p "$WORKROOT" "$RESDIR"

# One constrained (or plain) MgO SCF in its own work directory.
#   run_case <name> <ecutwfc> <ecutrho> <target-json> [restart_from_dir]
run_case()
{
    local name="$1" ecut="$2" rho="$3" json="$4" restart="${5:-}"
    local work="$WORKROOT/$name"
    rm -rf "$work"
    mkdir -p "$work"
    cp "$CASEDIR/STRU" "$CASEDIR/KPT" "$work/"
    printf '%s\n' "$json" > "$work/constraint_target.json"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix                autotest"
        echo "calculation           scf"
        echo "basis_type            lcao"
        echo "gamma_only            0"
        echo "nspin                 1"
        echo "nbands                40"
        echo "ecutwfc               $ecut"
        echo "ecutrho               $rho"
        echo "scf_thr               1e-8"
        echo "scf_nmax              ${SCF_NMAX:-800}"
        echo "smearing_method       gaussian"
        echo "smearing_sigma        0.002"
        echo "mixing_type           broyden"
        echo "mixing_beta           0.7"
        echo "ks_solver             genelpa"
        echo "symmetry              0"
        echo "pseudo_dir            $PPORB"
        echo "orbital_dir           $PPORB"
        if [ "${CONSTRAIN:-1}" = "1" ]; then
            echo "constraint            true"
            echo "constraint_type       charge"
            echo "constraint_weight_type becke"
            echo "constraint_target_file constraint_target.json"
            echo "constraint_target_mode delta"
            echo "constraint_mu_max     5.0"
            echo "constraint_thr        1e-4"
        fi
        if [ -n "$restart" ]; then
            # Branch: warm start from a previous run's density matrix.
            echo "init_chg              file"
            echo "read_file_dir         $restart"
        fi
    } > "$work/INPUT"
    echo "===== [$name] ecutwfc=$ecut ecutrho=$rho nproc=$NPROC work=$work"
    ( cd "$work" && $MPIRUN -np "$NPROC" "$ABACUS" > run.log 2>&1 )
    local rc=$?
    echo "----- [$name] exit=$rc"
    grep -E "CONSTRAINT_AUDIT|final status|charge density convergence|^!FINAL_ETOT_IS|Total  Time|WARNING|ERROR" \
        "$work/run.log" | tail -20
    cp "$work/run.log" "$RESDIR/$name.log"
    return $rc
}

audit()
{
    grep -E "CONSTRAINT_AUDIT|final status" "$WORKROOT/$1/run.log"
}

case "$STEP" in
S1)
    # Grid calibration: the O-sublattice reference charge must be grid-converged
    # to < 3e-5 e (= constraint_thr/3) before any delta scan is meaningful.
    # A fixed basis (LCAO) still needs a converged real-space grid: ecutrho sets
    # the partition/integration grid the Becke weights and the density live on.
    JSON='{"constraints": [{"type": "charge", "target": 0.0, "atoms": [4, 5, 6, 7]}]}'
    for g in ${S1_GRIDS:-60:240 80:320 100:400}; do
        ecut="${g%%:*}"; rho="${g##*:}"
        run_case "S1_grid${ecut}" "$ecut" "$rho" "$JSON"
    done
    echo "===== S1 verdict (Q_ref of the 4-atom O sublattice, e) ====="
    prev=""
    for g in ${S1_GRIDS:-60:240 80:320 100:400}; do
        ecut="${g%%:*}"
        q=$(grep -m1 "CONSTRAINT_AUDIT c\[0\]" "$WORKROOT/S1_grid${ecut}/OUT.autotest/running_scf.log" \
            | sed -E 's/.* q=([-0-9.eE+]+) .*/\1/')
        dev=$(grep -m1 "CONSTRAINT_AUDIT nconstraint" "$WORKROOT/S1_grid${ecut}/OUT.autotest/running_scf.log" \
            | sed -E 's/.*maxdev=([-0-9.eE+]+).*/\1/')
        echo "  ecutwfc=$ecut  Q_O4=$q  perO=$(python3 -c "print(f'{$q/4:.10f}')")  maxdev=$dev"
        if [ -n "$prev" ]; then
            echo "    dQ vs previous grid = $(python3 -c "print(f'{abs($q-$prev):.3e}')") e"
        fi
        prev="$q"
    done
    ;;
S1X)
    # Two-factor separation (S1 follow-up): is the Q_ref grid drift driven by
    # ecutrho (the partition/integration grid) or by ecutwfc (which in LCAO
    # still sets the auxiliary grid the density is evaluated on)?
    # Cross cells: (ecutwfc=60, ecutrho=400) and (100, 240).
    JSON='{"constraints": [{"type": "charge", "target": 0.0, "atoms": [4, 5, 6, 7]}]}'
    run_case S1x_e60r400 60 400 "$JSON"
    run_case S1x_e100r240 100 240 "$JSON"
    echo "===== S1X cross (Q_ref of the O sublattice, e) ====="
    for n in S1_grid60 S1x_e60r400 S1_grid100 S1x_e100r240; do
        f="$WORKROOT/$n/OUT.autotest/running_scf.log"
        q=$(grep -m1 "CONSTRAINT_AUDIT c\[0\]" "$f" | sed -E 's/.* q=([-0-9.eE+]+) .*/\1/')
        echo "  $n  Q_O4=$q"
    done
    ;;
S2)
    GRID_ECUT="${GRID_ECUT:-160}"
    GRID_RHO="${GRID_RHO:-640}"
    # Full-coverage partition (O sublattice + Mg sublattice) with delta=0:
    # the reference phase records Q_ref for both while sum rule
    # (total_charge == nelec = 64) audits the periodic-image weight partition.
    JSON='{"constraints": [{"type": "charge", "target": 0.0, "atoms": [4, 5, 6, 7]}, {"type": "charge", "target": 0.0, "atoms": [0, 1, 2, 3]}]}'
    run_case S2_ref "$GRID_ECUT" "$GRID_RHO" "$JSON"
    echo "===== S2 audit ====="; audit S2_ref
    ;;
S3)
    # O single-atom (atom 4) charge scan.  Two chains (negative side, positive
    # side), each hot-started: the first point of a chain restarts from the S2
    # reference density (delta = 0 => free density), later points from the
    # previous point in the same chain, so the |delta| sequence walks one
    # branch continuously (plan S3).
    ECUT="${GRID_ECUT:-60}"; RHO="${GRID_RHO:-240}"
    for chain in ${S3_CHAINS:--0.3 -0.5 -0.8 -1.0 ; 0.3 0.5 0.8 1.0}; do
        prev="$WORKROOT/S2_ref/OUT.autotest"
        for d in $chain; do
            # Branch: within a chain the hot start always walks from the
            # previous (nearby) delta; a chain starts from the free density.
            tag=$(echo "$d" | sed 's/^+//; s/-/m/; s/\./p/')
            JSON="{\"constraints\": [{\"type\": \"charge\", \"target\": $d, \"atoms\": [4]}]}"
            run_case "S3_O_${tag}" "$ECUT" "$RHO" "$JSON" "$prev"
            prev="$WORKROOT/S3_O_${tag}/OUT.autotest"
        done
    done
    echo "===== S3 table (single O, ecutwfc=$ECUT) ====="
    python3 - "$WORKROOT" "-0.3 -0.5 -0.8 -1.0 0.3 0.5 0.8 1.0" <<'PYEOF'
import re, sys, os
work, deltas = sys.argv[1], sys.argv[2].split()
print(f"  {'delta':>7} {'Q_ref':>14} {'q_final':>14} {'mu(Ry)':>12} {'outer':>6}  status")
for d in deltas:
    tag = d.lstrip('+').replace('-', 'm').replace('.', 'p')
    f = os.path.join(work, f"S3_O_{tag}", "OUT.autotest", "running_scf.log")
    txt = open(f).read()
    rows = re.findall(r"CONSTRAINT_AUDIT c\[0\] kind=\S+ q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    qref = rows[0][0] if rows else "?"
    q, mu = rows[-1][0], rows[-1][2]
    outer = txt.count("[constraint] outer step")
    st = re.search(r"final status: (\w+)", txt)
    print(f"  {d:>7} {qref:>14} {q:>14} {mu:>12} {outer:>6}  {st.group(1) if st else '?'}")
PYEOF
    ;;
S4)
    # Mg single-atom (atom 0) charge scan: +1.0 e (Mg(0) side) and -1.0 e
    # (Mg(3+) side, expected UNREACHABLE = the physical fuse case).
    ECUT="${GRID_ECUT:-160}"; RHO="${GRID_RHO:-640}"
    for d in ${DELTAS_MG:-1.0 -1.0}; do
        tag=$(echo "$d" | sed 's/^+//; s/-/m/; s/\./p/')
        JSON="{\"constraints\": [{\"type\": \"charge\", \"target\": $d, \"atoms\": [0]}]}"
        run_case "S4_Mg_${tag}" "$ECUT" "$RHO" "$JSON" "$WORKROOT/S2_ref/OUT.autotest"
        echo "-- Mg delta=$d"; audit "S4_Mg_${tag}"
    done
    ;;
S5)
    # Summary: mu(delta) response, local slope |dmu/ddelta| (Ry/e), deviation
    # from the lowest-|delta| pair, and the resulting linear window; compared
    # against the H2O reference (mu* ~ -1.7*delta over +-0.3 e, 7/7 reachable,
    # docs/superpowers/specs/2026-08-31-v1-v3-validation.md).
    python3 - "$WORKROOT" <<'PYEOF'
import re, os, sys
work = sys.argv[1]

def tag(d):
    return f"{d:.1f}".lstrip('+').replace('-', 'm').replace('.', 'p')

def read_case(prefix, deltas):
    out = []
    for d in deltas:
        f = os.path.join(work, f"{prefix}{tag(d)}", "OUT.autotest", "running_scf.log")
        if not os.path.exists(f):
            out.append((d, None, None, None, None, 'MISSING'))
            continue
        txt = open(f).read()
        rows = re.findall(r"CONSTRAINT_AUDIT c\[0\] kind=\S+ q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
        outer = txt.count("[constraint] outer step")
        st = re.search(r"final status: (\w+)", txt)
        out.append((d, float(rows[-1][0]), float(rows[0][0]), float(rows[-1][2]), outer,
                    st.group(1) if st else 'RUNNING'))
    return out

def report(title, rows, href):
    print(f"\n--- {title} ---")
    print(f"  {'delta':>7} {'Q_ref':>12} {'Q_final':>12} {'mu(Ry)':>10} {'outer':>6} "
          f"{'status':>10} {'kappa_cum':>9} {'|slope|':>9} {'dev%':>7}")
    # Reference kappa = the slope of the first (lowest |delta|) point, measured
    # from delta = 0, i.e. the small-perturbation stiffness of that side.
    ref = None
    if rows and rows[0][1] is not None and rows[0][0] != 0.0:
        ref = abs(rows[0][3] / rows[0][0])
    for i, (d, q, qref, mu, outer, st) in enumerate(rows):
        slope = dev = ""
        if i > 0 and q is not None and rows[i-1][1] is not None:
            dp = rows[i-1][0]
            # Branch: a slope is only meaningful between two points on the
            # same side of delta = 0 (different sides have different physics).
            if dp * d > 0 and abs(d - dp) > 1e-9:
                s_ = abs((mu - rows[i-1][3]) / (d - dp))
                slope = f"{s_:.3f}"
                if ref:
                    pct = 100.0 * (s_ - ref) / ref
                    dev = f"{pct:+.1f}"
                    # Linear-window marker: >20% deviation from the reference.
                    if abs(pct) > 20.0:
                        slope += "*"
        kc = f"{abs(mu/d):9.3f}" if q is not None and d != 0.0 else f"{'--':>9}"
        qs = f"{q:12.6f}" if q is not None else f"{'--':>12}"
        qrs = f"{qref:12.6f}" if qref is not None else f"{'--':>12}"
        mus = f"{mu:10.4f}" if mu is not None else f"{'--':>10}"
        print(f"  {d:>7.2f} {qrs} {qs} {mus} {str(outer):>6} {st:>10} {kc} {slope:>9} {dev:>7}")
    if ref:
        print(f"  reference kappa (delta=0 -> {rows[0][0]:+.2f}) = {ref:.3f} Ry/e"
              f"  (* = local slope >20% away = outside the linear window)")
    print(f"  H2O reference: {href}")

report("S3 single O (negative side)", read_case("S3_O_", [-0.3, -0.5, -0.8, -1.0]),
       "mu* = -1.7*delta (i.e. |slope| 1.7 Ry/e), +-0.3 e all reachable, no fuse")
report("S3 single O (positive side)", read_case("S3_O_", [0.3, 0.5, 0.8, 1.0]),
       "same")
report("S4 single Mg", read_case("S4_Mg_", [1.0, -1.0]), "n/a")
PYEOF
    ;;
*)
    echo "step $STEP not wired yet"; exit 2;;
esac
