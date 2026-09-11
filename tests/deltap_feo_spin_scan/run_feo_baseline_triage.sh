#!/bin/bash
# II-1 S1T -- baseline triage for the inherited FeO cell (PW / DFT+U).
#
# Why this step exists: II-1a found that the inherited baseline
# (tests/17_DS_DFTU/50_FeO_O_first_Fe_second) has at least two *converged
# unconstrained* AFM solutions on its own Gamma-only k mesh, and that
# result.ref records the higher one (E = -7652.3958757 eV, Fe +-3.485 uB)
# while a slightly perturbed start reaches the lower one
# (E = -7653.0079615 eV, Fe +-3.715 uB).  A constraint scan anchored on a
# metastable reference measures nothing, so this step maps the solution set
# and decides where the scan must be anchored.
#
# Two knobs are varied:
#   * the initial magnetic guess in STRU -- which basin the SCF starts in;
#   * the k mesh -- whether the multistability is a Gamma-only artifact.
#
# Usage: bash run_feo_baseline_triage.sh [nproc]
# Records: results/triage/<case>.txt, one trimmed record per case
#          (.log would be swallowed by the repo-wide *.log gitignore).
set -uo pipefail

NPROC="${1:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build_rel/abacus_basic_para}"
MPIRUN="${MPIRUN:-mpirun}"
CASEDIR="$(cd "$(dirname "$0")" && pwd)"
PPORB="$(cd "$CASEDIR/.." && pwd)/PP_ORB"
WORKROOT="${WORKROOT:-/tmp/feo_triage}"
RESDIR="$CASEDIR/results/triage"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

ECUT="${ECUT:-50}"
SCF_THR="${SCF_THR:-1e-7}"
SCF_NMAX="${SCF_NMAX:-400}"
# out_chg for the next run_case call.  Left explicit (rather than a
# parameter) because the only case that needs it is the warm-start donor,
# whose whole point is to write the DFT+U onsite.dm next to the density.
OUTCHG="${OUTCHG:-0}"
# Space-separated case-name filter: lets a single case be re-run after a
# driver edit without repeating the whole triage.  Empty means "run all".
ONLY="${ONLY:-}"

mkdir -p "$WORKROOT" "$RESDIR"

# want <name>: 0 when <name> is selected by the ONLY filter.
want()
{
    # Branch: an empty filter selects every case (the normal full triage).
    if [ -z "$ONLY" ]; then
        return 0
    fi
    case " $ONLY " in
        *" $1 "*) return 0;;
        *) return 1;;
    esac
}

# run_case <name> <kmesh "n1 n2 n3"> <mag2> <mag3> [restart-dir]
run_case()
{
    local name="$1" kmesh="$2" mag2="$3" mag3="$4" restart="${5:-}"
    # Branch: honour the ONLY filter so single cases can be replayed.
    if ! want "$name"; then
        echo "===== [$name] skipped (ONLY filter)"
        return 0
    fi
    local work="$WORKROOT/$name"
    python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$work"
    mkdir -p "$work"
    python3 "$CASEDIR/tools/make_stru_mag.py" "$CASEDIR/STRU" "$work/STRU" "$mag2" "$mag3"
    {
        echo "K_POINTS"
        echo "0"
        echo "Gamma"
        echo "$kmesh 0 0 0"
    } > "$work/KPT"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix                autotest"
        echo "calculation           scf"
        echo "basis_type            pw"
        echo "gamma_only            0"
        echo "ecutwfc               $ECUT"
        echo "nspin                 2"
        echo "scf_thr               $SCF_THR"
        echo "scf_nmax              $SCF_NMAX"
        echo "out_chg               $OUTCHG"
        echo "smearing_method       gaussian"
        echo "smearing_sigma        0.01"
        echo "mixing_type           broyden"
        echo "mixing_beta           0.4"
        echo "ks_solver             dav_subspace"
        echo "symmetry              0"
        echo "dft_plus_u            1"
        echo "orbital_corr          -1 2"
        echo "hubbard_u             0 5.0"
        echo "onsite_radius         3.0"
        # Branch: with a restart directory this case re-enters an already
        # converged density (stationarity test of that basin); without one the
        # SCF starts from the atomic guess.
        if [ -n "$restart" ]; then
            echo "init_chg              file"
            echo "read_file_dir         $restart"
        fi
        echo "pseudo_dir            $PPORB"
        echo "orbital_dir           $PPORB"
        echo "pw_seed               1"
    } > "$work/INPUT"

    echo "===== [$name] kmesh=\"$kmesh\" mag=$mag2/$mag3 restart=${restart:-none}"
    ( cd "$work" && $MPIRUN -np "$NPROC" "$ABACUS" > run.log 2>&1 )
    local rc=$?
    local log="$work/OUT.autotest/running_scf.log"
    : > "$RESDIR/$name.txt"
    # Branch: parse the SCF log only when ABACUS produced one -- a hard crash
    # leaves no log and the exit code alone is the record.
    if [ -f "$log" ]; then
        grep -E "!FINAL_ETOT_IS|#SCF IS CONVERGED#" "$log" | tail -2 >> "$RESDIR/$name.txt"
        grep -E "Total magnetism \(Bohr|Absolute magnetism \(Bohr" "$log" | tail -2 >> "$RESDIR/$name.txt"
        grep -E "atomic mag" "$log" | tail -2 >> "$RESDIR/$name.txt"
        awk '/^ *[A-Z]{2}[0-9]+ /{n=$1;e=$4;d=$6} END{printf " last_iter=%s etot=%s drho=%s\n", n, e, d}' \
            "$work/run.log" >> "$RESDIR/$name.txt"
    fi
    echo "----- [$name] exit=$rc"
    sed 's/^/    /' "$RESDIR/$name.txt"
}

# Group 1: Gamma-only cold starts from different magnetic guesses.  If a cold
# start can land in the lower basin, the harness default is simply an unlucky
# initial guess; if none can, the lower basin needs a perturbation.
run_case T1_gamma_cold_mag2   "1 1 1" 2.0 -2.0
run_case T2_gamma_cold_mag4   "1 1 1" 4.0 -4.0
run_case T3_gamma_cold_mag1   "1 1 1" 1.0 -1.0
run_case T4_gamma_cold_mag0p2 "1 1 1" 0.2 -0.2

# Group 2: warm re-entry into each known basin (stationarity check).  The
# reference density is /tmp/feo50/S0B (harness state), the lower one
# /tmp/feo50/W2.
# The lower basin needs its own out_chg 1 donor: the /tmp/feo50/W2 directory
# was written with out_chg 0, so it holds the charge density but no DFT+U
# onsite.dm, and Plus_U::read_occup_m aborts on re-entry.  T6D builds the
# donor (cold, mag 4.0 -- the guess that reaches the lower basin), T6 then
# re-enters it unconstrained.
OUTCHG=1 run_case T6D_donor_low_outchg "1 1 1" 4.0 -4.0
OUTCHG=0
run_case T6_gamma_warm_low "1 1 1" 2.0 -2.0 "$WORKROOT/T6D_donor_low_outchg/OUT.autotest"

# Group 3: dense k mesh, cold.  A denser mesh is the standard way to quench
# DFT+U multiple minima; if the two states merge here, the Gamma-only
# multistability is a k-sampling artifact rather than a property of the cell.
run_case T7_k222_cold_mag2 "2 2 2" 2.0 -2.0
run_case T8_k222_cold_mag4 "2 2 2" 4.0 -4.0

# Group 4: k-mesh convergence.  If the 2x2x2 state is the converged one, the
# 4x4x4 total energy must agree with it to ~10 meV; a further multi-eV drop
# would mean neither mesh is usable as the II-1 anchor.
run_case T9_k444_cold_mag2 "4 4 4" 2.0 -2.0
