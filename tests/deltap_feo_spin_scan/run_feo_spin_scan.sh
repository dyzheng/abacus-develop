#!/bin/bash
# II-1 FeO spin-constraint scan
# (plan: docs/superpowers/plans/2026-09-09-p0-cases-feomgo.md, case II-1).
#
# Verification questions:
#   1. can the framework constrain a TM d local moment at production quality?
#   2. is DFT+U + constraint simultaneous activation compatible?  (Two
#      independent potential channels: the on-site DFT+U projection and the
#      veff-grid constraint injection had never been run in the same job.)
#   3. (deferred to II-1b) quantitative mu vs DeltaSpin lambda attribution.
#
# SYSTEM AMENDMENT vs the plan: the plan named
# tests/17_DS_DFTU/11_PW_DFTU_S2_FeO as the inherited baseline, but that case's
# STRU holds only the Fe sublattice of the rocksalt cell (TOTAL ATOM NUMBER = 2,
# species "Fe" only) -- it is the Fe sublattice, not FeO, despite the directory
# name.  The real FeO case is tests/17_DS_DFTU/50_FeO_O_first_Fe_second
# (2 O + 2 Fe, rocksalt primitive cell, ecutwfc 50, orbital_corr -1 2).  This
# scan inherits case 50.  Atom indexing there: 0,1 = O; 2,3 = Fe (2 carries
# mag +2, 3 carries mag -2).
#
# Steps (PW / dav_subspace / nspin 2 / DFT+U / symmetry 0 / Gamma-only):
#   S1   baseline reproduction, constraint off, at the harness scf_thr 1e-6
#   S1X  threshold ladder: (a) baseline at scf_thr 1e-8, (b) constraint on at
#        scf_thr 1e-8, (c) constraint on at scf_thr 1e-7 -> which setting gives
#        an SCF that actually reaches drho < scf_thr?
#   S2   reference phase + first constrained point, delta = +0.1 uB
#   S2R  delta = 0 no-op check (the reference phase must already satisfy it)
#   S3   scan delta = +-0.1 / +-0.3 / +-0.5 uB on Fe atom 2 (cold start)
#   S4   fuse case delta = +3.0 uB (beyond the atomic limit -> UNREACHABLE)
#   S5   DeltaSpin control (case 12 settings) at the matched state
#   S6   summary table: mu(delta), kappa, on-site moment response
#
# Usage: bash run_feo_spin_scan.sh <S1|S1X|S2|S2R|S3|S4|S5|S6> [nproc]
set -uo pipefail

STEP="${1:?usage: run_feo_spin_scan.sh <S1|S1X|S2|S2R|S3|S4|S5|S6> [nproc]}"
NPROC="${2:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build_rel/abacus_basic_para}"
MPIRUN="${MPIRUN:-mpirun}"
CASEDIR="$(cd "$(dirname "$0")" && pwd)"
PPORB="$(cd "$CASEDIR/.." && pwd)/PP_ORB"
WORKROOT="${WORKROOT:-/tmp/feo_spin}"
RESDIR="$CASEDIR/results"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

ECUT="${ECUT:-50}"
SCF_THR="${SCF_THR:-1e-7}"
SCF_NMAX="${SCF_NMAX:-400}"
MIXB="${MIXB:-0.4}"
TARGET_ATOM="${TARGET_ATOM:-2}"
# Warm start (the plan's hot start): a converged unconstrained FeO density +
# onsite.dm.  Needed because AFM FeO has several low-lying magnetic solutions
# and a cold start (atomic charge) picks whichever branch the SCF wanders into,
# not the one adiabatically connected to the reference state.
RESTART="${RESTART:-}"
# Initial magnetic guess override (the S1T triage showed the SCF basin is
# selected by the STRU mag value: mag 2 -> the metastable result.ref state,
# mag 4 -> the lower one).  Empty = use the case STRU verbatim.
MAG2="${MAG2:-}"; MAG3="${MAG3:-}"
# Outer-step caps (II-1 framework hardening).  The history-free first step
# otherwise always sits at the step_max cap (kappa falls back to kappa_min),
# which on FeO flips the magnetic branch; the probe lets the secant measure a
# local slope first.
STEP_MAX="${STEP_MAX:-0.05}"; STEP_PROBE="${STEP_PROBE:-0.0}"

mkdir -p "$WORKROOT" "$RESDIR"

# Audit facts for one finished case: CONVERGED / UNREACHABLE / RUNNING plus the
# first and last constraint observation.
audit()
{
    local log="$WORKROOT/$1/OUT.autotest/running_scf.log"
    grep -E "CONSTRAINT_AUDIT|final status" "$log" 2>/dev/null
}

# run_case <name> <target-json|empty>
# Empty target-json => plain SCF, no constraint block at all.
run_case()
{
    local name="$1" json="$2"
    local work="$WORKROOT/$name"
    python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$work"
    mkdir -p "$work"
    cp "$CASEDIR/KPT" "$work/"
    # Branch: with MAG2/MAG3 set the Fe starting guesses are rewritten;
    # without them the inherited STRU is used unchanged.
    if [ -n "$MAG2" ]; then
        python3 "$CASEDIR/tools/make_stru_mag.py" "$CASEDIR/STRU" "$work/STRU" "$MAG2" "$MAG3"
    else
        cp "$CASEDIR/STRU" "$work/"
    fi
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
        echo "out_chg               ${OUTCHG:-0}"
        echo "smearing_method       gaussian"
        echo "smearing_sigma        0.01"
        echo "mixing_type           broyden"
        echo "mixing_beta           $MIXB"
        echo "ks_solver             dav_subspace"
        echo "symmetry              0"
        echo "dft_plus_u            1"
        echo "orbital_corr          -1 2"
        echo "hubbard_u             0 5.0"
        echo "onsite_radius         3.0"
        if [ -n "$json" ]; then
            printf '%s\n' "$json" > "$work/constraint_target.json"
            echo "constraint            true"
            echo "constraint_type       spin"
            echo "constraint_weight_type becke"
            echo "constraint_target_file constraint_target.json"
            echo "constraint_target_mode delta"
            echo "constraint_mu_max     5.0"
            echo "constraint_thr        1e-4"
            echo "constraint_step_max   $STEP_MAX"
            echo "constraint_step_probe $STEP_PROBE"
        fi
        # Branch: warm start from a previous run's charge density + onsite.dm.
        if [ -n "$RESTART" ]; then
            echo "init_chg              file"
            echo "read_file_dir         $RESTART"
        fi
        echo "pseudo_dir            $PPORB"
        echo "orbital_dir           $PPORB"
        echo "pw_seed               1"
    } > "$work/INPUT"
    echo "===== [$name] ecutwfc=$ECUT scf_thr=$SCF_THR scf_nmax=$SCF_NMAX nproc=$NPROC work=$work"
    ( cd "$work" && $MPIRUN -np "$NPROC" "$ABACUS" > run.log 2>&1 )
    local rc=$?
    echo "----- [$name] exit=$rc"
    grep -E "CONSTRAINT_AUDIT|final status" "$work/OUT.autotest/running_scf.log" 2>/dev/null | tail -10
    grep -E "!FINAL_ETOT_IS|charge density convergence" "$work/OUT.autotest/running_scf.log" 2>/dev/null
    cp "$work/OUT.autotest/running_scf.log" "$RESDIR/$name.running_scf.log" 2>/dev/null
    return $rc
}

case "$STEP" in
S0B)
    # Warm-start donor: unconstrained FeO at scf_thr 1e-8 with out_chg 1, which
    # is what makes ABACUS write the DFT+U onsite.dm next to the charge density.
    # Every constrained point of S3 is warm started from this directory, so the
    # reference phase of each point is a genuine mu = 0 observation of the same
    # AFM branch.
    SCF_THR=1e-8 SCF_NMAX=200 OUTCHG=1 run_case "S0B_warmstart" ""
    ;;
S1)
    # Baseline reproduction at the harness threshold: must match result.ref of
    # the inherited case (reproducibility gate before any constraint run).
    SCF_THR=1e-6 run_case "S1_base_1e6" ""
    ;;
S1X)
    # Threshold ladder.  The audit observable Q is read off the converged
    # density, so an SCF that plateaus above scf_thr writes its own noise floor
    # into the residual: the plan's 1e-4 reading precision is only meaningful
    # when the SCF actually reaches drho < scf_thr.
    JSON='{"constraints": [{"type": "spin", "target": 0.1, "atoms": [2]}]}'
    SCF_THR=1e-8 run_case "S1x_base_1e8" ""
    SCF_THR=1e-8 run_case "S1x_cstr_1e8" "$JSON"
    SCF_THR=1e-7 run_case "S1x_cstr_1e7" "$JSON"
    SCF_THR=1e-7 MIXB=0.2 run_case "S1x_cstr_1e7_mix02" "$JSON"
    ;;
S2)
    JSON='{"constraints": [{"type": "spin", "target": 0.1, "atoms": [2]}]}'
    run_case "S2_fe2_p01" "$JSON"
    ;;
S2R)
    JSON='{"constraints": [{"type": "spin", "target": 0.0, "atoms": [2]}]}'
    run_case "S2R_ref_delta0" "$JSON"
    ;;
S3)
    # Six-point scan on Fe atom 2.  Cold start everywhere (not the plan's hot
    # start): the reference phase of every point IS the mu = 0 observation, so a
    # cold start makes the six Q_ref readings six independent measurements of
    # the same reference state -- a free consistency check on the anchor that a
    # hot start from an already-constrained density would destroy.
    for d in ${DELTAS:-0.1 0.3 0.5 -0.1 -0.3 -0.5}; do
        tag=$(echo "$d" | sed 's/^+//; s/-/m/; s/\./p/')
        JSON="{\"constraints\": [{\"type\": \"spin\", \"target\": $d, \"atoms\": [$TARGET_ATOM]}]}"
        run_case "S3_fe2_${tag}" "$JSON"
        echo "-- delta=$d"; audit "S3_fe2_${tag}"
    done
    ;;
S3L)
    # Re-anchored six-point scan on the LOWER Gamma-only branch (S1T triage
    # verdict; see run_feo_baseline_triage.sh).  The inherited result.ref
    # state (Fe +-3.4850 uB, E = -7652.3958757 eV) is the metastable one, so
    # the scan anchors on the lower solution (Fe +-3.7148 uB,
    # E = -7653.0079658 eV), which the mag 4.0 cold start reaches.
    #
    # Hot start in delta order within each side: 0 -> +0.1 -> +0.3 -> +0.5 and
    # 0 -> -0.1 -> -0.3 -> -0.5, every point warm started from the previous
    # point's OUT.autotest, so the SCF always begins from a density already on
    # this branch (adiabatic continuation in delta).  The two sides restart
    # from the donor instead of chaining through each other, which would need
    # one unphysical -0.6 uB jump.  out_chg 1 is what makes ABACUS write the
    # DFT+U onsite.dm that Plus_U::read_occup_m needs on re-entry.
    POS_DELTAS="${POS_DELTAS-0.1 0.3 0.5}"
    NEG_DELTAS="${NEG_DELTAS--0.1 -0.3 -0.5}"
    MAG2=4.0; MAG3=-4.0; OUTCHG=1
    SCF_THR="${SCF_THR:-1e-7}"; SCF_NMAX="${SCF_NMAX:-400}"
    run_case "S3L_donor" ""
    for side in pos neg; do
        prev="$WORKROOT/S3L_donor/OUT.autotest"
        # Branch: pos walks the moment up, neg walks it down; both chains
        # leave the donor, so neither inherits the other side's displacement.
        if [ "$side" = "pos" ]; then
            deltas="$POS_DELTAS"
        else
            deltas="$NEG_DELTAS"
        fi
        for d in $deltas; do
            tag=$(echo "$d" | sed 's/^+//; s/-/m/; s/\./p/')
            JSON="{\"constraints\": [{\"type\": \"spin\", \"target\": $d, \"atoms\": [$TARGET_ATOM]}]}"
            RESTART="$prev" run_case "S3L_fe2_${tag}" "$JSON"
            echo "-- delta=$d"; audit "S3L_fe2_${tag}"
            prev="$WORKROOT/S3L_fe2_${tag}/OUT.autotest"
        done
    done
    ;;
S4)
    # Fuse case: +3.0 uB on one Fe is beyond the atomic limit, so the outer loop
    # must report UNREACHABLE at the mu cap and print the Q(mu) endpoint without
    # diverging.  scf_nmax is raised because step_max = 0.05 Ry caps one outer
    # step at |mu|/0.05 ~ 45 steps for a ~2.3 Ry response.
    JSON='{"constraints": [{"type": "spin", "target": 3.0, "atoms": [2]}]}'
    SCF_NMAX=3000 run_case "S4_fuse_p3" "$JSON"
    audit "S4_fuse_p3" | tail -4
    ;;
S5)
    # DeltaSpin control at the matched state: the targets are the on-site Fe
    # moments measured in the constrained delta=+0.1 run, so both codes are
    # compared at the same physical state and only the multiplier definition
    # (Becke-weighted veff multiplier mu vs on-site-projection multiplier
    # lambda) differs.
    M2="${M2:-3.54107616}"; M3="${M3:--3.52315803}"
    work="$WORKROOT/S5_deltaspin"
    python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$work"
    mkdir -p "$work"
    cp "$CASEDIR/KPT" "$work/"
    python3 "$CASEDIR/tools/make_deltaspin_stru.py" "$CASEDIR/STRU" "$work/STRU" "$M2" "$M3"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix                autotest"
        echo "calculation           scf"
        echo "basis_type            pw"
        echo "gamma_only            0"
        echo "ecutwfc               $ECUT"
        echo "nspin                 2"
        echo "scf_thr               1e-7"
        echo "scf_nmax              300"
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
        echo "sc_mag_switch         1"
        echo "sc_thr                1e-4"
        echo "nsc                   200"
        echo "nsc_min               2"
        echo "alpha_trial           0.01"
        echo "sccut                 3.0"
        echo "sc_scf_thr_mode       immediate"
        echo "pseudo_dir            $PPORB"
        echo "orbital_dir           $PPORB"
        echo "pw_seed               1"
    } > "$work/INPUT"
    echo "===== [S5_deltaspin] target atom2=$M2 atom3=$M3"
    ( cd "$work" && $MPIRUN -np "$NPROC" "$ABACUS" > run.log 2>&1 )
    echo "----- [S5_deltaspin] exit=$?"
    grep -iE "lambda|Orbital Charge|Total Magnetism|!FINAL_ETOT_IS" \
        "$work/OUT.autotest/running_scf.log" 2>/dev/null | tail -20
    cp "$work/OUT.autotest/running_scf.log" "$RESDIR/S5_deltaspin.running_scf.log" 2>/dev/null
    ;;
S6)
    python3 "$CASEDIR/tools/summarize.py" "$WORKROOT"
    ;;
*)
    echo "step $STEP not wired yet"; exit 2;;
esac
