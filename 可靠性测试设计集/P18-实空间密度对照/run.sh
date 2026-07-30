#!/bin/bash
# P18 实空间 Delta rho(r) 对照：lambda=0.02 vs 等效 E 的密度指纹（30 Bohr H2O）
# 用法: bash run.sh   （可用环境变量覆盖 ABACUS / PSEUDO_DIR / ORBITAL_DIR / NPROC）
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="$BASEDIR/cases"
RUNS="$BASEDIR/runs"
mkdir -p "$RUNS"
RESULTS="$RUNS/results.txt"

# ---- F1/F2 换算常量（备忘录定稿前工作值，修改集中于此） ----
PI=3.141592653589793
A_BOHR=30.0                       # gdir 方向盒边长 (Bohr)
LAM_DP=0.02                       # DeltaP 扰动 lambda (Ry)
# F1: E(Ha/Bohr) = -PI*lambda/(2*A_BOHR)
# F2: mu = (A_BOHR/PI)*gamma_raw  (归一化用 Dmu 标定)

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }
mixb_for() { awk -v l="$1" 'BEGIN{print (l+0<0)?"0.3":"0.4"}'; }
E_PERTURB="$(awk -v l="$LAM_DP" -v pi="$PI" -v a="$A_BOHR" 'BEGIN{printf "%.8f", -pi*l/(2*a)}')"

run_abacus() {
    local d="$1"
    cd "$d"
    rm -rf OUT.*
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 || echo "WARNING: abacus exit nonzero in $d"
    else
        "$ABACUS" > run.log 2>&1 || echo "WARNING: abacus exit nonzero in $d"
    fi
    cd "$BASEDIR"
}

gen_input() { # $1=模板 $2=输出 $3=lambda $4=mixing_beta $5=efield_amp
    sed -e "s#@PSEUDO_DIR@#$PSEUDO_DIR#g" \
        -e "s#@ORBITAL_DIR@#$ORBITAL_DIR#g" \
        -e "s#@LAMBDA@#$3#g" \
        -e "s#@MIXB@#$4#g" \
        -e "s#@EFIELD@#$5#g" "$1" > "$2"
}

get_rawg()     { grep -o 'Σγ_raw=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
get_pw_gamma() { grep -o 'γ_total=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
is_conv()      { if grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; then echo INVALID; else echo OK; fi; }
find_cube()    { ls "$1"/OUT.*/chg.cube 2>/dev/null | head -1 || true; }

echo "===== P18 实空间密度对照 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC  E(扰动)=$E_PERTURB Ha/Bohr"

# ---- 四次 LCAO SCF (out_chg 1, scf_thr 1e-9) ----
for SPEC in "lam_0 0.0 0.0 deltap" "lam_p$(tag "$LAM_DP") $LAM_DP 0.0 deltap" "ef_0 0.0 0.0 efield" "ef_pert 0.0 $E_PERTURB efield"; do
    read -r NAME LAM EAMP KIND <<< "$SPEC"
    d="$RUNS/$NAME"
    mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.$KIND.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" "$EAMP"
    run_abacus "$d"
    echo "  [$NAME] SCF=$(is_conv "$d") gamma=$(get_rawg "$d") cube=$(find_cube "$d")"
done

# ---- 可选: PW 同 lambda 密度对照 ----
PW_OK=1
for SPEC in "pw_0 0.0" "pw_p$(tag "$LAM_DP") $LAM_DP"; do
    read -r NAME LAM <<< "$SPEC"
    d="$RUNS/$NAME"
    mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.pw.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" 0.0
    run_abacus "$d"
    C="$(find_cube "$d")"
    echo "  [$NAME] SCF=$(is_conv "$d") gamma=$(get_pw_gamma "$d") cube=$C"
    [ -n "$C" ] || PW_OK=0
done

C_L0="$(find_cube "$RUNS/lam_0")";   C_LP="$(find_cube "$RUNS/lam_p$(tag "$LAM_DP")")"
C_E0="$(find_cube "$RUNS/ef_0")";    C_EP="$(find_cube "$RUNS/ef_pert")"

# ---- 归一化因子: Dmu_lam / Dmu_ef ----
G_L0="$(get_rawg "$RUNS/lam_0")"; G_LP="$(get_rawg "$RUNS/lam_p$(tag "$LAM_DP")")"
G_E0="$(get_rawg "$RUNS/ef_0")";  G_EP="$(get_rawg "$RUNS/ef_pert")"
SCALE="$(awk -v lp="$G_LP" -v l0="$G_L0" -v ep="$G_EP" -v e0="$G_E0" \
    'BEGIN{dl=lp-l0; de=ep-e0; if(de==0||dl==""||de==""){print "NAN"}else printf "%.8f", dl/de}')"
# 注: Dmu=(a/pi)*Dgamma, 比值中 a/pi 约去, SCALE = Dgamma_lam/Dgamma_ef

analyze() { # $1..$4 = base_A pert_A base_B pert_B  -> "pearson rms"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$1" "$2" "$3" "$4" "$SCALE" << 'PYEOF'
import sys
def zprof(fn):
    with open(fn) as f:
        f.readline(); f.readline()
        nat = abs(int(f.readline().split()[0]))
        nx = int(f.readline().split()[0]); ny = int(f.readline().split()[0]); nz = int(f.readline().split()[0])
        for _ in range(nat): f.readline()
        vals = [float(x) for tok in f.read().split() for x in [tok]]
    prof = [0.0]*nz
    for i, v in enumerate(vals[:nx*ny*nz]):
        prof[i//(nx*ny)] += v/(nx*ny)
    return prof
bA, pA, bB, pB, s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
dA = [x-y for x, y in zip(zprof(pA), zprof(bA))]
dB = [s*(x-y) for x, y in zip(zprof(pB), zprof(bB))]
n = len(dA)
mA = sum(dA)/n; mB = sum(dB)/n
cov = sum((x-mA)*(y-mB) for x, y in zip(dA, dB))
vA = sum((x-mA)**2 for x in dA); vB = sum((y-mB)**2 for y in dB)
r = cov/((vA*vB)**0.5) if vA*vB > 0 else float('nan')
rms = (sum((x-y)**2 for x, y in zip(dA, dB))/n)**0.5
ref = (sum(x*x for x in dA)/n)**0.5
print(f"{r:.6f} {rms/ref if ref>0 else float('nan'):.6f}")
PYEOF
    else
        awk -v s="$SCALE" -v f1="$1" -v f2="$2" -v f3="$3" -v f4="$4" '
        FNR==1 || FNR==2 { next }
        FNR==3 { nat=($1<0?-$1:$1); next }
        FNR==4 { nx=$1; next }
        FNR==5 { ny=$1; next }
        FNR==6 { nz=$1; cnt=0; next }
        FNR<=6+nat { next }
        {
          for(j=1;j<=NF;j++){ iz=int(cnt/(nx*ny)); prof[cur,iz]+=$j/(nx*ny); cnt++ }
        }
        END{
          for(iz=0; iz<nz; iz++){
            da[iz]=prof[f2,iz]-prof[f1,iz];
            db[iz]=s*(prof[f4,iz]-prof[f3,iz]);
            sa+=da[iz]; sb+=db[iz];
          }
          ma=sa/nz; mb=sb/nz; cov=0; va=0; vb=0; rr=0; ra=0;
          for(iz=0; iz<nz; iz++){
            x=da[iz]-ma; y=db[iz]-mb; cov+=x*y; va+=x*x; vb+=y*y;
            rr+=(da[iz]-db[iz])**2; ra+=da[iz]*da[iz];
          }
          r=(va*vb>0)? cov/sqrt(va*vb) : "NAN";
          rms=(ra>0)? sqrt(rr/nz)/sqrt(ra/nz) : "NAN";
          printf "%.6f %.6f", r, rms
        }' "$1" "$2" "$3" "$4"
    fi
}

{
echo ""
echo "----- 归一化 -----"
echo "  Dgamma_lam = $G_LP - $G_L0 ; Dgamma_ef = $G_EP - $G_E0"
echo "  SCALE = Dgamma_lam/Dgamma_ef = $SCALE  (按 Dmu 标定, a/pi 约去)"

echo ""
echo "----- LCAO: Drho_lambda vs Drho_efield (z 轴平面平均剖面) -----"
if [ -n "$C_L0" ] && [ -n "$C_LP" ] && [ -n "$C_E0" ] && [ -n "$C_EP" ] && [ "$SCALE" != "NAN" ]; then
    read -r PEAR RMS <<< "$(analyze "$C_L0" "$C_LP" "$C_E0" "$C_EP")"
    echo "  Pearson r = $PEAR (判据 >= 0.99)"
    echo "  RMS 残差  = $RMS  (判据 <= 0.05)"
else
    PEAR=NAN; RMS=NAN
    echo "  SKIP: 密度文件或归一化因子缺失 (cubes: $C_L0 $C_LP $C_E0 $C_EP)"
fi

echo ""
echo "----- 可选通道: LCAO vs PW 同 lambda 密度对照 -----"
C_P0="$(find_cube "$RUNS/pw_0")"; C_PP="$(find_cube "$RUNS/pw_p$(tag "$LAM_DP")")"
if [ "$PW_OK" = "1" ] && [ -n "$C_L0" ] && [ -n "$C_LP" ]; then
    G_P0="$(get_pw_gamma "$RUNS/pw_0")"; G_PP="$(get_pw_gamma "$RUNS/pw_p$(tag "$LAM_DP")")"
    SCALE_PW="$(awk -v pp="$G_PP" -v p0="$G_P0" -v lp="$G_LP" -v l0="$G_L0" -v pi="$PI" '
        BEGIN{dp=pp-p0; if(dp>pi)dp-=2*pi; else if(dp<-pi)dp+=2*pi; dl=lp-l0;
              if(dp==0){print "NAN"}else printf "%.8f", dl/dp}')"
    echo "  SCALE_PW = Dgamma_lcao/Dgamma_pw(unwrapped) = $SCALE_PW"
    SAVE_SCALE="$SCALE"; SCALE="$SCALE_PW"
    read -r PEAR2 RMS2 <<< "$(analyze "$C_L0" "$C_LP" "$C_P0" "$C_PP")"
    SCALE="$SAVE_SCALE"
    echo "  Pearson r = $PEAR2 ; RMS 残差 = $RMS2"
else
    echo "  SKIP: PW 密度缺失，通道跳过"
fi

echo ""
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）——以下判据仅记录数值不参与 SUMMARY"
echo "[J1] LCAO Drho_lam vs Drho_ef: Pearson r>=0.99 (实测 $PEAR), RMS<=5% (实测 $RMS)"
echo "[J2] LCAO vs PW 同 lambda:   Pearson/RMS 见上（可选通道）"

echo ""
echo "SUMMARY: 0/0 PASS (全部判据阻塞于 F1/F2, 见 WARNING)"
} | tee "$RESULTS"

exit 0
