#!/bin/bash
# =====================================================================
# P15 极化率张量各向异性（H2O/NH3/CO, 分量级验证）
# 每分子 × gdir=1/2/3: efield FD (±0.001) 与 DeltaP ±λ (±0.02, 含 λ=0 点)
# 双通道得 α_xx/α_yy/α_zz, 比较逐分量与各向异性比 α_∥/α_⊥ (∥=z)
# 阻塞: P01 (F1/F2) —— 判定前打印 WARNING, 换算系数集中于常量区
# 用法: bash run.sh  (env: ABACUS PSEUDO_DIR ORBITAL_DIR NPROC)
# =====================================================================
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

# ---------------- 换算常量区（待 F1/F2 备忘录定稿后单点修改） ----------------
PI=3.141592653589793
F2_SPIN_FACTOR=0.5            # γ↔μ 自旋因子 (nspin=1 取 1/2)，冒烟实测 H2O 确认 (2026-07-30)，待备忘录定稿
# α = -(a/π)·dγ/dE, E = -πλ/(2a) ⇒ α = (2a²/π²)·dγ/dλ
alpha_coef() { awk -v a="$1" -v pi="$PI" -v f="$F2_SPIN_FACTOR" 'BEGIN{printf "%.10e", 2*a*a/(pi*pi)*f}'; }
# efield 3 点 FD: α(a.u.) = -[E(+δ)+E(−δ)−2E(0)]/(2δ²)（E 单位 Ry, δ 单位 Ha）
DELTA_E=0.001
TOL_COMP=0.10                 # 逐分量两通道差
TOL_RATIO=0.05                # 各向异性比两通道差
R2_MIN=0.99                   # 窗口斜率线性
LAMBDAS="-0.02 0.0 0.02"      # 含 λ=0 点以使 R² 有意义
declare -A AB=( [h2o]=30.0 [nh3]=22.6767135 [co]=22.6767135 )
MOLS="h2o nh3 co"
DIRS="1 2 3"
# ---------------------------------------------------------------------------

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"
mkdir -p "$RUNS"
: > "$RESULTS"

NPASS=0; NTOTAL=0
report() {
    echo "$1: $2  $3" | tee -a "$RESULTS"
    case "$2" in
        PASS) NPASS=$((NPASS+1)); NTOTAL=$((NTOTAL+1));;
        FAIL) NTOTAL=$((NTOTAL+1));;
    esac
}

run_abacus() {
    rm -rf "$1"/OUT.*
    if [ "$NPROC" -gt 1 ]; then
        (cd "$1" && mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1)
    else
        (cd "$1" && "$ABACUS" > run.log 2>&1)
    fi
    if grep -q "SCF IS NOT CONVERGED" "$1/run.log"; then return 1; fi
    return 0
}

get_rawg() { grep "rawG" "$1/run.log" 2>/dev/null | tail -1 | sed -E 's/.*Σγ_raw=([^ ]+).*/\1/'; }
get_eks()  { grep "E_KohnSham" "$1"/OUT.*/running_scf.log 2>/dev/null | tail -1 | awk '{print $2}'; }

linfit() { # stdin "x y" → "slope intercept R2"
awk '
{ x[NR]=$1; y[NR]=$2; sx+=$1; sy+=$2; sxx+=$1*$1; sxy+=$1*$2; syy+=$2*$2; n=NR }
END {
    denom = n*sxx - sx*sx
    slope = (n*sxy - sx*sy)/denom
    icept = (sy - slope*sx)/n
    sst = syy - sy*sy/n
    ssr = 0
    for (i=1;i<=n;i++){ r=y[i]-(slope*x[i]+icept); ssr+=r*r }
    r2 = (sst>0) ? 1-ssr/sst : 1
    printf "%.10e %.10e %.6f", slope, icept, r2
}'
}

echo "==================== P15 极化率张量各向异性 ===================="
echo "ABACUS=$ABACUS  NPROC=$NPROC"

declare -A AREF ADP R2V
for mol in $MOLS; do
    acoef=$(alpha_coef "${AB[$mol]}")
    for gd in $DIRS; do
        echo "---- [$mol gdir=$gd] ----"
        # ---- 通道①: efield FD (±0.001, 0) ----
        declare -A EE=()
        for ea in -0.001 0.0 0.001; do
            tag=$(echo "$ea" | sed 's/-/m/;s/\./p/')
            d="${RUNS}/${mol}_d${gd}_ef_${tag}"
            mkdir -p "$d"
            cp "${CASES}/${mol}/STRU" "$d/"
            mesh="1 1 2"; [ "$gd" = "1" ] && mesh="2 1 1"; [ "$gd" = "2" ] && mesh="1 2 1"
            printf 'K_POINTS\n0\nGamma\n%s 0 0 0\n' "$mesh" > "$d/KPT"
            sed -e "s|@EFIELD_AMP@|$ea|" -e "s|@EFIELD_DIR@|$gd|" \
                -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" -e "s|@ORBITAL_DIR@|$ORBITAL_DIR|" \
                "${CASES}/INPUT_efield.tmpl" > "$d/INPUT"
            if run_abacus "$d"; then
                EE[$ea]=$(get_eks "$d")
                echo "  efield=$ea  E_KS=${EE[$ea]} Ry" | tee -a "$RESULTS"
            else
                report "$mol/d$gd efield=$ea" FAIL "SCF 未收敛"; EE[$ea]="NA"
            fi
        done
        if [ "${EE[-0.001]}" != "NA" ] && [ "${EE[0.0]}" != "NA" ] && [ "${EE[0.001]}" != "NA" ]; then
            a_ref=$(awk -v ep="${EE[0.001]}" -v em="${EE[-0.001]}" -v e0="${EE[0.0]}" -v de="$DELTA_E" \
                'BEGIN{printf "%.6f", -(ep+em-2*e0)/(2*de*de)}')
        else
            a_ref="NA"
        fi
        unset EE; declare -A EE=()
        # ---- 通道②: DeltaP ±λ (含 λ=0) ----
        XY=""
        for lam in $LAMBDAS; do
            tag=$(echo "$lam" | sed 's/-/m/;s/\./p/')
            d="${RUNS}/${mol}_d${gd}_lam_${tag}"
            mkdir -p "$d"
            cp "${CASES}/${mol}/STRU" "$d/"
            mesh="1 1 2"; [ "$gd" = "1" ] && mesh="2 1 1"; [ "$gd" = "2" ] && mesh="1 2 1"
            printf 'K_POINTS\n0\nGamma\n%s 0 0 0\n' "$mesh" > "$d/KPT"
            mb=$(awk -v l="$lam" 'BEGIN{print (l<0)?"0.3":"0.4"}')
            sed -e "s|@LAMBDA_INIT@|$lam|" -e "s|@MIXING_BETA@|$mb|" -e "s|@GDIR@|$gd|" \
                -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" -e "s|@ORBITAL_DIR@|$ORBITAL_DIR|" \
                "${CASES}/INPUT_deltap.tmpl" > "$d/INPUT"
            if run_abacus "$d"; then
                g=$(get_rawg "$d")
                echo "  λ=$lam  Σγ_raw=$g" | tee -a "$RESULTS"
                [ -n "$g" ] && XY="${XY}${lam} ${g}\n"
            else
                report "$mol/d$gd λ=$lam" FAIL "SCF 未收敛"
            fi
        done
        FIT=$(printf "$XY" | linfit)
        SLOPE=$(echo "$FIT" | awk '{print $1}')
        R2=$(echo "$FIT" | awk '{print $3}')
        a_dp=$(awk -v s="$SLOPE" -v c="$acoef" 'BEGIN{printf "%.6f", c*s}')
        echo "  [$mol/d$gd] α_ref=$a_ref  α_DeltaP=$a_dp  (R2=$R2)" | tee -a "$RESULTS"
        AREF[$mol,$gd]="$a_ref"; ADP[$mol,$gd]="$a_dp"; R2V[$mol,$gd]="$R2"
    done
done

# ================= 判定 =================
echo "---- [判定] ----"
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README，前置 P01）" | tee -a "$RESULTS"

# 判据1+4: 逐分量两通道差 ≤10%; 窗口斜率 R² ≥ 0.99
for mol in $MOLS; do
    for gd in $DIRS; do
        r="${AREF[$mol,$gd]}"; p="${ADP[$mol,$gd]}"; r2="${R2V[$mol,$gd]}"
        if [ "$r" = "NA" ] || [ -z "$r" ]; then
            report "$mol/d$gd 分量一致" FAIL "缺数据"; continue
        fi
        ok=$(awk -v a="$r" -v b="$p" -v t="$TOL_COMP" \
            'BEGIN{d=a-b; if(d<0)d=-d; base=(a<0)?-a:a; print (base>0 && d/base<=t)?1:0}')
        [ "$ok" = "1" ] && report "$mol/d$gd 分量一致" PASS "α_ref=$r α_DeltaP=$p" \
                         || report "$mol/d$gd 分量一致" FAIL "α_ref=$r α_DeltaP=$p"
        ok=$(awk -v r="$r2" -v m="$R2_MIN" 'BEGIN{print (r>=m)?1:0}')
        [ "$ok" = "1" ] && report "$mol/d$gd 窗口线性" PASS "R2=$r2 >= $R2_MIN" \
                         || report "$mol/d$gd 窗口线性" FAIL "R2=$r2 < $R2_MIN"
    done
done

# 判据2: 各向异性比 α_∥/α_⊥ (∥=z, ⊥=(x+y)/2) 两通道差 ≤5%
for mol in $MOLS; do
    r_par="${AREF[$mol,3]}"; r_perp=$(awk -v x="${AREF[$mol,1]}" -v y="${AREF[$mol,2]}" 'BEGIN{printf "%.6f",(x+y)/2}')
    p_par="${ADP[$mol,3]}";  p_perp=$(awk -v x="${ADP[$mol,1]}" -v y="${ADP[$mol,2]}" 'BEGIN{printf "%.6f",(x+y)/2}')
    [ "$r_par" = "NA" ] && { report "$mol 各向异性比" FAIL "缺数据"; continue; }
    ratio_r=$(awk -v a="$r_par" -v b="$r_perp" 'BEGIN{printf "%.6f", a/b}')
    ratio_p=$(awk -v a="$p_par" -v b="$p_perp" 'BEGIN{printf "%.6f", a/b}')
    echo "  $mol 各向异性比: ref=$ratio_r  DeltaP=$ratio_p" | tee -a "$RESULTS"
    ok=$(awk -v a="$ratio_r" -v b="$ratio_p" -v t="$TOL_RATIO" \
        'BEGIN{d=a-b; if(d<0)d=-d; print (d/a<=t)?1:0}')
    [ "$ok" = "1" ] && report "$mol 各向异性比" PASS "ref=$ratio_r DeltaP=$ratio_p" \
                     || report "$mol 各向异性比" FAIL "ref=$ratio_r DeltaP=$ratio_p"
done

# 判据3: CO α_∥ > α_⊥（两通道各自）
for ch in AREF ADP; do
    if [ "$ch" = "AREF" ]; then par="${AREF[co,3]}"; px="${AREF[co,1]}"; py="${AREF[co,2]}"; name="efield"
    else par="${ADP[co,3]}"; px="${ADP[co,1]}"; py="${ADP[co,2]}"; name="DeltaP"; fi
    [ "$par" = "NA" ] && { report "CO ∥>⊥ ($name)" FAIL "缺数据"; continue; }
    ok=$(awk -v a="$par" -v x="$px" -v y="$py" 'BEGIN{print (a>x && a>y)?1:0}')
    [ "$ok" = "1" ] && report "CO ∥>⊥ ($name)" PASS "α_zz=$par > α_xx=$px, α_yy=$py" \
                     || report "CO ∥>⊥ ($name)" FAIL "α_zz=$par, α_xx=$px, α_yy=$py"
done

echo "SUMMARY: ${NPASS}/${NTOTAL} PASS" | tee -a "$RESULTS"
[ "$NPASS" -eq "$NTOTAL" ] && exit 0 || exit 1
