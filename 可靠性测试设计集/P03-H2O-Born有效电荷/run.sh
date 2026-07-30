#!/bin/bash
# =====================================================================
# P03 H2O Born 有效电荷（dF/dλ → Z*）
# 通道①: DeltaP ±λ 扫描, 拟合 dF_O/dλ → Z*_O = (a/π)·dF_O/dλ
# 通道②: PW berry_phase 位移 FD (O 原子 z 向 ±0.005 Å), Z* = (a/π)·Δγ/Δr
# 阻塞: P01 (F1/F2 换算链) —— 判定段打印 WARNING, 换算系数集中于下方常量区
# 用法: bash run.sh  (env: ABACUS PSEUDO_DIR ORBITAL_DIR NPROC)
# =====================================================================
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

# ---------------- 换算常量区（待 F1/F2 备忘录定稿后单点修改） ----------------
PI=3.141592653589793
A_BOHR=30.0                  # H2O 盒子边长 (Bohr), gdir=3 方向
F1_PI_OVER_A=$(awk -v a="$A_BOHR" -v pi="$PI" 'BEGIN{printf "%.10e", a/pi}')   # a/π
EVA_TO_RYBOHR=0.0388938      # 1 eV/Angstrom = 0.0388938 Ry/Bohr（单位换算，待备忘录确认）
ANG_TO_BOHR=1.8897261254578284
F2_SPIN_FACTOR=1.0           # γ↔μ 自旋因子占位（nspin=1, 待备忘录定稿）
DISP_ANG=0.005               # 位移 FD 步长 (Angstrom)
ZSTAR_LO=-2.0                # 文献区间
ZSTAR_HI=-1.5
TOL_ZSTAR=0.05               # 通道互差 / 中和判据
R2_MIN=0.98                  # 力-λ 窗口线性判据
LAMBDAS="-0.01 -0.005 0.005 0.01"
# ---------------------------------------------------------------------------

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"
mkdir -p "$RUNS"
: > "$RESULTS"

NPASS=0; NTOTAL=0
report() { # $1=项目 $2=PASS/FAIL/SKIP/WARNING $3=说明
    echo "$1: $2  $3" | tee -a "$RESULTS"
    case "$2" in
        PASS) NPASS=$((NPASS+1)); NTOTAL=$((NTOTAL+1));;
        FAIL) NTOTAL=$((NTOTAL+1));;
    esac
}

run_abacus() { # $1=工作目录
    rm -rf "$1"/OUT.*
    if [ "$NPROC" -gt 1 ]; then
        (cd "$1" && mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1)
    else
        (cd "$1" && "$ABACUS" > run.log 2>&1)
    fi
    if grep -q "SCF IS NOT CONVERGED" "$1/run.log"; then return 1; fi
    return 0
}

get_fz() { # $1=rundir $2=原子标签(如 O1) → 最后 TOTAL-FORCE 块的 Fz (eV/Ang)
    local log; log=$(ls "$1"/OUT.*/running_scf.log 2>/dev/null | head -1)
    [ -n "$log" ] || { echo "NA"; return; }
    awk -v lbl="$2" '
        /TOTAL-FORCE/ {blk=1; delete F; next}
        blk && $1==lbl {F[$1]=$4}
        END {if (lbl in F) printf "%.10e", F[lbl]; else print "NA"}' "$log"
}

get_pw_gamma() { # $1=rundir → [DeltaP-PW] γ_total
    grep "DeltaP-PW" "$1/run.log" 2>/dev/null | grep "γ_total" | tail -1 \
        | sed -E 's/.*γ_total=([^ ]+).*/\1/'
}

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

echo "==================== P03 H2O Born 有效电荷 ===================="
echo "ABACUS=$ABACUS  NPROC=$NPROC"
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README，前置 P01）" | tee -a "$RESULTS"

# ================= 通道①: DeltaP ±λ 力扫描 =================
echo "---- [Stage 1] DeltaP λ 扫描: $LAMBDAS ----"
XY_O=""; XY_H=""
for lam in $LAMBDAS; do
    tag=$(echo "$lam" | sed 's/-/m/;s/\./p/')
    d="${RUNS}/lam_${tag}"
    mkdir -p "$d"
    cp "${CASES}/h2o/STRU" "${CASES}/h2o/KPT" "$d/"
    # 负 λ 更难收敛, mixing_beta 0.3; 正 λ 用 0.4
    mb=$(awk -v l="$lam" 'BEGIN{print (l<0)?"0.3":"0.4"}')
    sed -e "s|@LAMBDA_INIT@|$lam|" -e "s|@MIXING_BETA@|$mb|" \
        -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" -e "s|@ORBITAL_DIR@|$ORBITAL_DIR|" \
        "${CASES}/INPUT_lcao_force.tmpl" > "$d/INPUT"
    if run_abacus "$d"; then
        fo=$(get_fz "$d" O1); fh1=$(get_fz "$d" H1); fh2=$(get_fz "$d" H2)
        echo "λ=$lam  Fz_O=$fo  Fz_H1=$fh1  Fz_H2=$fh2 (eV/Ang)" | tee -a "$RESULTS"
        if [ "$fo" != "NA" ] && [ "$fh1" != "NA" ] && [ "$fh2" != "NA" ]; then
            XY_O="${XY_O}${lam} ${fo}\n"
            fh=$(awk -v a="$fh1" -v b="$fh2" 'BEGIN{printf "%.10e",(a+b)/2}')
            XY_H="${XY_H}${lam} ${fh}\n"
        else
            report "λ=$lam 力提取" FAIL "TOTAL-FORCE 未找到"
        fi
    else
        report "λ=$lam SCF" FAIL "未收敛"
    fi
done

FIT_O=$(printf "$XY_O" | linfit)
FIT_H=$(printf "$XY_H" | linfit)
SLOPE_O=$(echo "$FIT_O" | awk '{print $1}')
R2_O=$(echo "$FIT_O" | awk '{print $3}')
SLOPE_H=$(echo "$FIT_H" | awk '{print $1}')
R2_H=$(echo "$FIT_H" | awk '{print $3}')
echo "dF_O/dλ=$SLOPE_O (eV/Ang/Ry) R2=$R2_O ; dF_H/dλ=$SLOPE_H R2=$R2_H" | tee -a "$RESULTS"

# Z* = (a/π) · dF/dλ · (eV/Ang→Ry/Bohr) · F2
Z_O=$(awk -v s="$SLOPE_O" -v c="$F1_PI_OVER_A" -v u="$EVA_TO_RYBOHR" -v f="$F2_SPIN_FACTOR" 'BEGIN{printf "%.6f", c*s*u*f}')
Z_H=$(awk -v s="$SLOPE_H" -v c="$F1_PI_OVER_A" -v u="$EVA_TO_RYBOHR" -v f="$F2_SPIN_FACTOR" 'BEGIN{printf "%.6f", c*s*u*f}')
echo "通道①: Z*_O=$Z_O  Z*_H=$Z_H" | tee -a "$RESULTS"

ok=$(awk -v r="$R2_O" -v m="$R2_MIN" 'BEGIN{print (r>=m)?1:0}')
[ "$ok" = "1" ] && report "通道① O 力-λ 线性" PASS "R2=$R2_O >= $R2_MIN" \
                 || report "通道① O 力-λ 线性" FAIL "R2=$R2_O < $R2_MIN"
ok=$(awk -v r="$R2_H" -v m="$R2_MIN" 'BEGIN{print (r>=m)?1:0}')
[ "$ok" = "1" ] && report "通道① H 力-λ 线性" PASS "R2=$R2_H >= $R2_MIN" \
                 || report "通道① H 力-λ 线性" FAIL "R2=$R2_H < $R2_MIN"

# ================= 通道②: PW berry_phase 位移 FD =================
echo "---- [Stage 2] PW berry_phase 位移 FD (O z ±${DISP_ANG} Ang) ----"
GAMMAS=""
for sign in plus minus; do
    d="${RUNS}/disp_O_${sign}"
    mkdir -p "$d"
    cp "${CASES}/h2o/KPT" "$d/"
    # 生成 O 原子 z 向位移后的 STRU（O 块中唯一的坐标行）
    if [ "$sign" = "plus" ]; then sgn=1; else sgn=-1; fi
    awk -v dz="$DISP_ANG" -v s="$sgn" '
        BEGIN{OFMT="%.7f"; CONVFMT="%.7f"}
        /^O[ \t]*$/ {ino=1; print; next}
        ino==1 && NF==1 {print; ino=2; next}      # 磁矩行 0.0
        ino==2 && NF==1 {print; ino=3; next}      # 原子数行 1
        ino==3 && NF==3 {$3=$3+s*dz; ino=0; print; next}
        {print}' "${CASES}/h2o/STRU" > "$d/STRU"
    sed -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" "${CASES}/INPUT_pw_berry.tmpl" > "$d/INPUT"
    if run_abacus "$d"; then
        g=$(get_pw_gamma "$d")
        echo "disp_${sign}: γ_total=$g" | tee -a "$RESULTS"
        if [ "$sign" = "plus" ]; then G_PLUS="$g"; else G_MINUS="$g"; fi
    else
        report "disp_${sign} SCF" FAIL "未收敛"
        if [ "$sign" = "plus" ]; then G_PLUS="NA"; else G_MINUS="NA"; fi
    fi
done

if [ "$G_PLUS" != "NA" ] && [ -n "$G_PLUS" ] && [ "$G_MINUS" != "NA" ] && [ -n "$G_MINUS" ]; then
    DR_BOHR=$(awk -v d="$DISP_ANG" -v c="$ANG_TO_BOHR" 'BEGIN{printf "%.10e", 2*d*c}')
    Z_O_DISP=$(awk -v gp="$G_PLUS" -v gm="$G_MINUS" -v c="$F1_PI_OVER_A" -v dr="$DR_BOHR" -v f="$F2_SPIN_FACTOR" \
        'BEGIN{printf "%.6f", c*(gp-gm)/dr*f}')
    echo "通道②: Z*_O(位移FD)=$Z_O_DISP" | tee -a "$RESULTS"
else
    Z_O_DISP="NA"
    report "通道② γ 提取" FAIL "γ_total 未找到"
fi

# ================= 判定 =================
echo "---- [判定] ----"
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README，前置 P01）" | tee -a "$RESULTS"

if [ "$Z_O_DISP" != "NA" ]; then
    ok=$(awk -v a="$Z_O" -v b="$Z_O_DISP" -v t="$TOL_ZSTAR" 'BEGIN{d=a-b; if(d<0)d=-d; print (d<=t)?1:0}')
    [ "$ok" = "1" ] && report "两通道 Z*_O 互差" PASS "|$Z_O-$Z_O_DISP| <= $TOL_ZSTAR" \
                     || report "两通道 Z*_O 互差" FAIL "|$Z_O-$Z_O_DISP| > $TOL_ZSTAR"
fi

SUMZ=$(awk -v o="$Z_O" -v h="$Z_H" 'BEGIN{printf "%.6f", o+2*h}')
ok=$(awk -v s="$SUMZ" -v t="$TOL_ZSTAR" 'BEGIN{d=s; if(d<0)d=-d; print (d<=t)?1:0}')
[ "$ok" = "1" ] && report "电荷中和 |ΣZ*|" PASS "ΣZ*=$SUMZ" \
                 || report "电荷中和 |ΣZ*|" FAIL "ΣZ*=$SUMZ"

ok=$(awk -v z="$Z_O" -v lo="$ZSTAR_LO" -v hi="$ZSTAR_HI" 'BEGIN{print (z>=lo && z<=hi)?1:0}')
if [ "$ok" = "1" ]; then
    report "Z*_O 文献区间" PASS "Z*_O=$Z_O ∈ [$ZSTAR_LO,$ZSTAR_HI]"
else
    echo "Z*_O 文献区间: WARNING  Z*_O=$Z_O 越出 [$ZSTAR_LO,$ZSTAR_HI]（不计 FAIL）" | tee -a "$RESULTS"
fi

echo "SUMMARY: ${NPASS}/${NTOTAL} PASS" | tee -a "$RESULTS"
[ "$NPASS" -eq "$NTOTAL" ] && exit 0 || exit 1
