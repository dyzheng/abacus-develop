#!/bin/bash
# P06 盒尺寸收敛：H2O 偶极 mu(L) 与极化率 alpha(L) 随盒尺寸 L 的收敛
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
BOHR_PER_ANG=1.8897261254578284
# F1: lambda(Ry) -> E(Ha/Bohr):  E = -PI*lambda/(2*a),  a = gdir 方向盒边长 (Bohr)
# F2: mu(e*Bohr) = (a/PI)*unwrap(gamma_raw)/F2_SPIN ;  alpha = -(a/PI/F2_SPIN)*dgamma/dE
#     F2_SPIN=2 为 nspin=1 自旋因子（2026-07-30 冒烟实测 H2O 确认，待备忘录定稿）
# DeltaP 通道: alpha_dp = (2*a^2/PI^2/F2_SPIN) * dgamma/dlambda
F2_SPIN=2.0

BOX_LIST="12 15 18 21 24"                               # Angstrom
EFIELD_POINTS="-0.001 -0.0005 0.0 0.0005 0.001"         # Ha/Bohr, dip_cor_flag 1
LAM_DP=0.02                                             # DeltaP +/- lambda (Ry)

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }
mixb_for() { awk -v l="$1" 'BEGIN{print (l+0<0)?"0.3":"0.4"}'; }

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

get_rawg() { grep -o 'Σγ_raw=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
get_eks()  { tail -n +1 "$1"/OUT.*/running_scf.log 2>/dev/null | grep 'E_KohnSham' | tail -1 | awk '{print $2}' || true; }
is_conv()  { if grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; then echo INVALID; else echo OK; fi; }

DATA="$RUNS/data.tsv"
: > "$DATA"

echo "===== P06 盒尺寸收敛 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC"

for L in $BOX_LIST; do
    STRU="$CASES/STRU_L$L"
    # 1) lambda=0 偶极
    d="$RUNS/L$L/lam_0"
    mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.deltap.tmpl" "$d/INPUT" 0.0 0.4 0.0
    run_abacus "$d"
    echo -e "$L\tlam0\t0.0\t$(get_rawg "$d")\t\t$(is_conv "$d")" >> "$DATA"

    # 2) efield 五点 FD
    for E in $EFIELD_POINTS; do
        d="$RUNS/L$L/ef_$(tag "$E")"
        mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.efield.tmpl" "$d/INPUT" 0.0 0.4 "$E"
        run_abacus "$d"
        echo -e "$L\tef\t$E\t\t$(get_eks "$d")\t$(is_conv "$d")" >> "$DATA"
    done

    # 3) DeltaP +/- lambda
    for LAM in -$LAM_DP $LAM_DP; do
        d="$RUNS/L$L/lam_$(tag "$LAM")"
        mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.deltap.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" 0.0
        run_abacus "$d"
        echo -e "$L\tdp\t$LAM\t$(get_rawg "$d")\t\t$(is_conv "$d")" >> "$DATA"
    done
done

# ---- 分析 ----
ANALYSIS="$(awk -v pi="$PI" -v bpa="$BOHR_PER_ANG" -v lamdp="$LAM_DP" -v f2="$F2_SPIN" '
BEGIN{ nbox=split("12 15 18 21 24", boxes, " ") }
$6=="INVALID"{ inv[$1" "$2" "$3]=1; next }
$2=="lam0"{ rawg0[$1]=$4; next }
$2=="ef"  { efe[$1" "$3]=$5; next }
$2=="dp"  { dpg[$1" "$3]=$4; next }
END{
  for (i=1; i<=nbox; i++) {
    L=boxes[i]; a=L*bpa;
    if (!(L in rawg0)) { printf "missing %s\n", L; continue }
    gg=rawg0[L]; nn=gg/(2*pi); nn=(nn>=0)?int(nn+0.5):int(nn-0.5); gg=gg-2*pi*nn;
    mu[L]=(a/pi)*gg/f2;
    # efield 五点二次最小二乘: E_Ry = c0 + c1*d + c2*d^2, alpha = -c2 (Ha 制 a.u.)
    n=0; s2=0; s4=0; se=0; sd2e=0;
    split("-0.001 -0.0005 0.0 0.0005 0.001", ep, " ");
    for (j=1; j<=5; j++) { d=ep[j]+0; key=L" "ep[j];
      if (key in efe) { n++; s2+=d*d; s4+=d*d*d*d; se+=efe[key]; sd2e+=d*d*efe[key] } }
    if (n>=3 && (n*s4-s2*s2)!=0) { c2=(n*sd2e-s2*se)/(n*s4-s2*s2); aref[L]=-c2 } else aref[L]="NAN";
    kp=L" "lamdp; km=L" -"lamdp;
    if ((kp in dpg) && (km in dpg)) {
      dgdl=(dpg[kp]-dpg[km])/(2*lamdp);
      adp[L]=(2*a*a/(pi*pi))*dgdl/f2;
    } else adp[L]="NAN";
    printf "mu %.6f %s\n", L, mu[L];
    printf "aref %.6f %s\n", L, aref[L];
    printf "adp %.6f %s\n", L, adp[L];
  }
}' "$DATA")"

declare -A MU AREF ADP
while read -r k L v; do
    case "$k" in
        mu)   MU[$L]="$v";;
        aref) AREF[$L]="$v";;
        adp)  ADP[$L]="$v";;
        missing) echo "WARNING: 缺数据 L=$L";;
    esac
done <<< "$ANALYSIS"

PASS=0; TOTAL=0
check_le() { # $1=描述 $2=数值 $3=限
    TOTAL=$((TOTAL+1))
    case "$2" in *NAN*|*NA*|"") echo "  [FAIL] $1 (数据缺失)"; return;; esac
    if awk -v v="$2" -v l="$3" 'BEGIN{exit !(v<=l)}'; then
        echo "  [PASS] $1 ($2 <= $3)"; PASS=$((PASS+1))
    else
        echo "  [FAIL] $1 ($2 > $3)"
    fi
}
blocked() { echo "  [BLOCKED] $1 (值: $2)"; }

reldiff() { # $1,$2 数值; 任一缺失/非数 -> NAN
    case "$1|$2" in *NAN*|*NA*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{if(y==0){print "NAN"}else{d=x-y; if(d<0)d=-d; b=(y>0?y:-y); printf "%.6f", d/b}}'
}

{
echo ""
echo "----- 数据表 -----"
printf "%6s %14s %14s %14s\n" "L(Ang)" "mu(e*Bohr)" "alpha_ref" "alpha_DeltaP"
for L in $BOX_LIST; do
    printf "%6s %14s %14s %14s\n" "$L" "${MU[$L]:-NA}" "${AREF[$L]:-NA}" "${ADP[$L]:-NA}"
done
echo ""
echo "----- 判据 -----"
echo "[J1] 偶极: 15 Ang 与 24 Ang 相对差 <= 3%"
check_le "|mu15-mu24|/|mu24| <= 0.03" "$(reldiff "${MU[15]:-NAN}" "${MU[24]:-NAN}")" 0.03

echo ""
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）——以下 alpha 相关判据仅记录数值不参与 SUMMARY"
echo "[J2] alpha_ref(efield FD): 15 Ang 与 24 Ang 相对差 <= 5%"
blocked "|aref15-aref24|/|aref24| <= 0.05" "$(reldiff "${AREF[15]:-NAN}" "${AREF[24]:-NAN}")"
echo "[J3] alpha_DeltaP: 15 Ang 与 24 Ang 相对差 <= 5%"
blocked "|adp15-adp24|/|adp24| <= 0.05" "$(reldiff "${ADP[15]:-NAN}" "${ADP[24]:-NAN}")"
echo "[J4] 两通道 alpha(L) 单调趋势一致（逐段符号比对）"
TREND="$(awk -v r="${AREF[12]:-NAN} ${AREF[15]:-NAN} ${AREF[18]:-NAN} ${AREF[21]:-NAN} ${AREF[24]:-NAN}" \
            -v d="${ADP[12]:-NAN} ${ADP[15]:-NAN} ${ADP[18]:-NAN} ${ADP[21]:-NAN} ${ADP[24]:-NAN}" '
BEGIN{ n=split(r, R, " "); split(d, D, " ");
       nagree=0; nseg=0;
       for (i=1; i<n; i++) {
         if (R[i]=="NAN" || R[i+1]=="NAN" || D[i]=="NAN" || D[i+1]=="NAN") continue;
         dr=R[i+1]-R[i]; dd=D[i+1]-D[i];
         if (dr==0 || dd==0) continue; nseg++;
         if ((dr>0)==(dd>0)) nagree++ }
       printf "%d/%d", nagree, nseg }')"
blocked "alpha_ref 与 alpha_DeltaP 趋势一致段数" "$TREND"

echo ""
echo "原始数据: $DATA"
echo "SUMMARY: $PASS/$TOTAL PASS (alpha 判据阻塞, 见 WARNING)"
} | tee "$RESULTS"

[ "$PASS" -eq "$TOTAL" ]
