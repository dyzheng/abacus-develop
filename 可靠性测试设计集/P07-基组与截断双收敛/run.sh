#!/bin/bash
# P07 基组与截断双收敛：H2O (15 Ang 盒) LCAO DZP/TZDP/QZDP x PW ecut 40/60/80/100
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
BOX_ANG=15.0                       # Angstrom
EFIELD_POINTS="-0.001 -0.0005 0.0 0.0005 0.001"
ECUT_LIST="40 60 80 100"           # Ry
LEVEL_LIST="dzp tzdp qzdp"

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }

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

gen_input() { # $1=模板 $2=输出 $3=lambda $4=mixing_beta $5=efield_amp $6=ecutwfc
    sed -e "s#@PSEUDO_DIR@#$PSEUDO_DIR#g" \
        -e "s#@ORBITAL_DIR@#$ORBITAL_DIR#g" \
        -e "s#@LAMBDA@#$3#g" \
        -e "s#@MIXB@#$4#g" \
        -e "s#@EFIELD@#$5#g" \
        -e "s#@ECUT@#$6#g" "$1" > "$2"
}

get_rawg()     { grep -o 'Σγ_raw=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
get_pw_gamma() { grep -o 'γ_total=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
get_eks()      { tail -n +1 "$1"/OUT.*/running_scf.log 2>/dev/null | grep 'E_KohnSham' | tail -1 | awk '{print $2}' || true; }
is_conv()      { if grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; then echo INVALID; else echo OK; fi; }

# 检测 STRU 的 NUMERICAL_ORBITAL 块中轨道文件是否都存在于 $ORBITAL_DIR
orb_avail() { # $1=STRU 路径; 返回 0=齐全 1=缺失, 并打印缺失文件名
    local missing=0 f
    while read -r f; do
        [ -f "$ORBITAL_DIR/$f" ] || { echo "    缺轨道文件: $ORBITAL_DIR/$f" >&2; missing=1; }
    done < <(awk '/^NUMERICAL_ORBITAL/{f=1;next} f&&/^LATTICE_CONSTANT/{exit} f&&NF{print $1}' "$1")
    return $missing
}

DATA="$RUNS/data.tsv"
: > "$DATA"

echo "===== P07 基组与截断双收敛 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC"

# ---- LCAO 三层级 ----
declare -A LEVEL_OK
for LV in $LEVEL_LIST; do
    STRU="$CASES/basis_$LV/STRU"
    echo "[LCAO] 层级 $LV"
    if ! orb_avail "$STRU"; then
        echo "  SKIP: 层级 $LV 轨道文件缺失（见上，README 说明如何补充）"
        LEVEL_OK[$LV]=0
        continue
    fi
    LEVEL_OK[$LV]=1
    # lambda=0 偶极
    d="$RUNS/basis_$LV/lam_0"
    mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.deltap.tmpl" "$d/INPUT" 0.0 0.4 0.0 100
    run_abacus "$d"
    echo -e "lcao_$LV\tlam0\t0.0\t$(get_rawg "$d")\t\t$(is_conv "$d")" >> "$DATA"
    # efield 五点
    for E in $EFIELD_POINTS; do
        d="$RUNS/basis_$LV/ef_$(tag "$E")"
        mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.efield.tmpl" "$d/INPUT" 0.0 0.4 "$E" 100
        run_abacus "$d"
        echo -e "lcao_$LV\tef\t$E\t\t$(get_eks "$d")\t$(is_conv "$d")" >> "$DATA"
    done
done

# ---- PW 四截断 ----
STRU_PW="$CASES/basis_tzdp/STRU"   # PW 不读轨道块，复用 tzdp 几何
for EC in $ECUT_LIST; do
    echo "[PW] ecutwfc=$EC"
    d="$RUNS/pw_ecut$EC/lam_0"
    mkdir -p "$d"; cp "$STRU_PW" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.pw.tmpl" "$d/INPUT" 0.0 0.4 0.0 "$EC"
    run_abacus "$d"
    echo -e "pw_$EC\tlam0\t0.0\t$(get_pw_gamma "$d")\t\t$(is_conv "$d")" >> "$DATA"
    for E in $EFIELD_POINTS; do
        d="$RUNS/pw_ecut$EC/ef_$(tag "$E")"
        mkdir -p "$d"; cp "$STRU_PW" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.pw_efield.tmpl" "$d/INPUT" 0.0 0.4 "$E" "$EC"
        run_abacus "$d"
        echo -e "pw_$EC\tef\t$E\t\t$(get_eks "$d")\t$(is_conv "$d")" >> "$DATA"
    done
done

# ---- 分析 ----
A_BOHR="$(awk -v l="$BOX_ANG" -v b="$BOHR_PER_ANG" 'BEGIN{printf "%.10f", l*b}')"
ANALYSIS="$(awk -v pi="$PI" -v a="$A_BOHR" '
$6=="INVALID"{ next }
$2=="lam0"{ g0[$1]=$4; next }
$2=="ef"  { efe[$1" "$3]=$5; next }
END{
  for (k in g0) {
    mu[k]=(a/pi)*g0[k];
    n=0; s2=0; s4=0; se=0; sd2e=0;
    split("-0.001 -0.0005 0.0 0.0005 0.001", ep, " ");
    for (j=1; j<=5; j++) { d=ep[j]+0; key=k" "ep[j];
      if (key in efe) { n++; s2+=d*d; s4+=d*d*d*d; se+=efe[key]; sd2e+=d*d*efe[key] } }
    if (n>=3 && (n*s4-s2*s2)!=0) { c2=(n*sd2e-s2*se)/(n*s4-s2*s2); al[k]=-c2 } else al[k]="NAN";
    printf "%s mu %.8f\n", k, mu[k];
    printf "%s alpha %s\n", k, al[k];
  }
}' "$DATA")"

declare -A MU ALPHA
while read -r k qty v; do
    case "$qty" in
        mu)    MU[$k]="$v";;
        alpha) ALPHA[$k]="$v";;
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
reldiff() {
    case "$1|$2" in *NAN*|*NA*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{if(y==0){print "NAN"}else{d=x-y; if(d<0)d=-d; b=(y>0?y:-y); printf "%.6f", d/b}}'
}

{
echo ""
echo "----- 数据表 (a=$A_BOHR Bohr) -----"
printf "%-12s %16s %16s\n" "算例" "mu(e*Bohr)" "alpha(a.u.)"
for LV in $LEVEL_LIST; do
    [ "${LEVEL_OK[$LV]:-0}" = "1" ] && printf "%-12s %16s %16s\n" "lcao_$LV" "${MU[lcao_$LV]:-NA}" "${ALPHA[lcao_$LV]:-NA}"
done
for EC in $ECUT_LIST; do
    printf "%-12s %16s %16s\n" "pw_ecut$EC" "${MU[pw_$EC]:-NA}" "${ALPHA[pw_$EC]:-NA}"
done

echo ""
echo "----- 判据 -----"
echo "[J1] LCAO 相邻层级 mu 差 <= 2%, alpha 差 <= 3%（层级缺失自动 SKIP）"
PREV=""
for LV in $LEVEL_LIST; do
    [ "${LEVEL_OK[$LV]:-0}" = "1" ] || continue
    if [ -n "$PREV" ]; then
        check_le "mu: $PREV->$LV 相对差 <= 0.02" "$(reldiff "${MU[lcao_$LV]:-NAN}" "${MU[lcao_$PREV]:-NAN}")" 0.02
        check_le "alpha: $PREV->$LV 相对差 <= 0.03" "$(reldiff "${ALPHA[lcao_$LV]:-NAN}" "${ALPHA[lcao_$PREV]:-NAN}")" 0.03
    fi
    PREV="$LV"
done

echo "[J2] PW ecut80->100: mu 差 <= 1%, alpha 差 <= 2%"
check_le "mu: ecut80->100 相对差 <= 0.01" "$(reldiff "${MU[pw_100]:-NAN}" "${MU[pw_80]:-NAN}")" 0.01
check_le "alpha: ecut80->100 相对差 <= 0.02" "$(reldiff "${ALPHA[pw_100]:-NAN}" "${ALPHA[pw_80]:-NAN}")" 0.02

echo "[J3] LCAO 与 PW 收敛极限 mu 互差 <= 5%（mod 2pi 对齐 gamma 分支后比较）"
BEST=""
for LV in qzdp tzdp dzp; do
    if [ "${LEVEL_OK[$LV]:-0}" = "1" ]; then BEST="$LV"; break; fi
done
if [ -z "$BEST" ]; then
    echo "  [FAIL] 无可用 LCAO 层级"; TOTAL=$((TOTAL+1))
else
    # gamma 域 mod-2pi 差 -> mu 差
    G_L="$(awk -v m="${MU[lcao_$BEST]:-NAN}" -v a="$A_BOHR" -v pi="$PI" 'BEGIN{printf "%.10f", m*pi/a}')"
    G_P="$(awk -v m="${MU[pw_100]:-NAN}" -v a="$A_BOHR" -v pi="$PI" 'BEGIN{printf "%.10f", m*pi/a}')"
    REL="$(awk -v gl="$G_L" -v gp="$G_P" -v a="$A_BOHR" -v pi="$PI" -v mu="${MU[lcao_$BEST]:-NAN}" '
        BEGIN{ d=gl-gp; t=2*pi; d=d-t*int(d/t); if(d>pi)d=t-d; if(d<0)d=-d;
               dmu=(a/pi)*d; b=(mu>0?mu:-mu); if(b==0){print "NAN"}else printf "%.6f", dmu/b }')"
    check_le "mu: lcao_$BEST vs pw_ecut100 (mod 2pi) 相对差 <= 0.05" "$REL" 0.05
fi

echo ""
echo "原始数据: $DATA"
echo "SUMMARY: $PASS/$TOTAL PASS"
} | tee "$RESULTS"

[ "$PASS" -eq "$TOTAL" ]
