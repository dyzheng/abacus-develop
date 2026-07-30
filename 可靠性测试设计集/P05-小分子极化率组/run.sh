#!/bin/bash
# =====================================================================
# P05 小分子极化率组三方对标（H2O/NH3/CH4/CO/HF）
# 每分子两通道: ① efield 能量 FD → α_ref; ② DeltaP ±λ → dγ/dλ → α_DeltaP
# 基组层级: tzdp（默认轨道）+ dzp（占位, 缺文件自动 SKIP）
# CH4 无极性方向, 三方向独立测量后平均
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
ANG_TO_BOHR=1.8897261254578284
F2_SPIN_FACTOR=0.5            # γ↔μ 自旋因子 (nspin=1 取 1/2)，冒烟实测 H2O 确认 (2026-07-30)，待备忘录定稿
# λ→E: E = -πλ/(2a) (λ 单位 Ry, E 单位 Ha, a 单位 Bohr)
# α = -(a/π)·dγ/dE = (2a²/π²)·dγ/dλ  （推导见 README §4）
alpha_coef() { awk -v a="$1" -v pi="$PI" -v f="$F2_SPIN_FACTOR" 'BEGIN{printf "%.10e", 2*a*a/(pi*pi)*f}'; }
# efield FD: α(a.u.) = -c2, c2 为 E(Ry) 对 δ(Ha) 二次拟合系数
TOL_REL=0.10                  # 逐分子 |Δα|/α_ref
TOL_MAE=0.15                  # 组 MAE vs CCSD(T)
LAMBDAS="-0.08 -0.02 0.02 0.08"
EAMPS="-0.001 -0.0005 0.0 0.0005 0.001"
declare -A REF=( [h2o]=9.85 [nh3]=14.6 [ch4]=17.5 [co]=13.1 [hf]=5.6 )
declare -A AB=( [h2o]=30.0 [nh3]=22.6767135 [ch4]=22.6767135 [co]=22.6767135 [hf]=22.6767135 )
REF_ORDER="hf h2o co nh3 ch4"   # α 升序参考
MOLS="h2o nh3 ch4 co hf"
LEVELS="tzdp dzp"
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

quad_c2() { # stdin "x y"（x 关于 0 对称）→ 二次系数 c2
awk '
{ sx2+=$1*$1; sx2y+=$1*$1*$2; sy+=$2; sx4+=$1*$1*$1*$1; n++ }
END { printf "%.10e", (n*sx2y - sx2*sy)/(n*sx4 - sx2*sx2) }'
}

check_level() { # $1=level → 0 可用 / 1 缺轨道
    local lvl="$1" mol orb
    for mol in $MOLS; do
        while read -r orb; do
            [ -z "$orb" ] && continue
            if [ ! -f "${ORBITAL_DIR}/${orb}" ]; then
                echo "  [$lvl/$mol] 缺轨道文件: $orb" | tee -a "$RESULTS"
                return 1
            fi
        done < <(awk '/NUMERICAL_ORBITAL/{f=1;next} f&&NF==0{exit} f{print $1}' "${CASES}/${lvl}/${mol}/STRU")
    done
    return 0
}

echo "==================== P05 小分子极化率组 ===================="
echo "ABACUS=$ABACUS  NPROC=$NPROC"

declare -A AREF ADP
for lvl in $LEVELS; do
    echo "==== [层级 $lvl] ===="
    if ! check_level "$lvl"; then
        echo "层级 $lvl: SKIP（ORBITAL_DIR 缺该层级轨道文件，见上）" | tee -a "$RESULTS"
        continue
    fi
    for mol in $MOLS; do
        if [ "$mol" = "ch4" ]; then DIRS="1 2 3"; else DIRS="3"; fi
        acoef=$(alpha_coef "${AB[$mol]}")
        for gd in $DIRS; do
            echo "---- [$lvl/$mol gdir=$gd] ----"
            # ---- 通道①: efield 能量 FD ----
            XY=""
            for ea in $EAMPS; do
                tag=$(echo "$ea" | sed 's/-/m/;s/\./p/')
                d="${RUNS}/${lvl}_${mol}_d${gd}_ef_${tag}"
                mkdir -p "$d"
                cp "${CASES}/${lvl}/${mol}/STRU" "$d/"
                mesh="1 1 2"; [ "$gd" = "1" ] && mesh="2 1 1"; [ "$gd" = "2" ] && mesh="1 2 1"
                printf 'K_POINTS\n0\nGamma\n%s 0 0 0\n' "$mesh" > "$d/KPT"
                sed -e "s|@EFIELD_AMP@|$ea|" -e "s|@EFIELD_DIR@|$gd|" \
                    -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" -e "s|@ORBITAL_DIR@|$ORBITAL_DIR|" \
                    "${CASES}/INPUT_efield.tmpl" > "$d/INPUT"
                if run_abacus "$d"; then
                    e=$(get_eks "$d")
                    echo "  efield=$ea  E_KS=$e Ry" | tee -a "$RESULTS"
                    [ -n "$e" ] && XY="${XY}${ea} ${e}\n"
                else
                    report "$lvl/$mol/d$gd efield=$ea" FAIL "SCF 未收敛"
                fi
            done
            C2=$(printf "$XY" | quad_c2)
            a_ref=$(awk -v c="$C2" 'BEGIN{printf "%.6f", -c}')
            # ---- 通道②: DeltaP ±λ ----
            XY=""
            for lam in $LAMBDAS; do
                tag=$(echo "$lam" | sed 's/-/m/;s/\./p/')
                d="${RUNS}/${lvl}_${mol}_d${gd}_lam_${tag}"
                mkdir -p "$d"
                cp "${CASES}/${lvl}/${mol}/STRU" "$d/"
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
                    report "$lvl/$mol/d$gd λ=$lam" FAIL "SCF 未收敛"
                fi
            done
            FIT=$(printf "$XY" | linfit)
            SLOPE=$(echo "$FIT" | awk '{print $1}')
            R2=$(echo "$FIT" | awk '{print $3}')
            a_dp=$(awk -v s="$SLOPE" -v c="$acoef" 'BEGIN{printf "%.6f", c*s}')
            echo "  [$lvl/$mol/d$gd] α_ref=$a_ref  α_DeltaP=$a_dp  (dγ/dλ R2=$R2)" | tee -a "$RESULTS"
            AREF[$lvl,$mol]="${AREF[$lvl,$mol]} $a_ref"
            ADP[$lvl,$mol]="${ADP[$lvl,$mol]} $a_dp"
        done
    done
done

# ================= 判定 =================
echo "---- [判定] ----"
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README，前置 P01）" | tee -a "$RESULTS"

mean() { awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF; else print "NA"}'; }

for lvl in $LEVELS; do
    [ -z "${AREF[$lvl,h2o]}" ] && continue   # 层级被 SKIP
    echo "==== [层级 $lvl 判定] ====" | tee -a "$RESULTS"
    declare -A AVG=()
    for mol in $MOLS; do
        r=$(echo ${AREF[$lvl,$mol]} | mean)
        p=$(echo ${ADP[$lvl,$mol]} | mean)
        AVG[$mol]=$(awk -v a="$r" -v b="$p" 'BEGIN{printf "%.6f",(a+b)/2}')
        echo "  $mol: α_ref=$r  α_DeltaP=$p  CCSD(T)=${REF[$mol]}" | tee -a "$RESULTS"
        ok=$(awk -v a="$r" -v b="$p" -v t="$TOL_REL" \
            'BEGIN{d=a-b; if(d<0)d=-d; base=(a<0)?-a:a; print (base>0 && d/base<=t)?1:0}')
        [ "$ok" = "1" ] && report "$lvl/$mol 双通道一致" PASS "|Δα|/α_ref <= $TOL_REL" \
                         || report "$lvl/$mol 双通道一致" FAIL "α_ref=$r α_DeltaP=$p"
    done
    # 组 MAE vs CCSD(T)（两通道均值, 相对误差）
    MAE=$(for mol in $MOLS; do
        awk -v m="${AVG[$mol]}" -v r="${REF[$mol]}" 'BEGIN{d=m-r; if(d<0)d=-d; printf "%.6f\n", d/r}'
    done | awk '{s+=$1; n++} END{printf "%.6f", s/n}')
    echo "  [$lvl] 组 MAE(相对) = $MAE" | tee -a "$RESULTS"
    ok=$(awk -v m="$MAE" -v t="$TOL_MAE" 'BEGIN{print (m<=t)?1:0}')
    [ "$ok" = "1" ] && report "$lvl 组 MAE" PASS "MAE=$MAE <= $TOL_MAE" \
                     || report "$lvl 组 MAE" FAIL "MAE=$MAE > $TOL_MAE"
    # α 排序一致性
    ORDER=$(for mol in $MOLS; do echo "${AVG[$mol]} $mol"; done | sort -n | awk '{printf "%s ", $2}')
    ORDER=$(echo $ORDER)
    [ "$ORDER" = "$REF_ORDER" ] && report "$lvl α 排序" PASS "$ORDER" \
                                 || report "$lvl α 排序" FAIL "$ORDER != $REF_ORDER"
    unset AVG; declare -A AVG=()
done

echo "SUMMARY: ${NPASS}/${NTOTAL} PASS" | tee -a "$RESULTS"
[ "$NPASS" -eq "$NTOTAL" ] && exit 0 || exit 1
