#!/bin/bash
# =====================================================================
# P04 小分子偶极组四方对标（CH4/CO/NH3/HF/H2S）
# 每分子: LCAO λ=0 测量 (rawG Σγ_raw) → μ_LCAO; PW berry_phase → μ_PW
# 判据: LCAO-PW 互差 ≤0.02 D; 对 CCSD(T) MAE ≤0.03 D; CH4 |μ|≤0.01 D; CO 符号
# 无阻塞。用法: bash run.sh  (env: ABACUS PSEUDO_DIR ORBITAL_DIR NPROC)
# =====================================================================
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

# ---------------- 换算常量区 ----------------
PI=3.141592653589793
BOX_ANG=12.0                                    # 所有分子盒边长 (Angstrom)
A_BOHR=$(awk -v l="$BOX_ANG" 'BEGIN{printf "%.10f", l*1.8897261254578284}')
F1_PI_OVER_A=$(awk -v a="$A_BOHR" -v pi="$PI" 'BEGIN{printf "%.10e", a/pi}')  # a/π
EBOHR_TO_DEBYE=2.541746                         # 1 e·Bohr = 2.541746 Debye
F2_SPIN_FACTOR=0.5                              # γ↔μ 自旋因子 (nspin=1 取 1/2)。冒烟实测(2026-07-30, H2O):
                                                # Σγ_raw=-12.718 rad → unwrap -0.1517 → ×0.5 → μ=1.841 D (实验 1.855)。待备忘录定稿
TOL_INTER=0.02                                  # LCAO-PW 互差 (D)
TOL_MAE=0.03                                    # 组 MAE (D)
TOL_CH4=0.01                                    # CH4 零点 (D)
CO_EXPECT_SIGN=-1                               # CO μ_z 期望符号（O 端为负约定, 见 README §5）
# CCSD(T) 参考偶极 (D) —— GSCDB138/Dip146
declare -A REF=( [ch4]=0.0 [co]=0.122 [nh3]=1.47 [hf]=1.83 [h2s]=0.97 )
MOLS="ch4 co nh3 hf h2s"
# --------------------------------------------

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"
mkdir -p "$RUNS"
: > "$RESULTS"

NPASS=0; NTOTAL=0
report() { # $1=项目 $2=PASS/FAIL/SKIP $3=说明
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

get_rawg() { # $1=rundir → Σγ_raw
    grep "rawG" "$1/run.log" 2>/dev/null | tail -1 | sed -E 's/.*Σγ_raw=([^ ]+).*/\1/'
}
get_pw_gamma() { # $1=rundir → [DeltaP-PW] γ_total
    grep "DeltaP-PW" "$1/run.log" 2>/dev/null | grep "γ_total" | tail -1 \
        | sed -E 's/.*γ_total=([^ ]+).*/\1/'
}

gamma_to_debye() { # $1=γ → μ(D) = (a/π)·unwrap(γ)·F2 ×2.541746（raw γ 为多圈 unwrapped 值，先折叠到 (-π,π]）
    awk -v g="$1" -v c="$F1_PI_OVER_A" -v f="$F2_SPIN_FACTOR" -v d="$EBOHR_TO_DEBYE" \
        'BEGIN{pi=atan2(0,-1); n=g/(2*pi); n=(n>=0)?int(n+0.5):int(n-0.5); g=g-2*pi*n; printf "%.6f", c*g*f*d}'
}

echo "==================== P04 小分子偶极组 ===================="
echo "ABACUS=$ABACUS  NPROC=$NPROC"
echo "分子    μ_LCAO(D)    μ_PW(D)    CCSD(T)(D)" | tee -a "$RESULTS"

declare -A MU_LCAO MU_PW
for mol in $MOLS; do
    echo "---- [${mol}] ----"
    # LCAO 通道
    d="${RUNS}/${mol}_lcao"
    mkdir -p "$d"
    cp "${CASES}/${mol}/STRU" "${CASES}/${mol}/KPT" "$d/"
    sed -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" -e "s|@ORBITAL_DIR@|$ORBITAL_DIR|" \
        "${CASES}/INPUT_lcao.tmpl" > "$d/INPUT"
    if run_abacus "$d"; then
        g=$(get_rawg "$d")
        [ -n "$g" ] && MU_LCAO[$mol]=$(gamma_to_debye "$g") || MU_LCAO[$mol]="NA"
    else
        report "${mol} LCAO SCF" FAIL "未收敛"; MU_LCAO[$mol]="NA"
    fi
    # PW 通道
    d="${RUNS}/${mol}_pw"
    mkdir -p "$d"
    cp "${CASES}/${mol}/STRU" "${CASES}/${mol}/KPT" "$d/"
    sed -e "s|@PSEUDO_DIR@|$PSEUDO_DIR|" "${CASES}/INPUT_pw.tmpl" > "$d/INPUT"
    if run_abacus "$d"; then
        g=$(get_pw_gamma "$d")
        [ -n "$g" ] && MU_PW[$mol]=$(gamma_to_debye "$g") || MU_PW[$mol]="NA"
    else
        report "${mol} PW SCF" FAIL "未收敛"; MU_PW[$mol]="NA"
    fi
    printf "%-6s %10s %10s %10s\n" "$mol" "${MU_LCAO[$mol]}" "${MU_PW[$mol]}" "${REF[$mol]}" | tee -a "$RESULTS"
done

echo "---- [判定] ----"
# 判据1: 逐分子 LCAO-PW 互差 ≤0.02 D
for mol in $MOLS; do
    if [ "${MU_LCAO[$mol]}" = "NA" ] || [ "${MU_PW[$mol]}" = "NA" ]; then
        report "${mol} LCAO-PW 互差" FAIL "缺数据"; continue
    fi
    ok=$(awk -v a="${MU_LCAO[$mol]}" -v b="${MU_PW[$mol]}" -v t="$TOL_INTER" \
        'BEGIN{d=a-b; if(d<0)d=-d; print (d<=t)?1:0}')
    [ "$ok" = "1" ] && report "${mol} LCAO-PW 互差" PASS "|${MU_LCAO[$mol]}-${MU_PW[$mol]}| <= $TOL_INTER D" \
                     || report "${mol} LCAO-PW 互差" FAIL "|${MU_LCAO[$mol]}-${MU_PW[$mol]}| > $TOL_INTER D"
done

# 判据2: 对 CCSD(T) 组 MAE ≤0.03 D（LCAO 通道, 用 |μ| 比较大小）
MAE=$( for mol in $MOLS; do
    [ "${MU_LCAO[$mol]}" = "NA" ] && continue
    awk -v m="${MU_LCAO[$mol]}" -v r="${REF[$mol]}" 'BEGIN{a=(m<0)?-m:m; d=a-r; if(d<0)d=-d; printf "%.6f\n", d}'
done | awk '{s+=$1; n++} END{if(n>0) printf "%.6f", s/n; else print "NA"}' )
echo "组 MAE (LCAO vs CCSD(T)) = $MAE D" | tee -a "$RESULTS"
if [ "$MAE" != "NA" ]; then
    ok=$(awk -v m="$MAE" -v t="$TOL_MAE" 'BEGIN{print (m<=t)?1:0}')
    [ "$ok" = "1" ] && report "组 MAE ≤ $TOL_MAE D" PASS "MAE=$MAE" \
                     || report "组 MAE ≤ $TOL_MAE D" FAIL "MAE=$MAE"
fi

# 判据3: CH4 零点 |μ| ≤ 0.01 D（两通道各自）
for ch in LCAO PW; do
    v="MU_${ch}[ch4]"; m=${!v}
    [ "$m" = "NA" ] && { report "CH4 零点 ($ch)" FAIL "缺数据"; continue; }
    ok=$(awk -v m="$m" -v t="$TOL_CH4" 'BEGIN{a=(m<0)?-m:m; print (a<=t)?1:0}')
    [ "$ok" = "1" ] && report "CH4 零点 ($ch)" PASS "|μ|=$m <= $TOL_CH4 D" \
                     || report "CH4 零点 ($ch)" FAIL "|μ|=$m > $TOL_CH4 D"
done

# 判据4: CO 符号检查（O 端为负约定: O 在 +z, μ = Σq·r ⇒ 期望 μ_z < 0）
m=${MU_LCAO[co]}
if [ "$m" = "NA" ]; then
    report "CO 符号" FAIL "缺数据"
else
    ok=$(awk -v m="$m" -v e="$CO_EXPECT_SIGN" 'BEGIN{s=(m>0)?1:(m<0?-1:0); print (s==e)?1:0}')
    [ "$ok" = "1" ] && report "CO 符号" PASS "μ_z=$m D 符合约定（O 端为负）" \
                     || report "CO 符号" FAIL "μ_z=$m D 方向相反"
fi

echo "SUMMARY: ${NPASS}/${NTOTAL} PASS" | tee -a "$RESULTS"
[ "$NPASS" -eq "$NTOTAL" ] && exit 0 || exit 1
