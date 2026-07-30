#!/bin/bash
# P16 PW 约束核整改验证：onsite_radius 扫描 x PW/LCAO 响应 dγ/dλ 对照
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
LAM_SCAN=0.01                     # +/- lambda (Ry)
RADIUS_LIST="6 10 14 20"          # onsite_radius (Bohr)
BOX_LIST="15 30"                  # 盒边长 (Bohr)
PW_ECUT=80                        # Ry

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

gen_input() { # $1=模板 $2=输出 $3=lambda $4=mixing_beta $5=onsite_radius $6=ecut
    sed -e "s#@PSEUDO_DIR@#$PSEUDO_DIR#g" \
        -e "s#@ORBITAL_DIR@#$ORBITAL_DIR#g" \
        -e "s#@LAMBDA@#$3#g" \
        -e "s#@MIXB@#$4#g" \
        -e "s#@RADIUS@#$5#g" \
        -e "s#@ECUT@#$6#g" "$1" > "$2"
}

get_rawg()     { grep -o 'Σγ_raw=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
get_pw_gamma() { grep -o 'γ_total=[^ ]*' "$1/run.log" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
is_conv()      { if grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; then echo INVALID; else echo OK; fi; }

# 覆盖率线索提取：deltap_results.dat 或 run.log 中 SMO 投影电荷相关行
coverage_hint() { # $1=run 目录
    local d="$1" out
    out="$(grep -iE 'SMO|coverage|proj.*charge|P_I' "$d/run.log" 2>/dev/null | head -5 || true)"
    if [ -n "$out" ]; then
        echo "$out"
        return
    fi
    local f
    f="$(ls "$d"/OUT.*/deltap_results.dat 2>/dev/null | head -1 || true)"
    if [ -n "$f" ]; then
        grep -E '# SMO radius|# Total' "$f" 2>/dev/null || true
        return
    fi
    # TODO: 当前 PW/LCAO 路径均无 SMO 投影电荷覆盖率直接输出键（见 README §4）；
    # 若后续版本在 run.log 增加覆盖率输出，请在此补充 grep 键。
    echo "N/A"
}

DATA="$RUNS/data.tsv"
: > "$DATA"

echo "===== P16 PW 约束核整改验证 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC"

for B in $BOX_LIST; do
    STRU="$CASES/STRU_${B}BOHR"
    # ---- PW: radius x lambda ----
    for R in $RADIUS_LIST; do
        for LAM in -$LAM_SCAN 0.0 $LAM_SCAN; do
            d="$RUNS/box$B/pw_r$R/lam_$(tag "$LAM")"
            mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
            gen_input "$CASES/INPUT.pw.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" "$R" "$PW_ECUT"
            run_abacus "$d"
            echo -e "box$B\tpw_r$R\t$LAM\t$(get_pw_gamma "$d")\t$(is_conv "$d")" >> "$DATA"
        done
    done
    # ---- LCAO 对照: lambda ----
    for LAM in -$LAM_SCAN 0.0 $LAM_SCAN; do
        d="$RUNS/box$B/lcao/lam_$(tag "$LAM")"
        mkdir -p "$d"; cp "$STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.lcao.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" 6.0 100
        run_abacus "$d"
        echo -e "box$B\tlcao\t$LAM\t$(get_rawg "$d")\t$(is_conv "$d")" >> "$DATA"
    done
done

# ---- 分析: dgamma/dlambda = unwrap(gamma(+)-gamma(-))/(2*LAM_SCAN) ----
ANALYSIS="$(awk -v lam="$LAM_SCAN" -v pi="$PI" '
$5=="INVALID"{ inv[$1" "$2]=1; next }
{ g[$1" "$2" "$3]=$4 }
END{
  split("15 30", boxes, " "); split("6 10 14 20", rs, " ");
  for (bi=1; bi<=2; bi++) {
    b="box"boxes[bi];
    for (ri=1; ri<=4; ri++) {
      k=b" pw_r"rs[ri];
      if ((b" pw_r"rs[ri]) in inv) { printf "%s pw_r%s INVALID\n", b, rs[ri]; continue }
      kp=k" "lam; km=k" -"lam;
      if ((kp in g) && (km in g)) {
        d=g[kp]-g[km];
        if (d>pi) d-=2*pi; else if (d<-pi) d+=2*pi;   # PW gamma_total wrapped: 解绕差值
        printf "%s pw_r%s %.8f\n", b, rs[ri], d/(2*lam);
      } else printf "%s pw_r%s MISSING\n", b, rs[ri];
    }
    kl=b" lcao";
    if (kl in inv) { printf "%s lcao INVALID\n", b; continue }
    kp=kl" "lam; km=kl" -"lam;
    if ((kp in g) && (km in g)) printf "%s lcao %.8f\n", b, (g[kp]-g[km])/(2*lam);
    else printf "%s lcao MISSING\n", b;
  }
}' "$DATA")"

declare -A DGDL
while read -r b ch v; do DGDL["$b $ch"]="$v"; done <<< "$ANALYSIS"

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
reldiff_abs() { # |x|-|y| 相对 |y|
    case "$1|$2" in *NAN*|*INVALID*|*MISSING*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{ax=(x>0?x:-x); ay=(y>0?y:-y); if(ay==0){print "NAN"}else{d=ax-ay; if(d<0)d=-d; printf "%.6f", d/ay}}'
}

{
echo ""
echo "----- 覆盖率线索（SMO 投影电荷，见 README §4） -----"
for B in $BOX_LIST; do
    for R in $RADIUS_LIST; do
        d="$RUNS/box$B/pw_r$R/lam_0p0"
        echo "  [box$B r=$R] $(coverage_hint "$d" | head -3 | tr '\n' ' ')"
    done
done

echo ""
echo "----- d(gamma)/d(lambda) 趋势表 (1/Ry) -----"
printf "%-8s" "通道"
for R in $RADIUS_LIST; do printf " %14s" "PW r=$R"; done
printf " %14s\n" "LCAO(ref)"
for B in $BOX_LIST; do
    printf "%-8s" "box$B"
    for R in $RADIUS_LIST; do printf " %14s" "${DGDL[box$B pw_r$R]:-NA}"; done
    printf " %14s\n" "${DGDL[box$B lcao]:-NA}"
done

echo ""
echo "----- 判据 -----"
for B in $BOX_LIST; do
    echo "[J-box$B] r=20 Bohr 处 |PW dγ/dλ| 与 |LCAO dγ/dλ| 差 <= 10%"
    check_le "box$B: |dgdl_pw(r=20)| vs |dgdl_lcao| 相对差 <= 0.10" \
        "$(reldiff_abs "${DGDL[box$B pw_r20]:-NAN}" "${DGDL[box$B lcao]:-NAN}")" 0.10
done
echo "[趋势] dγ/dλ(r) 随 r 单调趋稳：见上表（人工检查；若 r=14 与 r=20 差 >5% 请警惕未收敛）"
for B in $BOX_LIST; do
    echo "  box$B r14->r20 相对变化: $(reldiff_abs "${DGDL[box$B pw_r20]:-NAN}" "${DGDL[box$B pw_r14]:-NAN}")"
done

echo ""
echo "原始数据: $DATA"
echo "SUMMARY: $PASS/$TOTAL PASS"
} | tee "$RESULTS"

[ "$PASS" -eq "$TOTAL" ]
