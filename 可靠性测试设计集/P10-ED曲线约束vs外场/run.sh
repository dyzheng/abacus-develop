#!/bin/bash
# P10 E-D 曲线：DeltaP total lambda 扫描 vs efield E 扫描（30 Bohr H2O）
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
# F1: E(Ha/Bohr) = -PI*lambda/(2*A_BOHR)   (lambda 单位 Ry)
# F2: mu = (A_BOHR/PI)*gamma/F2_SPIN ;  E_phys(Ry) = E_KS - lambda*gamma_raw
# F2_SPIN=2 为 nspin=1 自旋因子（2026-07-30 冒烟实测 H2O 确认，待备忘录定稿）
F2_SPIN=2.0

LAM_POINTS="-0.08 -0.04 -0.02 0.0 0.02 0.04 0.08"          # Ry
E_POINTS="-0.004 -0.002 -0.001 0.0 0.001 0.002 0.004"      # Ha/Bohr

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

echo "===== P10 E-D 曲线: 约束 vs 外场 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC"

for LAM in $LAM_POINTS; do
    d="$RUNS/lam_$(tag "$LAM")"
    mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.deltap.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" 0.0
    run_abacus "$d"
    echo -e "dp\t$LAM\t$(get_rawg "$d")\t$(get_eks "$d")\t$(is_conv "$d")" >> "$DATA"
done

for E in $E_POINTS; do
    d="$RUNS/ef_$(tag "$E")"
    mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
    gen_input "$CASES/INPUT.efield.tmpl" "$d/INPUT" 0.0 0.4 "$E"
    run_abacus "$d"
    echo -e "ef\t$E\t$(get_rawg "$d")\t$(get_eks "$d")\t$(is_conv "$d")" >> "$DATA"
done

# ---- 分析（全部阻塞于 F1/F2，仅记录数值） ----
REPORT="$(awk -v pi="$PI" -v a="$A_BOHR" -v f2="$F2_SPIN" '
$5=="INVALID"{ next }
$1=="dp"{ nd++; dlam[nd]=$2; dsg[nd]=$3; deks[nd]=$4; next }
$1=="ef"{ ne++; eamp[ne]=$2; esg[ne]=$3; eeks[ne]=$4; next }
END{
  # DeltaP 侧: x = E(lambda) = -pi*lam/(2a), y = gamma_raw
  sx=0; sy=0; sxx=0; sxy=0; n=0;
  for (i=1; i<=nd; i++) { x=-pi*dlam[i]/(2*a); y=dsg[i];
    sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; n++ }
  den=n*sxx-sx*sx;
  if (den!=0) { slope_dp=(n*sxy-sx*sy)/den; icept_dp=(sy-slope_dp*sx)/n } else { slope_dp="NAN"; icept_dp="NAN" }
  # efield 侧: x = E, y = gamma_raw
  sx=0; sy=0; sxx=0; sxy=0; n=0;
  for (i=1; i<=ne; i++) { x=eamp[i]; y=esg[i];
    sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; n++ }
  den=n*sxx-sx*sx;
  if (den!=0) { slope_ef=(n*sxy-sx*sy)/den; icept_ef=(sy-slope_ef*sx)/n } else { slope_ef="NAN"; icept_ef="NAN" }
  # kappa_dp: E_phys(Ry)=EKS-lam*gamma vs gamma 二次拟合, kappa=2*c2
  s2=0; s4=0; se=0; sd2e=0; se1=0; n=0;
  for (i=1; i<=nd; i++) { g=dsg[i]; ep=deks[i]-dlam[i]*dsg[i];
    s2+=g*g; s4+=g*g*g*g; se+=ep; sd2e+=g*g*ep; n++ }
  den=n*s4-s2*s2;
  kap_dp=(den!=0)? 2*(n*sd2e-s2*se)/den : "NAN";
  # kappa_ef: EKS(Ry) vs E 二次拟合, kappa=2*c2
  s2=0; s4=0; se=0; sd2e=0; n=0;
  for (i=1; i<=ne; i++) { x=eamp[i];
    s2+=x*x; s4+=x*x*x*x; se+=eeks[i]; sd2e+=x*x*eeks[i]; n++ }
  den=n*s4-s2*s2;
  kap_ef=(den!=0)? 2*(n*sd2e-s2*se)/den : "NAN";
  # kappa_fd: efield 能量中心差分 (delta=0.001)
  ep=""; em=""; e0="";
  for (i=1; i<=ne; i++) { if(eamp[i]=="0.001")ep=eeks[i]; if(eamp[i]=="-0.001")em=eeks[i]; if(eamp[i]=="0.0"||eamp[i]=="0")e0=eeks[i] }
  kap_fd=(ep!=""&&em!=""&&e0!="") ? (ep+em-2*e0)/(0.001*0.001) : "NAN";
  # 零点 gamma 自洽
  g0_dp=""; g0_ef="";
  for (i=1; i<=nd; i++) if(dlam[i]=="0.0"||dlam[i]=="0") g0_dp=dsg[i];
  for (i=1; i<=ne; i++) if(eamp[i]=="0.0"||eamp[i]=="0") g0_ef=esg[i];
  printf "slope_dp %s\n", (slope_dp=="NAN"?"NAN":sprintf("%.8e", slope_dp));
  printf "slope_ef %s\n", (slope_ef=="NAN"?"NAN":sprintf("%.8e", slope_ef));
  printf "icept_dp %s\n", (icept_dp=="NAN"?"NAN":sprintf("%.8f", icept_dp));
  printf "icept_ef %s\n", (icept_ef=="NAN"?"NAN":sprintf("%.8f", icept_ef));
  printf "kappa_dp %s\n", (kap_dp=="NAN"?"NAN":sprintf("%.8e", kap_dp));
  printf "kappa_ef %s\n", (kap_ef=="NAN"?"NAN":sprintf("%.8e", kap_ef));
  printf "kappa_fd %s\n", kap_fd;
  printf "g0_dp %s\n", g0_dp;
  printf "g0_ef %s\n", g0_ef;
  # alpha 等价量（换算后对比）:
  # alpha = -(a/pi)*slope ; alpha_dp_k = -(2a^2/pi^2)/kappa_dp ; alpha_ef_k = -kappa_ef (Ry/Ha^2 -> a.u.)
  printf "alpha_s_dp %s\n", (slope_dp=="NAN"?"NAN":sprintf("%.8e", -(a/pi)*slope_dp/f2));
  printf "alpha_s_ef %s\n", (slope_ef=="NAN"?"NAN":sprintf("%.8e", -(a/pi)*slope_ef/f2));
  printf "alpha_k_dp %s\n", (kap_dp=="NAN"||kap_dp==0?"NAN":sprintf("%.8e", -(2*a*a/(pi*pi))/kap_dp/f2));
  printf "alpha_k_ef %s\n", (kap_ef=="NAN"?"NAN":sprintf("%.8e", -kap_ef));
  printf "alpha_k_fd %s\n", (kap_fd=="NAN"?"NAN":sprintf("%.8e", -kap_fd/2));
}' "$DATA")"

declare -A V
while read -r k v; do V[$k]="$v"; done <<< "$REPORT"

reldiff() {
    case "$1|$2" in *NAN*|*NA*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{if(y==0){print "NAN"}else{d=x-y; if(d<0)d=-d; b=(y>0?y:-y); printf "%.6f", d/b}}'
}
absdiff() {
    case "$1|$2" in *NAN*|*NA*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{d=x-y; if(d<0)d=-d; printf "%.6f", d}'
}

{
echo ""
echo "----- 数据表 -----"
echo "DeltaP 扫描 (E(lambda) = -pi*lambda/(2a), a=$A_BOHR Bohr):"
printf "%10s %14s %14s %16s\n" "lambda(Ry)" "E(Ha/Bohr)" "gamma_raw" "E_KS(Ry)"
grep -P '^dp\t' "$DATA" | while IFS=$'\t' read -r _ lam g e c; do
    EL="$(awk -v l="$lam" -v pi="$PI" -v a="$A_BOHR" 'BEGIN{printf "%.6e", -pi*l/(2*a)}')"
    printf "%10s %14s %14s %16s   [%s]\n" "$lam" "$EL" "$g" "$e" "$c"
done
echo "efield 扫描:"
printf "%10s %14s %16s\n" "E(Ha/Bohr)" "gamma_raw" "E_KS(Ry)"
grep -P '^ef\t' "$DATA" | while IFS=$'\t' read -r _ e g eks c; do
    printf "%10s %14s %16s   [%s]\n" "$e" "$g" "$eks" "$c"
done

echo ""
echo "----- 拟合结果 -----"
printf "%-28s %16s %16s\n" "量" "DeltaP 侧" "efield 侧"
printf "%-28s %16s %16s\n" "slope dGamma/dE" "${V[slope_dp]:-NA}" "${V[slope_ef]:-NA}"
printf "%-28s %16s %16s\n" "alpha=-(a/pi)*slope" "${V[alpha_s_dp]:-NA}" "${V[alpha_s_ef]:-NA}"
printf "%-28s %16s %16s\n" "gamma(0) (自洽)" "${V[g0_dp]:-NA}" "${V[g0_ef]:-NA}"
printf "%-28s %16s %16s\n" "kappa (能量二阶导)" "${V[kappa_dp]:-NA}" "${V[kappa_ef]:-NA}"
printf "%-28s %16s %16s\n" "alpha(kappa) 等价" "${V[alpha_k_dp]:-NA}" "${V[alpha_k_ef]:-NA}"
printf "%-28s %16s\n" "kappa_fd (efield 差分)" "${V[kappa_fd]:-NA}"
printf "%-28s %16s\n" "alpha_k_fd 等价" "${V[alpha_k_fd]:-NA}"

echo ""
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）——以下判据仅记录数值不参与 SUMMARY"
echo "[J1] 斜率差 <= 10%:        $(reldiff "${V[slope_dp]:-NAN}" "${V[slope_ef]:-NAN}")"
echo "[J2] gamma(0) 自洽 <0.01:  $(absdiff "${V[g0_dp]:-NAN}" "${V[g0_ef]:-NAN}")"
echo "[J3] kappa 三方互差 <=10%: dp_vs_ef=$(reldiff "${V[kappa_dp]:-NAN}" "${V[kappa_ef]:-NAN}")  dp_vs_fd=$(reldiff "${V[kappa_dp]:-NAN}" "${V[kappa_fd]:-NAN}")  ef_vs_fd=$(reldiff "${V[kappa_ef]:-NAN}" "${V[kappa_fd]:-NAN}")"
echo "     (kappa_dp 与 kappa_ef/fd 量纲不同, 可比量为上表 alpha(kappa) 等价行: dp_vs_ef=$(reldiff "${V[alpha_k_dp]:-NAN}" "${V[alpha_k_ef]:-NAN}"))"

echo ""
echo "原始数据: $DATA"
echo "SUMMARY: 0/0 PASS (全部判据阻塞于 F1/F2, 见 WARNING)"
} | tee "$RESULTS"

exit 0
