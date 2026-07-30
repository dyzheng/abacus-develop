#!/bin/bash
# P08 约束响应线性与 ±λ 对称性（无阻塞）
# 九点 lambda 扫描: -0.08 -0.04 -0.02 -0.01 0.0 0.01 0.02 0.04 0.08
# 收集 Σγ_raw(lambda): 全线性 R^2>=0.98, 反对称偏差<5%, 子窗口(|lam|<=0.02)斜率差<=3%
# 逐点记录 SCF 迭代数；标记未收敛点为 INVALID（λ_crit 候选）
# 同时提取 TOTAL-FORCE 中 O 原子 z 分量（供 P03 复用，存结果表）
# 用法：./run.sh  （环境变量 ABACUS/PSEUDO_DIR/ORBITAL_DIR/NPROC 可覆盖）
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases/h2o"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"
DAT="${RUNS}/lambda_scan.dat"

LAMS="-0.08 -0.04 -0.02 -0.01 0.0 0.01 0.02 0.04 0.08"

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }
run_abacus() {
    cd "$1"
    rm -rf OUT.*
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 || true
    else
        "$ABACUS" > run.log 2>&1 || true
    fi
    cd "$BASEDIR"
}
get_rawg() { grep "rawG" "$1"/run.log 2>/dev/null | tail -1 | sed -E 's/.*Σγ_raw=([^ ]+).*/\1/'; }
get_scfiter() { grep -c "E_KohnSham" "$1"/OUT.*/running_scf.log 2>/dev/null || echo 0; }
get_oforce_z() { awk '/TOTAL-FORCE/{f=1} f && $1 ~ /^O[0-9]/ {print $NF; exit}' "$1"/OUT.*/running_scf.log 2>/dev/null; }

echo "======================================================"
echo " P08 约束响应线性与 ±λ 对称性"
echo " ABACUS=$ABACUS  NPROC=$NPROC"
echo "======================================================"
mkdir -p "$RUNS"
: > "$RESULTS"
echo "# lambda  gamma_raw(rad)  scf_iter  O_force_z(eV/Ang)  status" > "$DAT"

declare -A GG IT FZ ST
for lam in $LAMS; do
    d="${RUNS}/lam_$(tag "$lam")"
    mkdir -p "$d"
    cp "$CASES/STRU" "$CASES/KPT" "$d/"
    # 负 lambda 更难收敛（R1），mixing_beta 降到 0.3
    MB=$(awk -v l="$lam" 'BEGIN{print (l<0)?"0.3":"0.4"}')
    cat > "$d/INPUT" <<EOF
INPUT_PARAMETERS
suffix          autotest
calculation     scf
basis_type      lcao
ecutwfc         100
gamma_only      0
nspin           1
scf_thr         1.0e-8
scf_nmax        200
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     $MB
ks_solver       genelpa
symmetry        0
deltap_switch   1
deltap_corr     1
deltap_gdir     3
deltap_rm       6.0
onsite_radius   6.0
deltap_lambda_init   $lam
deltap_lambda_step   0.0
deltap_lambda_mixing 0.1
deltap_inner_thr     1.0e-2
deltap_constraint_mode total
cal_force       1
pseudo_dir  $PSEUDO_DIR
orbital_dir $ORBITAL_DIR
EOF
    echo "  [run] lambda=$lam (mixing_beta=$MB)"
    run_abacus "$d"
    GG[$lam]=$(get_rawg "$d")
    IT[$lam]=$(get_scfiter "$d")
    FZ[$lam]=$(get_oforce_z "$d")
    if grep -q "SCF IS NOT CONVERGED" "$d/run.log" 2>/dev/null; then
        ST[$lam]="INVALID"
    elif [ -z "${GG[$lam]}" ]; then
        ST[$lam]="INVALID"
    else
        ST[$lam]="VALID"
    fi
    echo "$lam  ${GG[$lam]:-NA}  ${IT[$lam]:-NA}  ${FZ[$lam]:-NA}  ${ST[$lam]}" >> "$DAT"
done

echo "===== 扫描结果（存 $DAT，force 列供 P03 复用） =====" | tee -a "$RESULTS"
cat "$DAT" | tee -a "$RESULTS"

# λ_crit 候选：最小的 |lambda| 使该点 INVALID
LCRIT=$(awk '$5=="INVALID"{a=($1<0)?-$1:$1; if(m==""||a<m)m=a} END{print (m=="")?"none":m}' "$DAT")
echo "  λ_crit 候选（最小 |λ| INVALID 点）: $LCRIT" | tee -a "$RESULTS"

# 最小二乘分析（仅 VALID 点）：全窗口 + 子窗口(|λ|<=0.02)
read S_FULL R2_FULL S_SUB <<< $(awk '$5=="VALID"{
    x=$1; y=$2; n++; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; syy+=y*y;
    if (x>=-0.02 && x<=0.02) { m++; tx+=x; ty+=y; txx+=x*x; txy+=x*y }
} END{
    if (n<3 || m<3) { print "NA NA NA"; exit }
    sf=(n*sxy-sx*sy)/(n*sxx-sx*sx);
    ss=(m*txy-tx*ty)/(m*txx-tx*tx);
    r=(n*sxy-sx*sy)/sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
    printf "%.8f %.8f %.8f", sf, r*r, ss}')
echo "  全窗口斜率 = $S_FULL, 全线性 R^2 = $R2_FULL, 子窗口(|λ|<=0.02)斜率 = $S_SUB" | tee -a "$RESULTS"

# 反对称偏差 max |γ(λ)+γ(−λ)| / |γ(λ)−γ(−λ)|
ASMAX=0
for lp in 0.01 0.02 0.04 0.08; do
    lm="-$lp"
    [ "${ST[$lp]}" = "VALID" ] && [ "${ST[$lm]}" = "VALID" ] || continue
    a=$(awk -v gp="${GG[$lp]}" -v gm="${GG[$lm]}" 'BEGIN{d=gp-gm; if(d<0)d=-d; s=gp+gm; if(s<0)s=-s; printf "%.6f",(d>1e-12)?s/d:0}')
    echo "  反对称偏差 ±$lp: $a" | tee -a "$RESULTS"
    ASMAX=$(awk -v a="$a" -v m="$ASMAX" 'BEGIN{print (a>m)?a:m}')
done

SDIFF=$(awk -v f="$S_FULL" -v s="$S_SUB" 'BEGIN{if(f=="NA"||s=="NA"||f==0){print 999}else{d=f-s; if(d<0)d=-d; g=f; if(g<0)g=-g; printf "%.6f", d/g}}')
echo "  子窗口/全窗口斜率差 = $SDIFF (要求 <= 0.03)" | tee -a "$RESULTS"

NP=0; NT=0
judge() {
    NT=$((NT+1))
    if [ "$2" = "1" ]; then echo "PASS: $1" | tee -a "$RESULTS"; NP=$((NP+1))
    else echo "FAIL: $1" | tee -a "$RESULTS"; fi
}
OK1=$(awk -v r="$R2_FULL" 'BEGIN{print (r!="NA" && r>=0.98)?1:0}')
OK2=$(awk -v a="$ASMAX" 'BEGIN{print (a<0.05)?1:0}')
OK3=$(awk -v d="$SDIFF" 'BEGIN{print (d<=0.03)?1:0}')
judge "全线性 R^2 >= 0.98 ($R2_FULL)" "$OK1"
judge "反对称偏差 max < 5% ($ASMAX)" "$OK2"
judge "子窗口斜率差 <= 3% ($SDIFF)" "$OK3"
echo "SUMMARY: $NP/$NT PASS" | tee -a "$RESULTS"
[ "$NP" = "$NT" ] && exit 0 || exit 1
