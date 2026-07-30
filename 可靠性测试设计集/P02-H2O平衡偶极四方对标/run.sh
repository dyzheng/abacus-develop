#!/bin/bash
# P02 H2O 平衡偶极四方对标（无阻塞）
# 通道1: LCAO DeltaP lambda=0 测量模式 (gdir=3) -> Σγ_raw -> mu
# 通道2: PW berry_phase=1 gdir=1/2/3 -> γ_total -> mu
# 通道3: wannier90 MLWF 中心（未安装则 SKIP）
# 通道4: 实验值 1.855 D
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

# ============ 常量区 ============
A_BOHR=30.0                 # 盒边长 (Bohr)
PI=$(awk 'BEGIN{print atan2(0,-1)}')
DEBYE_PER_EBOHR=2.541746    # 1 e·Bohr = 2.541746 D (= 1/0.393430)
MU_EXP=1.855                # 实验偶极 (D)
# 偶极换算: mu(e·Bohr) = (A_BOHR/PI) * unwrap(gamma) / F2_SPIN；再乘 DEBYE_PER_EBOHR 得 Debye
# unwrap: 将 gamma 折叠到 (-pi, pi]（raw γ 为 unwrapped 多圈值，实测 H2O Σγ_raw≈-12.718 rad）
F2_SPIN=2.0    # nspin=1 自旋因子。冒烟实测(2026-07-30): Σγ_raw=-12.718 -> unwrap -0.1517
               # -> /2 -> mu=1.841 D (实验 1.855 D)，吻合；待备忘录定稿确认
# 假设: 代码输出的 gamma 已含离子点电荷项（待 P02 实跑确认，见 README）
# ===============================

unwrap_gamma() { awk -v g="$1" 'BEGIN{pi=atan2(0,-1); n=g/(2*pi); n=(n>=0)?int(n+0.5):int(n-0.5); printf "%.10f", g-2*pi*n}'; }

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
get_gtot() { grep "DeltaP-PW" "$1"/run.log 2>/dev/null | grep "γ_total" | tail -1 | sed -E 's/.*γ_total=([^ ]+).*/\1/'; }

echo "======================================================"
echo " P02 H2O 平衡偶极四方对标"
echo " ABACUS=$ABACUS  NPROC=$NPROC"
echo "======================================================"
mkdir -p "$RUNS"
: > "$RESULTS"

# ---------------- 通道1: LCAO DeltaP lambda=0 (gdir=3) ----------------
echo "===== 通道1: LCAO DeltaP 测量模式 (lambda=0, gdir=3) =====" | tee -a "$RESULTS"
d="${RUNS}/lcao_gdir3"
mkdir -p "$d"
cp "$CASES/STRU" "$CASES/KPT" "$d/"
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
mixing_beta     0.4
ks_solver       genelpa
symmetry        0
deltap_switch   1
deltap_corr     1
deltap_gdir     3
deltap_rm       6.0
onsite_radius   6.0
deltap_lambda_init   0.0
deltap_lambda_step   0.0
deltap_lambda_mixing 0.1
deltap_inner_thr     1.0e-2
deltap_constraint_mode total
pseudo_dir  $PSEUDO_DIR
orbital_dir $ORBITAL_DIR
EOF
run_abacus "$d"
G_LCAO=$(get_rawg "$d")
GU_LCAO=$(unwrap_gamma "${G_LCAO:-0}")
MU_LCAO=$(awk -v g="$GU_LCAO" -v a="$A_BOHR" -v pi="$PI" -v c="$DEBYE_PER_EBOHR" -v f="$F2_SPIN" \
    'BEGIN{m=a/pi*g/f; if(m<0)m=-m; printf "%.6f", m*c}')
echo "  Σγ_raw = ${G_LCAO:-MISSING} rad -> mu_LCAO = $MU_LCAO D" | tee -a "$RESULTS"

# ---------------- 通道2: PW berry_phase gdir=1/2/3 ----------------
echo "===== 通道2: PW berry_phase=1 (gdir=1/2/3) =====" | tee -a "$RESULTS"
declare -A G_PW MU_PW
for gd in 1 2 3; do
    d="${RUNS}/pw_gdir${gd}"
    mkdir -p "$d"
    cp "$CASES/STRU" "$d/"
    mesh="1 1 2"; [ "$gd" = "1" ] && mesh="2 1 1"; [ "$gd" = "2" ] && mesh="1 2 1"
    printf 'K_POINTS\n0\nGamma\n%s 0 0 0\n' "$mesh" > "$d/KPT"
    cat > "$d/INPUT" <<EOF
INPUT_PARAMETERS
suffix          autotest
calculation     scf
basis_type      pw
ecutwfc         80
gamma_only      0
nspin           1
scf_thr         1.0e-8
scf_nmax        200
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     0.4
ks_solver       cg
symmetry        0
berry_phase     1
gdir            $gd
deltap_switch   true
deltap_corr     1
deltap_gdir     $gd
deltap_rm       6.0
onsite_radius   6.0
deltap_lambda_init   0.0
deltap_lambda_step   0.0
deltap_lambda_mixing 0.1
deltap_inner_thr     1.0e-2
deltap_constraint_mode total
pseudo_dir  $PSEUDO_DIR
orbital_dir $ORBITAL_DIR
EOF
    run_abacus "$d"
    G_PW[$gd]=$(get_gtot "$d")
    GU_PW=$(unwrap_gamma "${G_PW[$gd]:-0}")
    MU_PW[$gd]=$(awk -v g="$GU_PW" -v a="$A_BOHR" -v pi="$PI" -v c="$DEBYE_PER_EBOHR" -v f="$F2_SPIN" \
        'BEGIN{m=a/pi*g/f; if(m<0)m=-m; printf "%.6f", m*c}')
    echo "  gdir=$gd: γ_total = ${G_PW[$gd]:-MISSING} rad -> mu = ${MU_PW[$gd]} D" | tee -a "$RESULTS"
done
MU_PWZ="${MU_PW[3]}"

# ---------------- 通道3: wannier90 ----------------
echo "===== 通道3: wannier90 MLWF 中心 =====" | tee -a "$RESULTS"
if command -v wannier90 >/dev/null 2>&1; then
    echo "  检测到 wannier90: $(command -v wannier90)" | tee -a "$RESULTS"
    echo "  SKIP: wannier90 后处理通道需手动对接（本脚本不自动执行），结果请手工比对" | tee -a "$RESULTS"
else
    echo "  SKIP: 未检测到 wannier90" | tee -a "$RESULTS"
fi

# ---------------- 判定 ----------------
echo "===== 判定 =====" | tee -a "$RESULTS"
echo "  mu_LCAO = $MU_LCAO D, mu_PW(z) = $MU_PWZ D, mu_exp = $MU_EXP D" | tee -a "$RESULTS"
NP=0; NT=0
judge() {
    NT=$((NT+1))
    if [ "$2" = "1" ]; then echo "PASS: $1" | tee -a "$RESULTS"; NP=$((NP+1))
    else echo "FAIL: $1" | tee -a "$RESULTS"; fi
}
D_LP=$(awk -v a="$MU_LCAO" -v b="$MU_PWZ" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6f", d}')
D_LE=$(awk -v a="$MU_LCAO" -v b="$MU_EXP" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6f", d}')
D_PE=$(awk -v a="$MU_PWZ" -v b="$MU_EXP" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6f", d}')
OK1=$(awk -v d="$D_LP" 'BEGIN{print (d<=0.02)?1:0}')
OK2=$(awk -v d="$D_LE" 'BEGIN{print (d<=0.05)?1:0}')
OK3=$(awk -v d="$D_PE" 'BEGIN{print (d<=0.05)?1:0}')
judge "LCAO vs PW 互差 <= 0.02 D (=$D_LP)" "$OK1"
judge "LCAO vs 实验 <= 0.05 D (=$D_LE)" "$OK2"
judge "PW vs 实验 <= 0.05 D (=$D_PE)" "$OK3"
echo "SUMMARY: $NP/$NT PASS" | tee -a "$RESULTS"
[ "$NP" = "$NT" ] && exit 0 || exit 1
