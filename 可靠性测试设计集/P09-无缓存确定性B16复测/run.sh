#!/bin/bash
# P09 无缓存确定性（B16 复测，无阻塞）
# 体系: H2O (Γ 点分子) + h-BN (4x4x2 k 点体材料)，LCAO DeltaP lambda=0 测量模式
# 每体系: OMP_NUM_THREADS=1 全新运行 3 次 + OMP=4、8 各 1 次（每次 rm -rf OUT.*）
# 判据: 5 次 max|Δγ| <= 1e-6 rad, max|ΔE| <= 1e-8 Ha
# 用法：./run.sh  （环境变量 ABACUS/PSEUDO_DIR/ORBITAL_DIR/NPROC 可覆盖）
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"

run_abacus() { # $1 目录, $2 OMP 线程数
    cd "$1"
    rm -rf OUT.*
    export OMP_NUM_THREADS="$2"
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 || true
    else
        "$ABACUS" > run.log 2>&1 || true
    fi
    unset OMP_NUM_THREADS
    cd "$BASEDIR"
}
get_rawg() { grep "rawG" "$1"/run.log 2>/dev/null | tail -1 | sed -E 's/.*Σγ_raw=([^ ]+).*/\1/'; }
get_eks() { grep "E_KohnSham" "$1"/OUT.*/running_scf.log 2>/dev/null | tail -1 | awk '{print $2}'; }

echo "======================================================"
echo " P09 无缓存确定性（B16 复测）"
echo " ABACUS=$ABACUS  NPROC=$NPROC"
echo "======================================================"
mkdir -p "$RUNS"
: > "$RESULTS"

# ---- 编译器/BLAS 信息（zgeev 排序对 LAPACK 实现敏感，必须记录） ----
{
echo "===== 环境信息（请随结果一并记录） ====="
echo "  date: $(date)"
echo "  ABACUS: $ABACUS"
"$ABACUS" --version 2>/dev/null | head -3 || true
echo "  --- linked BLAS/LAPACK ---"
ldd "$ABACUS" 2>/dev/null | grep -iE "blas|lapack|mkl|scalapack|elpa" || echo "  (ldd 无匹配或静态链接)"
echo "  --- compilers ---"
gcc --version 2>/dev/null | head -1 || true
g++ --version 2>/dev/null | head -1 || true
mpirun --version 2>/dev/null | head -1 || true
} | tee -a "$RESULTS"

gen_input() { # $1 目标文件
    cat > "$1" <<EOF
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
}

NP=0; NT=0
for SYS in h2o hbn; do
    echo "===== 体系: $SYS =====" | tee -a "$RESULTS"
    # 5 次运行: OMP=1 x3, OMP=4, OMP=8
    RUNLIST="omp1_a:1 omp1_b:1 omp1_c:1 omp4:4 omp8:8"
    GAMAS=""; ENES=""
    for item in $RUNLIST; do
        name="${item%%:*}"; omp="${item##*:}"
        d="${RUNS}/${SYS}/${name}"
        mkdir -p "$d"
        cp "${BASEDIR}/cases/${SYS}/STRU" "${BASEDIR}/cases/${SYS}/KPT" "$d/"
        gen_input "$d/INPUT"
        echo "  [run] $SYS/$name OMP_NUM_THREADS=$omp"
        run_abacus "$d" "$omp"
        g=$(get_rawg "$d"); e=$(get_eks "$d")
        # E_KohnSham 单位 Ry，换算 Ha = Ry/2 后进入判据
        eha=$(awk -v e="$e" 'BEGIN{printf "%.12f", e/2}')
        echo "    gamma_raw=${g:-MISSING}  E_KohnSham=${e:-MISSING} Ry ($eha Ha)" | tee -a "$RESULTS"
        GAMAS="$GAMAS $g"; ENES="$ENES $eha"
    done
    # max|Δγ| 与 max|ΔE|（相对 5 次均值）
    GAMAS="${GAMAS# }"; ENES="${ENES# }"
    DG=$(awk 'BEGIN{n=split("'"$GAMAS"'",v," "); s=0; for(i=1;i<=n;i++)s+=v[i]; m=s/n;
        d=0; for(i=1;i<=n;i++){x=v[i]-m; if(x<0)x=-x; if(x>d)d=x} printf "%.3e", d}')
    DE=$(awk 'BEGIN{n=split("'"$ENES"'",v," "); s=0; for(i=1;i<=n;i++)s+=v[i]; m=s/n;
        d=0; for(i=1;i<=n;i++){x=v[i]-m; if(x<0)x=-x; if(x>d)d=x} printf "%.3e", d}')
    echo "  $SYS: max|Δγ| = $DG rad (<=1e-6), max|ΔE| = $DE Ha (<=1e-8)" | tee -a "$RESULTS"
    OKG=$(awk -v d="$DG" 'BEGIN{print (d<=1e-6)?1:0}')
    OKE=$(awk -v d="$DE" 'BEGIN{print (d<=1e-8)?1:0}')
    NT=$((NT+1)); if [ "$OKG" = "1" ]; then echo "PASS: $SYS max|Δγ| <= 1e-6 rad ($DG)" | tee -a "$RESULTS"; NP=$((NP+1)); else echo "FAIL: $SYS max|Δγ| <= 1e-6 rad ($DG)" | tee -a "$RESULTS"; fi
    NT=$((NT+1)); if [ "$OKE" = "1" ]; then echo "PASS: $SYS max|ΔE| <= 1e-8 Ha ($DE)" | tee -a "$RESULTS"; NP=$((NP+1)); else echo "FAIL: $SYS max|ΔE| <= 1e-8 Ha ($DE)" | tee -a "$RESULTS"; fi
done

echo "SUMMARY: $NP/$NT PASS" | tee -a "$RESULTS"
[ "$NP" = "$NT" ] && exit 0 || exit 1
