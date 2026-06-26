#!/bin/bash
# ============================================================
# ABACUS 混合精度 CG 测试脚本
# 一键完成: 编译 → 运行 cg → 运行 cg_mixed → 对比结果
#
# 用法: bash run_all.sh /path/to/pseudopotentials/
# 参数: 赝势文件所在目录（包含 Si_ONCV_PBE-1.0.upf 等）
# ============================================================
set -e

PSEUDO_DIR="${1:-.}"           # 赝势目录
NPROC_NP=4                      # MPI 进程数
NPROC_NT=4                      # OpenMP 线程数
export OMP_NUM_THREADS=$NPROC_NT

echo "============================================"
echo "  ABACUS CG vs CG_MIXED 对比测试"
echo "  进程数: $NPROC_NP, 线程数: $NPROC_NT"
echo "  开始时间: $(date)"
echo "============================================"

# --------------------------------------------------
# 一、编译 ABACUS（含 cg_mixed 支持）
# --------------------------------------------------
if [ ! -d abacus-pr7417 ]; then
    echo "[1/4] 克隆代码..."
    git clone -b feature/eigen-mixed-precision-cg-dev \
        https://github.com/Absolutely-Daisy/abacus-develop.git abacus-pr7417
fi

cd abacus-pr7417
echo "[2/4] 编译 ABACUS (约 5-10 分钟)..."
cmake -B build \
    -DCMAKE_CXX_COMPILER=mpiicpc \
    -DENABLE_LIBXC=ON \
    -DENABLE_LIBRI=ON \
    2>&1 | tail -5
cmake --build build -j$(nproc) 2>&1 | tail -10
ABACUS=$(realpath build/source/source_main/abacus)
echo "编译完成: $ABACUS"
cd ..

# --------------------------------------------------
# 二、准备测试输入文件
# --------------------------------------------------
echo "[3/4] 准备测试输入..."

# --- Si 金刚石(8原子) ---
prepare_si() {
    local SUFFIX=$1
    local SOLVER=$2
    mkdir -p test_${SUFFIX}
    cd test_${SUFFIX}

    # STRU
    cat > STRU << 'STRU_EOF'
ATOMIC_SPECIES
Si 28.085 Si_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
1.8897261258369282

LATTICE_VECTORS
5.4307000000 0.0000000000 0.0000000000
0.0000000000 5.4307000000 0.0000000000
0.0000000000 0.0000000000 5.4307000000

ATOMIC_POSITIONS Direct
Si
0.0000000000
8
0.0000000000 0.0000000000 0.0000000000 1 1 1 mag 0.0
0.0000000000 0.5000000000 0.5000000000 1 1 1 mag 0.0
0.5000000000 0.0000000000 0.5000000000 1 1 1 mag 0.0
0.5000000000 0.5000000000 0.0000000000 1 1 1 mag 0.0
0.7500000000 0.7500000000 0.2500000000 1 1 1 mag 0.0
0.7500000000 0.2500000000 0.7500000000 1 1 1 mag 0.0
0.2500000000 0.7500000000 0.7500000000 1 1 1 mag 0.0
0.2500000000 0.2500000000 0.2500000000 1 1 1 mag 0.0
STRU_EOF

    # KPT
    cat > KPT << 'KPT_EOF'
K_POINTS
0
Gamma
6 6 6 0 0 0
KPT_EOF

    # INPUT
    cat > INPUT << INPUT_EOF
#Parameters (1.General)
suffix ${SUFFIX}
calculation scf
symmetry 1
pseudo_dir ${PSEUDO_DIR}
basis_type pw
ecutwfc 60

#Parameters (2.SCF)
scf_nmax 100
scf_thr 1e-8

#Parameters (3.KS solver)
nbands 26
ks_solver ${SOLVER}

#Parameters (4.Smearing)
smearing_method gauss
smearing_sigma 0.01

#Parameters (5.Mixing)
mixing_type broyden
mixing_beta 0.7
mixing_gg0 0
INPUT_EOF

    # 拷贝赝势
    cp ${PSEUDO_DIR}/Si_ONCV_PBE-1.0.upf . 2>/dev/null || \
        echo "WARNING: Si_ONCV_PBE-1.0.upf not found in $PSEUDO_DIR"

    cd ..
}

prepare_si Si_cg       cg
prepare_si Si_cg_mixed cg_mixed

# --------------------------------------------------
# 三、运行测试
# --------------------------------------------------
echo "[4/4] 运行对比测试..."

run_test() {
    local DIR=$1
    local LABEL=$2
    echo ""
    echo ">>> 运行 $LABEL ..."
    cd $DIR
    mpirun -np $NPROC_NP $ABACUS 2>&1 | tee run.log
    cd ..
    echo "<<< $LABEL 完成"
}

run_test test_Si_cg       "Si 双精度 CG"
run_test test_Si_cg_mixed "Si 混合精度 CG_MIXED"

# --------------------------------------------------
# 四、提取对比结果
# --------------------------------------------------
echo ""
echo "============================================"
echo "  结果对比"
echo "============================================"

extract_info() {
    local DIR=$1
    local LABEL=$2
    local LOG=$(ls $DIR/OUT.*/running_scf.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        echo "$LABEL: 未找到 running_scf.log"
        return
    fi
    echo ""
    echo "--- $LABEL ---"
    echo "总能量:"
    grep "FINAL_ETOT_IS" "$LOG" | tail -1 | awk '{printf "  %s\n", $0}'
    echo "SCF 收敛步数:"
    grep -c "ITER" "$LOG" | awk '{printf "  %d 步\n", $1}'
    echo "各求解器耗时 (秒):"
    grep -i "CG.*time\|Diago.*time\|diag.*time\|hsolver.*time" "$LOG" | head -5
    echo "总运行时间:"
    grep "TOTAL.*TIME\|total.*time" "$LOG" | tail -1 | awk '{printf "  %s\n", $0}'
}

extract_info test_Si_cg       "CG (双精度基线)"
extract_info test_Si_cg_mixed "CG_MIXED (混合精度)"

echo ""
echo "============================================"
echo "  测试完成: $(date)"
echo "============================================"
