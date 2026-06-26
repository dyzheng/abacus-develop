#!/bin/bash
# ============================================================
# ABACUS CG vs CG_MIXED 对比测试（通用版）
# 
# 用法:
#   bash run_compare.sh                     # 编译/运行所有结果
#   bash run_compare.sh build               # 仅编译
#   bash run_compare.sh test                 # 仅测试 (复用上次编译)
#
# 目录结构要求 (你自己建):
#   tests/
#   ├── 003_4MoS2/
#   │   ├── INPUT          (你用旧代码跑过的, ks_solver 随便填)
#   │   ├── STRU
#   │   ├── KPT
#   │   └── *.upf          (赝势文件)
#   └── 006_16Na/
#       ├── INPUT
#       ├── STRU
#       ├── KPT
#       └── *.upf
#
# 脚本会自动:
#   1. 编译 ABACUS (含 cg_mixed)
#   2. 拷贝每个测试目录 → 用 cg 跑一遍
#   3. 修改 ks_solver 为 cg_mixed → 再跑一遍
#   4. 提取总能量和耗时对比
# ============================================================
set -e

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
NPROC=${NPROC:-4}
ABACUS_BIN=""
RESULTS_FILE="comparison_results.txt"

echo "============================================"
echo "  ABACUS CG vs CG_MIXED 对比测试"
echo "  NP=$NPROC  NT=$OMP_NUM_THREADS"
echo "  $(date)"
echo "============================================"

# --------------------------------------------------
# 编译
# --------------------------------------------------
do_build() {
    if [ ! -d abacus-pr7417 ]; then
        echo "[编译] 克隆代码..."
        git clone -b feature/eigen-mixed-precision-cg-dev \
            https://github.com/Absolutely-Daisy/abacus-develop.git abacus-pr7417
    fi
    cd abacus-pr7417
    echo "[编译] cmake..."
    cmake -B build \
        -DCMAKE_CXX_COMPILER=mpiicpc \
        -DENABLE_LIBXC=ON -DENABLE_LIBRI=ON \
        2>&1 | tail -3
    echo "[编译] make -j..."
    cmake --build build -j$(nproc) 2>&1 | tail -5
    ABACUS_BIN=$(realpath build/source/source_main/abacus)
    echo "[编译] 完成: $ABACUS_BIN"
    cd ..
}

# --------------------------------------------------
# 运行单个测试 (在给定目录下跑, ks_solver 由参数指定)
# --------------------------------------------------
run_one() {
    local WORK_DIR=$1
    local SOLVER=$2
    local LABEL=$3

    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"

    # 修改 INPUT 中的 ks_solver (只改已有行, 不新增)
    if grep -q "^ks_solver" INPUT 2>/dev/null; then
        sed -i "s/^ks_solver.*/ks_solver ${SOLVER}/" INPUT
    else
        echo "ks_solver ${SOLVER}" >> INPUT
    fi

    echo ""
    echo ">>> [$LABEL] $SOLVER 开始 $(date +%H:%M:%S)"
    mpirun -np $NPROC $ABACUS_BIN 2>&1 | tee run.log
    echo "<<< [$LABEL] $SOLVER 结束 $(date +%H:%M:%S)"
    cd ..
}

# --------------------------------------------------
# 提取结果
# --------------------------------------------------
extract() {
    local DIR=$1
    local LABEL=$2
    local LOG=""

    # 找到 running_scf.log (suffix 可能不同)
    for d in $DIR/OUT.*/; do
        [ -f "$d/running_scf.log" ] && LOG="$d/running_scf.log" && break
    done

    if [ -z "$LOG" ]; then
        echo "  $LABEL: running_scf.log 未找到"
        return
    fi

    local ENERGY=$(grep "FINAL_ETOT_IS" "$LOG" | tail -1 | awk '{print $NF}')
    local SCF=$(grep -c "ETOT_IS" "$LOG" 2>/dev/null || grep -c "Total Time" "$LOG")
    local TIME=$(grep "TOTAL.*TIME" "$LOG" | tail -1 | awk '{print $NF}')
    
    # 尝试提取 hsolver 时间
    local HS_TIME=$(grep -i "hsolver\|diago.*time\|CG.*time" "$LOG" | head -3 | paste -sd ';')

    echo "  $LABEL | Energy= $ENERGY | SCF steps= $SCF | Total time= ${TIME}s"
    echo "  $LABEL | Hsolver details: $HS_TIME"
    echo ""
}

# --------------------------------------------------
# 主流程
# --------------------------------------------------
main() {
    local MODE=$1

    # 编译
    if [ "$MODE" != "test" ]; then
        do_build
    fi
    
    # 查找二进制
    if [ -f abacus-pr7417/build/source/source_main/abacus ]; then
        ABACUS_BIN=$(realpath abacus-pr7417/build/source/source_main/abacus)
    else
        echo "ERROR: 找不到 ABACUS 二进制, 先运行: $0 build"
        exit 1
    fi

    # 测试
    if [ "$MODE" != "build" ]; then
        echo ""
        echo "[测试] 开始对比..."
        > $RESULTS_FILE

        for TEST_DIR in tests/*/; do
            local NAME=$(basename "$TEST_DIR")
            echo ""
            echo "========================================"
            echo "  $NAME"
            echo "========================================"

            # 第一轮: 双精度 CG (基准)
            cp -r "$TEST_DIR" "run_${NAME}_cg"
            run_one "run_${NAME}_cg" cg "${NAME} CG"
            echo "======== $NAME ========" >> $RESULTS_FILE
            extract "run_${NAME}_cg" "${NAME}_CG (基准)" | tee -a $RESULTS_FILE

            # 第二轮: 混合精度 CG_MIXED
            cp -r "$TEST_DIR" "run_${NAME}_cg_mixed"
            run_one "run_${NAME}_cg_mixed" cg_mixed "${NAME} CG_MIXED"
            extract "run_${NAME}_cg_mixed" "${NAME}_CG_MIXED (混合)" | tee -a $RESULTS_FILE

            # 快速对比
            echo "  >>> 对比:"
            paste <(extract "run_${NAME}_cg" "${NAME}_CG") \
                  <(extract "run_${NAME}_cg_mixed" "${NAME}_CG_MIXED") 2>/dev/null || true
        done

        echo ""
        echo "============================================"
        echo "  全部测试完成: $(date)"
        echo "  结果已保存到: $RESULTS_FILE"
        echo "============================================"
    fi
}

main "$@"
