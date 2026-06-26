#!/bin/bash
# ============================================================
# 复制粘贴到玻尔终端，回车即跑
# 第二步：运行 cg vs cg_mixed 对比
#
# 前提：step1 编译完成，且你把之前的测试文件放在了 tests/ 下
# 目录结构:
#   tests/
#   ├── 003_4MoS2/    ← 你的 INPUT, STRU, KPT, 赝势
#   └── 006_16Na/     ← 你的 INPUT, STRU, KPT, 赝势
# ============================================================
set -e

ABACUS=$(realpath abacus-pr7417/build/source/source_main/abacus)
NP=${NP:-4}
NT=${NT:-4}
export OMP_NUM_THREADS=$NT

echo "============================================"
echo "  ABACUS CG vs CG_MIXED 对比"
echo "  NP=$NP NT=$NT"
echo "============================================"

run_one() {
    local CASE=$1   # e.g. 003_4MoS2
    local SOLVER=$2 # cg or cg_mixed
    local LABEL=$3

    echo ""
    echo "================================================"
    echo "  $LABEL"
    echo "================================================"

    local WORK="run_${CASE}_${SOLVER}"
    rm -rf $WORK
    cp -r tests/$CASE $WORK
    cd $WORK

    # 改 ks_solver
    sed -i "s/^ks_solver.*/ks_solver ${SOLVER}/" INPUT

    echo "开始: $(date +%H:%M:%S)"
    mpirun -np $NP $ABACUS 2>&1 | tee run.log
    echo "结束: $(date +%H:%M:%S)"
    cd ..
}

show_result() {
    local CASE=$1
    local SOLVER=$2
    local LABEL=$3
    local WORK="run_${CASE}_${SOLVER}"
    local LOG=$(ls $WORK/OUT.*/running_scf.log 2>/dev/null | head -1)

    echo ""
    echo "--- $LABEL ---"
    if [ -z "$LOG" ]; then
        echo "  (running_scf.log 未找到)"
        return
    fi
    echo "  总能量: $(grep 'FINAL_ETOT_IS' $LOG | tail -1 | awk '{print $NF}')"
    echo "  总耗时: $(grep 'TOTAL.*TIME' $LOG | tail -1 | awk '{print $NF}')s"
    echo "  SCF步数: $(grep -c 'ETOT_IS' $LOG)"
    echo "  求解器时间:"
    grep -i "diago\|CG.*time\|hsolver.*time\|Time.*diag" $WORK/run.log | head -5 | sed 's/^/    /'
}

# === 003_4MoS2 ===
run_one   003_4MoS2 cg       "4MoS2: CG (双精度基线)"
run_one   003_4MoS2 cg_mixed "4MoS2: CG_MIXED (混合精度)"

# === 006_16Na ===
run_one   006_16Na cg       "16Na: CG (双精度基线)"
run_one   006_16Na cg_mixed "16Na: CG_MIXED (混合精度)"

# === 汇总 ===
echo ""
echo "============================================"
echo "  结果汇总"
echo "============================================"
for CASE in 003_4MoS2 006_16Na; do
    echo ""
    echo ">>>> $CASE <<<<"
    show_result $CASE cg       "CG (基准)"
    show_result $CASE cg_mixed "CG_MIXED"
done
echo ""
echo "全部完成: $(date)"
