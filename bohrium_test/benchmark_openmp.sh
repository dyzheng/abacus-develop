#!/bin/bash
# benchmark_openmp.sh — OpenMP 多线程加速比基准测试
# 用法: ./benchmark_openmp.sh [算例名] [最大线程数]

set -e

CASE="${1:-001_4GaAs}"
MAX_THREADS="${2:-8}"
ABACUS_BIN="/abacus-develop/build/abacus"
WORKDIR="/abacus-develop/bohrium_test"
TESTDIR="${WORKDIR}/${CASE}"

echo "=============================================="
echo " OpenMP 多线程加速比基准测试"
echo " 算例: ${CASE}"
echo " 最大线程数: ${MAX_THREADS}"
echo " 开始时间: $(date)"
echo "=============================================="

if [ ! -f "${ABACUS_BIN}" ]; then
    echo "错误: ABACUS 可执行文件不存在: ${ABACUS_BIN}"
    echo "请先编译: cd /abacus-develop/build && cmake .. && make -j"
    exit 1
fi

if [ ! -d "${TESTDIR}" ]; then
    echo "错误: 测试目录不存在: ${TESTDIR}"
    exit 1
fi

# 检查 OpenMP 支持
echo ""
echo ">>> 检查 OpenMP 编译状态..."
if ldd "${ABACUS_BIN}" 2>/dev/null | grep -q libomp; then
    echo "✓ OpenMP 动态库已链接"
elif readelf -s "${ABACUS_BIN}" 2>/dev/null | grep -q "omp_get_num_threads"; then
    echo "✓ OpenMP 符号已链接（静态）"
elif strings "${ABACUS_BIN}" 2>/dev/null | grep -q "_OPENMP"; then
    echo "⚠ OpenMP 可能已链接但方式不确定"
else
    echo "✗ OpenMP 未检测到！请检查 CMake 配置: cmake .. -DUSE_OPENMP=ON"
fi

# 确认 OMP 可用
if [ -n "${OMP_NUM_THREADS}" ]; then
    echo "  当前 OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi

# 基线测试 (单线程)
echo ""
echo "=============================================="
echo " 基线测试: OMP_NUM_THREADS=1"
echo "=============================================="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${TESTDIR}"
BASELINE_START=$(date +%s)
mpirun -np 1 "${ABACUS_BIN}" > "${WORKDIR}/log_omp1_${CASE}.out" 2>&1 || true
BASELINE_END=$(date +%s)
BASELINE_TIME=$((BASELINE_END - BASELINE_START))

echo "基线耗时: ${BASELINE_TIME}s"

# 提取 Hsolver 时间
grep -i "hsolver\|diag.*time\|eigenvalue" "${WORKDIR}/log_omp1_${CASE}.out" | tail -5 || true

RESULTS_FILE="${WORKDIR}/omp_benchmark_${CASE}.txt"
echo "# OpenMP 基准测试结果 - $(date)" > "${RESULTS_FILE}"
echo "# 算例: ${CASE}" >> "${RESULTS_FILE}"
echo "# 线程数 | 总耗时(s) | 加速比 | Hsolver耗时(s)" >> "${RESULTS_FILE}"
echo "${BASELINE_TIME} 1 1.00" >> "${RESULTS_FILE}"

# 多线程测试
PREV_TIME="${BASELINE_TIME}"
for nthread in $(seq 2 "${MAX_THREADS}"); do
    echo ""
    echo "=============================================="
    echo " 测试: OMP_NUM_THREADS=${nthread}"
    echo "=============================================="
    export OMP_NUM_THREADS="${nthread}"
    export MKL_NUM_THREADS="${nthread}"
    export OPENBLAS_NUM_THREADS="${nthread}"

    START_TIME=$(date +%s)
    mpirun -np 1 "${ABACUS_BIN}" > "${WORKDIR}/log_omp${nthread}_${CASE}.out" 2>&1 || true
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    SPEEDUP=$(echo "scale=2; ${BASELINE_TIME} / ${ELAPSED}" | bc)
    echo "耗时: ${ELAPSED}s | 加速比: ${SPEEDUP}x"

    # 提取 Hsolver 耗时
    grep -i "hsolver\|diag.*time" "${WORKDIR}/log_omp${nthread}_${CASE}.out" | tail -3 || true

    echo "${ELAPSED} ${nthread} ${SPEEDUP}" >> "${RESULTS_FILE}"
    PREV_TIME="${ELAPSED}"
done

echo ""
echo "=============================================="
echo " 测试完成"
echo " 结果文件: ${RESULTS_FILE}"
echo "=============================================="
cat "${RESULTS_FILE}"
