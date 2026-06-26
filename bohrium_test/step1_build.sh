#!/bin/bash
# ============================================================
# 复制粘贴到玻尔终端，回车即跑
# 第一步：编译 ABACUS (含 cg_mixed)
# ============================================================
set -e
echo ">>> 开始编译 ABACUS..."

# 克隆 PR 分支
git clone -b feature/eigen-mixed-precision-cg-dev \
    https://github.com/Absolutely-Daisy/abacus-develop.git abacus-pr7417
cd abacus-pr7417

# 编译
cmake -B build \
    -DCMAKE_CXX_COMPILER=mpiicpc \
    -DENABLE_LIBXC=ON -DENABLE_LIBRI=ON
cmake --build build -j$(nproc)

# 确认二进制存在
ls -lh build/source/source_main/abacus
echo ">>> 编译完成!"
pwd
cd ..
