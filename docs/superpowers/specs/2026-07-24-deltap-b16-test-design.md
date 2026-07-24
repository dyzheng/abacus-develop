# B16 确定性测试设计

> 日期: 2026-07-24
> 目的: 验证分支选择修复后, 相同输入全新运行 3 次是否得到完全一致的极化值

---

## 1. 问题定义

**B16**: 同一结构、同一份输入、删掉缓存文件后全新运行 3 次, 得到的极化值完全不同.

历史测试结果 (均未通过):

| 日期 | 条件 | 结果 |
|------|------|------|
| 07-13 | mixing_beta=0, 冻结电荷, 单进程 | γ₀ = +0.098, −0.062, −0.098 |
| 07-19 | mixing_beta=0, 冻结电荷, 单进程 | Px = −7.34e-3, 3.51e-3, −1.44e-4 |

验收标准: 冻结电荷下 3 次运行 **max|Δγ| < 1e-10**.

---

## 2. 测试体系与文件

**体系**: c-BN 原胞 (2 原子), zinc blende, a = 3.615 Bohr

**输入文件**: 复用 `tests/deltap_bn_test/` 中的 STRU, KPT, 赝势/轨道. 需要新建两个 INPUT:

### 2.1 INPUT 文件

#### 测试 A: 冻结电荷 (决定性测试)

```
INPUT_PARAMETERS
suffix              bn
calculation         scf
basis_type          lcao
ecutwfc             100
gamma_only          0
nspin               1
scf_thr             1.0e-8
scf_nmax            60
out_chg             1
smearing_method     gauss
smearing_sigma      0.002

# 关键: 冻结电荷, 使哈密顿量每轮完全相同
mixing_type         broyden
mixing_beta         0.0

ks_solver           genelpa
symmetry            -1

# DeltaP
deltap_switch       1
deltap_corr         1
deltap_inner_nmax   0
deltap_lambda_init  0.0
deltap_lambda_step  0.01
deltap_lambda_mixing  0.1
deltap_inner_thr    1.0e-3
deltap_target_file  target.dat

pseudo_dir    /root/pporb/apns-pseudopotentials-v1
orbital_dir   /root/pporb/apns-orbitals-efficiency-v1
```

#### 测试 B: 完整 SCF (实际应用测试)

与测试 A 相同, 但改为:

```
mixing_beta         0.4
mixing_restart      5e-4
```

### 2.2 target.dat

```
4.00
3.50
```

### 2.3 复用文件

直接拷贝自 `tests/deltap_bn_test/`:
- `STRU`
- `KPT` (Gamma-centered 2x2x2)
- 赝势与轨道文件路径 (由 INPUT 中 `pseudo_dir` / `orbital_dir` 指定)

---

## 3. 缓存文件清单

每次运行前 **必须** 删除以下全部文件, 确保全新运行:

```bash
rm -f deltap_branch.dat
rm -f deltap_match.dat
rm -f deltap_branch_enum.dat
rm -f deltap_zeta_debug.dat
rm -rf OUT.bn/
```

| 文件 | 作用 | 影响 |
|------|------|------|
| `deltap_branch.dat` | 跨 SCF 迭代的分支参考值 (W_prev_) | 直接影响分支选择 |
| `deltap_match.dat` | 匈牙利算法匹配结果 | 直接影响本征值追踪 |
| `deltap_branch_enum.dat` | 诊断输出 (仅写不读) | 无功能影响, 清理避免混淆 |
| `deltap_zeta_debug.dat` | 诊断输出 (仅写不读) | 无功能影响, 清理避免混淆 |
| `OUT.bn/` | 电荷密度等输出 | 可能含 restart 文件影响 SCF |

---

## 4. 运行方式

### 4.1 编译

```bash
cd /root/abacus-develop/build
cmake -DENABLE_ASAN=0 ..
make -j$(nproc) abacus_basic_para
```

### 4.2 运行

**必须单进程运行**, 排除 MPI 归约序引入的非确定性:

```bash
cd /root/abacus-develop/tests/B16_test/test_A/
OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para > run.log 2>&1
```

### 4.3 执行脚本

```bash
#!/bin/bash
# run_b16_test.sh
set -e

ABACUS=/root/abacus-develop/build/abacus_basic_para
export OMP_NUM_THREADS=1

clean_cache() {
    rm -f deltap_branch.dat deltap_match.dat deltap_branch_enum.dat deltap_zeta_debug.dat
    rm -rf OUT.bn/
}

for test_dir in test_A test_B; do
    echo "=== Test $test_dir ==="
    for run in 1 2 3; do
        dir="${test_dir}/run${run}"
        mkdir -p "$dir"
        cp "${test_dir}/INPUT" "${test_dir}/STRU" "${test_dir}/KPT" "${test_dir}/target.dat" "$dir/"
        cd "$dir"
        clean_cache
        echo "  Run $run ..."
        $ABACUS > run.log 2>&1
        cd -
    done
done
```

目录结构:

```
tests/B16_test/
├── run_b16_test.sh
├── test_A/              # 冻结电荷
│   ├── INPUT
│   ├── STRU
│   ├── KPT
│   ├── target.dat
│   ├── run1/
│   ├── run2/
│   └── run3/
└── test_B/              # 完整 SCF
    ├── INPUT
    ├── STRU
    ├── KPT
    ├── target.dat
    ├── run1/
    ├── run2/
    └── run3/
```

---

## 5. 数据提取

### 5.1 从每次运行的 output.log 提取以下数据

#### (a) 每次 SCF 迭代的 per-atom gamma

```bash
grep '\[DeltaP\] iter=' OUT.bn/../run.log
```

提取格式:
```
[DeltaP] iter=<N> max|gamma-target>=<maxdev> g0=<γ_B> l0=<λ_B> g1=<γ_N> l1=<λ_N>
```

需要提取: **每次迭代** 的 g0 (γ_B) 和 g1 (γ_N).

#### (b) 最终极化值 P_total

```bash
grep 'P_total (DeltaP)' run.log
```

输出 3 行 (x/y/z 方向), 格式:
```
   P_total (DeltaP)  = (Px, Py, Pz)
```

#### (c) 总能量

```bash
grep 'final etot is' run.log
```

#### (d) 分支选择诊断

```bash
grep 'DeltaP branch-set' run.log
```

格式:
```
DeltaP branch-set: atom <I> rescaled=<val> selected=<val> prev=<val> delta=<val>
```

#### (e) 跨 string 一致性

```bash
grep 'BRANCH INCONSISTENT' run.log
```

若出现即为异常.

#### (f) 收敛信息

```bash
grep 'drho' run.log | tail -1
grep 'converged' run.log
```

### 5.2 提取脚本

```bash
#!/bin/bash
# extract_b16.sh
# Usage: bash extract_b16.sh <run_dir>

dir=$1
log="$dir/run.log"

echo "=== $dir ==="

# Final P_total
echo "--- P_total (DeltaP) ---"
grep 'P_total (DeltaP)' "$log"

# Final E_tot
echo "--- E_tot ---"
grep 'final etot is' "$log" | tail -1

# Per-iteration gamma (all iterations)
echo "--- Per-iteration gamma ---"
grep '\[DeltaP\] iter=' "$log"

# Branch selection events
echo "--- Branch selection ---"
grep 'DeltaP branch-set' "$log"

# Branch inconsistency warnings
echo "--- Branch inconsistency ---"
grep 'BRANCH INCONSISTENT' "$log" || echo "(none)"

# SCF convergence
echo "--- SCF convergence ---"
grep 'Electron convergence' "$log" || grep 'drho' "$log" | tail -1
```

---

## 6. 判定标准

### 6.1 测试 A: 冻结电荷 (决定性)

| 判据 | 条件 | 级别 |
|------|------|------|
| **A1: γ 跨运行一致** | 3 次运行每步 iter 的 \|γ_B(run_i) − γ_B(run_j)\| < 1e-10 且 \|γ_N(run_i) − γ_N(run_j)\| < 1e-10 | **PASS 必要条件** |
| **A2: P_total 跨运行一致** | 3 次运行 P_total 每个分量差 < 1e-10 | **PASS 必要条件** |
| **A3: E_tot 跨运行一致** | 3 次运行 E_tot 差 < 1e-12 | 验证哈密顿量一致性 |
| **A4: 无分支不一致警告** | 不出现 `BRANCH INCONSISTENT` | 验证分支选择正确性 |

### 6.2 测试 B: 完整 SCF

| 判据 | 条件 | 级别 |
|------|------|------|
| **B1: γ 跨运行一致** | 3 次运行最终 iter 的 \|Δγ\| < 1e-6 | PASS 目标 (SCF 收敛精度限制) |
| **B2: P_total 跨运行一致** | 3 次运行 P_total 每个分量差 < 1e-6 | PASS 目标 |
| **B3: E_tot 跨运行一致** | 3 次运行 E_tot 差 < 1e-8 | 验证 SCF 收敛到同一态 |
| **B4: 收敛轨迹一致** | 3 次运行每步 iter 的 γ 差 < 1e-6 | 验证无混沌发散 |
| **B5: 无分支跳变** | 不出现 `BRANCH INCONSISTENT`, 且 γ 不出现 > π 的跳变 | 验证分支连续性 |

### 6.3 汇总判定

| 结果 | 条件 |
|------|------|
| **B16 FIXED** | 测试 A 全部 PASS + 测试 B 至少 B1/B2/B5 PASS |
| **B16 PARTIAL** | 测试 A 全部 PASS 但测试 B 不通过 |
| **B16 OPEN** | 测试 A 任一必要条件不通过 |

---

## 7. 结果记录模板

### 7.1 测试 A 结果表

```
Run | iter | γ_B            | γ_N            | E_tot          | P_x           | P_y           | P_z
----|------|----------------|----------------|----------------|---------------|---------------|---------------
 1  |  1   |                |                |                |               |               |
 1  |  2   |                |                |                |               |               |
 1  | ...  |                |                |                |               |               |
 1  | 最后 |                |                |                |               |               |
 2  |  1   |                |                |                |               |               |
 2  | ...  |                |                |                |               |               |
 2  | 最后 |                |                |                |               |               |
 3  |  1   |                |                |                |               |               |
 3  | ...  |                |                |                |               |               |
 3  | 最后 |                |                |                |               |               |

跨运行最大偏差:
  max|Δγ_B| =
  max|Δγ_N| =
  max|ΔP_x| =
  max|ΔP_y| =
  max|ΔP_z| =
  max|ΔE|   =

判定: □ PASS  □ FAIL
```

### 7.2 测试 B 结果表

```
Run | 最后 iter | γ_B     | γ_N     | E_tot          | P_x           | P_y           | P_z
----|-----------|---------|---------|----------------|---------------|---------------|---------------
 1  |           |         |         |                |               |               |
 2  |           |         |         |                |               |               |
 3  |           |         |         |                |               |               |

跨运行最大偏差:
  max|Δγ_B| =
  max|Δγ_N| =
  max|ΔP_x| =
  max|ΔP_y| =
  max|ΔP_z| =
  max|ΔE|   =

判定: □ PASS  □ FAIL
```

---

## 8. 故障诊断

如果测试不通过, 按以下流程排查:

### 8.1 检查累加器隔离

检查 `deltap_wannier.cpp` 中 `gamma_accum`, `n_strings_processed` 是否在 `for (int alpha = 0; ...)` 循环 **内部** 声明. 如果在外部声明, 三个方向共享累加器.

```bash
grep -n 'gamma_accum\|n_strings_processed' source/source_lcao/module_deltap/deltap_wannier.cpp | head -10
```

正确: 行号 349-355 应在 `alpha` 循环体内部.

### 8.2 检查分支状态按方向存储

检查 `W_prev_` 是否为 `Vector3<double>` 类型 (每个原子存储 x/y/z 三个分量). 如果是单个 `double`, 则三个方向共享分支参考.

```bash
grep -n 'W_prev_' source/source_lcao/module_deltap/deltap.h
```

### 8.3 检查匹配冻结

确认 `deltap_match.dat` 在第一次写入后, 后续运行能正确加载:

```bash
grep 'loaded eigenvalue matching' run.log
```

若 3 次运行中只有第 1 次出现此输出 (第 2/3 次因为缓存已存在所以跳过加载), 则正常.
若完全未出现, 说明加载逻辑有问题.

### 8.4 检查匈牙利算法边界

确认匈牙利算法中的边界检查是否存在 (07-20 修复):

```bash
grep -n 'j0 < 0\|j0 >= N\|m < 0\|m >= N' source/source_lcao/module_deltap/deltap_wannier.cpp
```

### 8.5 对比 deltap_branch.dat

3 次运行结束后, 比较生成的 `deltap_branch.dat`:

```bash
diff test_A/run1/deltap_branch.dat test_A/run2/deltap_branch.dat
diff test_A/run2/deltap_branch.dat test_A/run3/deltap_branch.dat
```

若不同, 说明分支状态写入的值在运行间不一致 → 分支选择本身非确定.

### 8.6 对比 deltap_match.dat

```bash
diff test_A/run1/deltap_match.dat test_A/run2/deltap_match.dat
diff test_A/run2/deltap_match.dat test_A/run3/deltap_match.dat
```

若不同, 说明匈牙利匹配结果在运行间不一致 → 匹配算法非确定.

---

## 9. 补充: 多 k 点验证 (可选)

如果 2x2x2 通过, 建议用更密 k 网格重复测试, 验证 k 点增多不会引入新非确定性:

### KPT_dense (4x4x4)

```
K_POINTS
0
Gamma
4 4 4 0 0 0
```

k 点增多 → Wilson loop 矩阵维度增大 → 本征值简并概率增加 → 分支选择压力更大.

---

## 10. 测试执行清单

- [ ] 编译 abacus_basic_para (无 ASAN)
- [ ] 创建目录结构 `tests/B16_test/test_A/` 和 `tests/B16_test/test_B/`
- [ ] 准备 INPUT/STRU/KPT/target.dat 文件
- [ ] 测试 A: 运行 3 次, 每次前删除全部缓存
- [ ] 测试 A: 提取数据, 填写结果表, 判定
- [ ] 测试 B: 运行 3 次, 每次前删除全部缓存
- [ ] 测试 B: 提取数据, 填写结果表, 判定
- [ ] 对比 deltap_branch.dat / deltap_match.dat
- [ ] 检查是否出现 `BRANCH INCONSISTENT`
- [ ] (可选) 4x4x4 k 点重复测试 A
- [ ] 汇总判定, 撰写结论
