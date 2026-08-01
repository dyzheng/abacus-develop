# DeltaP relax（多离子步）实验算例

## 状态：已知问题复现器（2026-08-01 R5）
本算例用于覆盖 `reset_ionic_step()` 的 relax 多离子步路径。**当前在力计算阶段崩溃**，
该崩溃为**重构（R1–R4）前既有的 bug**（`hamilt::DeltaPOperator` 自 2026-07-09 提交
`85b2af322` 引入，relax/force 路径从未跑通过），非本轮重构回归。保留本算例作为后续
修复的回归复现器。

## 复现
```bash
cd /root/abacus-develop/tests/deltap_relax
rm -rf OUT.*
OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para > relax.log 2>&1
```

## 现象（gdb 背靠栈）
1. 修复前：`DeltaPOperator` 构造函数空指针段错误
   （`FORCE_STRESS.cpp` 以 `hR=nullptr` 构造，构造函数无条件 `hR->get_paraV()`）。
   R5 已加空指针守卫（`deltap_lcao.cpp`），该段错误已消除。
2. 修复后：SCF 43 步收敛（DRHO≈1e-6）后进入 `cal_force`，在
   `DeltaPOperator::cal_force_stress` 的 OMP 区内 `double free or corruption` abort。

```
#0  __pthread_kill ... (SIGABRT)
#6  malloc_printerr "double free or corruption (!prev)"
#10 hamilt::DeltaPOperator<complex,double>::cal_force_stress(...) [._omp_fn.0]
#12 hamilt::DeltaPOperator<complex,double>::cal_force_stress(...)
#13 Force_Stress_LCAO<complex>::getForceStress(...)
#14 ESolver_KS_LCAO<complex,double>::cal_force(...)
#15 Relax_Driver::relax_driver(...)
```

## 根因分析（初步）
- `cal_force_stress` 的 nlm 填充存在混合基组越界读：
  `nlm_target[...] = nlm[channel][iw + m]`（`iw+m` 可超过 ket 原子轨道数）与
  `cal_force_IJR` 用 nlm1 的 `lmax` 索引 `nlm2[index]`（O–H 对越界）。越界读产生
  垃圾力值，且堆损坏在 SCF 阶段（43 步 deltap 算子）累积，最终在 force 区释放时
  触发 double-free。
- force 数学本身即标注不完整（`deltap_force_stress.hpp`：缺 H_HK 力项与 ∂τ/∂R 项），
  relax/MD + deltap_corr 属实验功能。

## 后续修复建议（不在 R5 范围）
1. 统一 nlm 布局：按 `(l,m)` 显式索引，越界即跳过（`if (iw+m >= num_ket) continue;`）。
2. `cal_force_IJR`/`cal_stress_IJR` 对 nlm2 加长度守卫（`index < nlm2.size()/4`）。
3. 修复后按 `tests/deltap_fd_force/run_fd.sh` 的 FD 判据（<5%）验收 C-02。
