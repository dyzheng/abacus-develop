# DeltaSpin LCAO Lambda 优化：下一步计划

## 当前状态总结

纯子空间对角化方案（`run_lambda_loop_lcao`）在 Fe 强关联体系上失败，根因是冻结基近似忽略了 SCF 自洽场响应（Root Cause B），导致子空间预测的 Mi 与全量对角化偏差 0.6-4.4 µB。

已修复的 4 个代码 bug（V 矩阵列主序、target_mag 读取、Newton 步长限制、chi k 权重）均已验证。

结论：需要转向保留全量对角化的优化策略。

---

## 方案选择：方案 A — 解析 chi 引导的 BFGS 加速

保留 `run_lambda_loop()` 的全量 `cal_mw_from_lambda()` 调用，但用解析 Jacobian chi 改进搜索方向，减少迭代步数。

核心思路：原始 BFGS 的第一步搜索方向是 `search = delta_spin`（梯度方向），`alpha_trial` 是固定初始步长。用 chi 可以直接给出更好的初始 `alpha_trial` 和搜索方向缩放，使第一步就接近最优，从而减少后续迭代。

---

## 实施步骤

### Step 1: 清理 debug 代码

移除 `run_lambda_loop_lcao()` 中的所有 debug 打印和全量对角化验证代码。保留核心的 P_I_sub 计算和 chi 计算逻辑，因为 Step 2 需要复用。

涉及文件：
- `lambda_loop.cpp`: 移除 debug std::cout 和 inner==0 的 fulldiag 验证块

### Step 2: 提取 `cal_chi_analytical()` 为独立方法

从 `run_lambda_loop_lcao()` 的 Phase 2-3 提取为独立方法：

```cpp
/// @brief 计算解析 Jacobian chi_I = dM_I^z / dlambda_I
/// @return chi[nat] 向量，单位 Ry/uB
std::vector<double> cal_chi_analytical();
```

内部流程：
1. 对每个 k 点调用 `dspin_op->cal_PI_sub()` 计算投影矩阵
2. 用微扰论公式计算 chi（已验证正确的版本，含 k 权重修正）
3. 返回 chi 向量

涉及文件：
- `spin_constrain.h`: 声明 `cal_chi_analytical()`
- `lambda_loop.cpp` 或新文件 `cal_chi.cpp`: 实现
- `template_helpers.cpp`: 添加 double 模板空实现
- `CMakeLists.txt`: 如果新建文件则添加

### Step 3: 在 `run_lambda_loop()` 中用 chi 优化 alpha_trial

原始代码在 i_step==0 时使用固定的 `alpha_trial_`（用户输入参数）。改为：

```cpp
if (i_step == 0 && PARAM.inp.basis_type == "lcao" && PARAM.inp.nspin == 2)
{
    // 用解析 chi 计算最优初始步长
    auto chi = this->cal_chi_analytical();
    // alpha_trial_opt = 1/|chi_avg| 使得第一步 Newton 步长合理
    double chi_avg = 0.0;
    int n_constrained = 0;
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_[iat].z != 0)
        {
            chi_avg += std::abs(chi[iat]);
            n_constrained++;
        }
    }
    if (n_constrained > 0) chi_avg /= n_constrained;
    if (chi_avg > 1e-10)
    {
        alpha_trial = 1.0 / chi_avg;  // 最优步长估计
    }
}
```

这样第一步的 `dnu = search * alpha_trial = delta_spin / chi_avg` 就近似于 Newton 步 `delta_lambda = delta_M / chi`，大幅减少后续迭代。

涉及文件：
- `lambda_loop.cpp`: 在 BFGS 搜索方向计算前插入 chi 引导逻辑

### Step 4: 恢复 esolver 路由到 `run_lambda_loop()`

当前 `esolver_ks_lcao.cpp` 对 nspin==2 路由到 `run_lambda_loop_lcao()`。改回统一调用 `run_lambda_loop()`（内部已包含 chi 优化）。

涉及文件：
- `esolver_ks_lcao.cpp`: 移除 nspin==2 的特殊分支，统一调用 `run_lambda_loop()`

### Step 5: 编译验证

```bash
cd /root/abacus-ds-lcao-optimize/abacus-develop/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON -DBUILD_TESTING=OFF ..
cmake --build . -j$(nproc)
```

验收标准：0 error，生成 abacus 二进制。

### Step 6: 集成测试

运行测试算例 `tests/17_DS_DFTU/24_LCAO_DS_S2_Z`：

```bash
cd tests/17_DS_DFTU/24_LCAO_DS_S2_Z
mpirun -np 4 /path/to/abacus
```

验收标准：
- SCF 收敛（drho < scf_thr）
- 磁矩收敛到目标值 ±2.0 µB（RMS < sc_thr）
- 对比原始 `run_lambda_loop()` 的迭代步数，确认 chi 引导后步数减少

### Step 7: 可选 — 逐原子 chi 缩放搜索方向

如果 Step 3 的全局 `chi_avg` 效果不够好（不同原子的 chi 差异大），可以进一步做逐原子缩放：

```cpp
// 用 chi 做对角预条件
for (int iat = 0; iat < nat; iat++)
{
    if (this->constrain_[iat].z != 0 && std::abs(chi[iat]) > 1e-10)
    {
        search[iat].z /= std::abs(chi[iat]);  // 预条件化搜索方向
    }
}
alpha_trial = 1.0;  // 预条件化后步长为 1
```

这等价于用对角 Hessian 逆做预条件的 BFGS，收敛速度应接近 Newton 法。

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| chi 计算开销抵消收敛加速 | 低 | 低 | chi 计算只需 P_I_sub（O(N^2)），远小于全量对角化 O(N^3) |
| chi 在某些体系上不准确 | 中 | 低 | chi 仅用于初始步长估计，后续 BFGS 自适应修正 |
| cal_PI_sub 在大体系上内存不足 | 中 | 中 | P_I_sub 是 nbands×nbands 矩阵，对大体系可能需要流式计算 |

---

## 时间估计

| 步骤 | 预计耗时 |
|------|----------|
| Step 1: 清理 debug | 15 min |
| Step 2: 提取 cal_chi | 30 min |
| Step 3: 集成到 BFGS | 20 min |
| Step 4: 恢复 esolver 路由 | 5 min |
| Step 5: 编译验证 | 10 min |
| Step 6: 集成测试 | 30 min |
| Step 7: 逐原子缩放（可选） | 20 min |
| 总计 | ~2 小时 |
