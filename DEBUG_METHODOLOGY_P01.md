# P0-1 调试经验总结 — 分离验证法

> 日期: 2026-04-29
> 问题: nspin=2 PW+DFTU SCF 在迭代 4 发散（能量爆炸到 10^33 eV）
> 状态: 未解决，已定位到 vu 应用阶段

---

## 一、错误的工作方式（本次教训）

### ❌ 问题 1: 混入大量 debug print 但无法定位根因
```cpp
// 错误做法：到处加 std::cout
std::cout << "[DFTU-DEBUG] ik=" << ik << " vu=" << vu << std::endl;
std::cout << "[DFTU-KERNEL] ps=" << ps << std::endl;
// ... 20+ 处 debug print
```
**后果**: 输出混乱，无法区分"值正确"和"逻辑正确"，代码难以维护。

### ❌ 问题 2: 同时修改多处，无法确认哪个修复有效
- 修改了 mix_uom 支持 nspin=2
- 修改了 allocate_mixing_uom 的 uom_fold
- 修改了 cal_occ_pw 的结构
- 修改了 iter_init_dftu_pw 的 iter==1 skip
**后果**: 无法确认哪个改动是必要的，哪个引入了新问题。

### ❌ 问题 3: 只验证了"vu 计算正确"，没验证"vu 被正确使用"
- 验证了 cal_occ_pw 计算的 vu 值正确 ✅
- 验证了 eff_pot_pw 指针地址一致 ✅
- **但没有验证**: Hamiltonian 中 vu 是否真的被应用到 hpsi ❌

---

## 二、正确的工作方式 — 分离验证法

### 原则：每次只验证一个假设，确认后再提交

```
假设 → 最小修改 → 独立测试 → 确认/否定 → 提交 → 下一个假设
```

### 步骤 1: 建立基线
```bash
# 1. 确认 zdy-tmp 收敛
cd /root/abacus-zdy-tmp/tests/integrate/222_PW_DFTU_S2_Z
mpirun -np 1 /root/abacus-zdy-tmp/build/abacus
# 结果: -6792.33 eV, 45 次迭代 ✅

# 2. 确认我们的代码发散
cd /root/abacus-dftu-pw-port/tests/integrate/222_PW_DFTU_S2_Z
mpirun -np 1 /root/abacus-dftu-pw-port/build/abacus_basic_para
# 结果: iter=4 能量爆炸到 10^33 eV ❌
```

### 步骤 2: 分离验证 vu 计算
```cpp
// 在 cal_occ_pw 末尾添加最小验证
// dftu_pw_test.cpp 中添加单元测试
TEST(DFTU_PW, Nspin2_VU_Calculation) {
    // 输入: 已知波函数和权重
    // 输出: 验证 locale 和 vu 值与预期一致
    // 预期: spin-up vu ≈ -0.133 Ry, spin-down vu ≈ 0.088 Ry
}
```
**关键**: 单元测试不依赖完整 SCF 循环，只验证 vu 计算函数本身。

### 步骤 3: 分离验证 vu 传递
```cpp
// 在 cal_ps_dftu 入口验证
// 不使用 full SCF，只调用 OnsiteProj 算子
TEST(DFTU_PW, Nspin2_VU_Transfer) {
    // 1. 设置已知 eff_pot_pw 值
    // 2. 调用 cal_ps_dftu
    // 3. 验证 vu_device 与 eff_pot_pw 一致
    // 4. 验证 ps 输出与预期一致
}
```

### 步骤 4: 分离验证 Hamiltonian 作用
```cpp
// 在 hpsi 计算后验证
// 比较应用 DFT+U 前后的 hpsi 差异
TEST(DFTU_PW, Nspin2_Hpsi_Correction) {
    // 1. 计算无 DFT+U 的 hpsi_0
    // 2. 计算有 DFT+U 的 hpsi
    // 3. 验证 delta = hpsi - hpsi_0 与 vu * becp 一致
}
```

### 步骤 5: 二分法定位发散点
```
迭代 1: 能量 -6795 eV ✅
迭代 2: 能量 -6780 eV ✅（微小偏差可接受）
迭代 3: 能量 -6780 eV ✅
迭代 4: 能量 10^33 eV ❌

→ 问题在迭代 3→4 的某个环节
→ 检查迭代 3 结束时的 locale/vu 值
→ 检查迭代 4 开始时的波函数
→ 检查迭代 4 的 Hamiltonian 构建
```

---

## 三、本次调试已确认的事实

| 验证项 | 结果 | 方法 |
|--------|------|------|
| vu 矩阵计算 | ✅ 正确 | `[DFTU-END]` 输出 vs 预期值 |
| eff_pot_pw 指针 | ✅ 一致 | 对比 cal_occ_pw 和 cal_ps_dftu 中的地址 |
| vu 同步到 vu_device | ✅ 正确 | `[DFTU-VUDEV]` 显示 sync 后值正确 |
| locale 矩阵稳定性 | ✅ 稳定 | `[DFTU-SUB]` 3 次迭代值不变 |
| nspin=4 测试 | ✅ 通过 | 全部 23 个测试 PASS |
| nspin=2 无 DFT+U | ✅ 通过 | -6797 eV, 10 次迭代 |
| **nspin=2 有 DFT+U** | **❌ 发散** | iter=4 能量爆炸 |

---

## 四、待验证的假设（下一步）

### 假设 A: onsite_ps_op kernel 有 bug
**验证方法**: 用 zdy-tmp 的 onsite_op.cpp 替换我们的版本
```bash
# 1. 备份当前文件
cp source/source_pw/module_pwdft/kernels/onsite_op.cpp /tmp/

# 2. 从 zdy-tmp 复制（调整 include 路径）
# ... 修改 include 路径 ...

# 3. 编译并运行测试 222
# 如果收敛 → kernel 有 bug
# 如果仍发散 → 问题在别处
```

### 假设 B: 波函数在迭代 3→4 之间被破坏
**验证方法**: 在 iter_finish 时输出波函数范数
```cpp
// iter_finish 中添加
psi_norm = psi.norm();
std::cout << "[ITER-FINISH] iter=" << iter << " psi_norm=" << psi_norm << std::endl;
```

### 假设 C: 电荷混合导致发散
**验证方法**: 关闭 mixing_dftu
```
INPUT: mixing_dftu 0
# 如果仍发散 → mixing 不是原因
```

---

## 五、关键教训

1. **先写测试，再改代码**: 每次修改前，先写一个能捕获该问题的测试
2. **最小修改原则**: 每次只改一处，确认有效后再改下一处
3. **分离关注点**: vu 计算 ≠ vu 传递 ≠ Hamiltonian 应用 ≠ SCF 收敛
4. **提交要小**: 每个 commit 只包含一个验证通过的修复
5. **保留现场**: 调试代码用 `#ifdef DEBUG_DFTU` 保护，不要直接提交

---

## 六、推荐调试流程模板

```bash
# 1. 复现问题
cd test_case && mpirun -np 1 abacus > run.log 2>&1

# 2. 提取关键数据
grep "E_KohnSham\|FINAL" run.log

# 3. 对比参考实现
cd zdy_tmp/test_case && mpirun -np 1 abacus > ref.log 2>&1

# 4. 差异分析
diff <(grep "E_KohnSham" run.log) <(grep "E_KohnSham" ref.log)

# 5. 定位发散迭代
# 找到第一个能量偏差 > 1e6 的迭代

# 6. 在该迭代前后添加最小验证
# 只输出关键变量的值，不要全量 dump

# 7. 修复 → 测试 → 提交
```
