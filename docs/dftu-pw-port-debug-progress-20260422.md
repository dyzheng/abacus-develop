# DFTU PW Port Debug Progress

> Updated: 2026-04-22 20:00
> Branch: `feat/dftu-pw-port`
> Test target: `tests/17_DS_DFTU/08_PW_DFTU_S2_Z`
> Reference: ETOT = -6792.33 eV (zdy-tmp binary)

---

## 已修复问题 (4 bugs)

### 1. iter=1 istep=0 缺少跳过逻辑

**Bug**: `iter_init_dftu_pw` 在第一次 SCF 迭代时调用 `cal_occ_pw`，此时 psi 是随机初始化的，计算出的 VU 是垃圾值，污染 Hamiltonian 导致对角化崩溃。

**修复**: 在 `dftu_pw.cpp` 的 `iter_init_dftu_pw` 中添加 guard：
```cpp
if (iter == 1 && istep == 0) return;
```

**文件**: `source/source_pw/module_pwdft/dftu_pw.cpp`

### 2. `ps` 缓冲区未在每次调用时清零

**Bug**: `cal_ps_dftu` 中 `setmem_complex_op()(this->ps, 0, ...)` 只在 `!init_dftu` 块内执行。on site kernel 使用 `+=` 累加，后续 Davidson step 调用时 ps 累积了上一次的值，导致数值暴增。

**修复**: 将清零逻辑移到 `init_dftu` 块外部，确保每次调用都清零：
```cpp
// Always zero ps before each call (kernel uses += accumulation)
if (this->nkb_m < m * tnp) { ... }
setmem_complex_op()(this->ps, 0, tnp * m);
```

**文件**: `source/source_pw/module_pwdft/op_pw_proj.cpp:cal_ps_dftu`

### 3. nspin=2 spin-down VU 指针偏移缺失

**Bug**: `eff_pot_pw` 在 nspin=2 时布局为 `[spin0_iat0 | spin0_iat1 | spin1_iat0 | spin1_iat1]`，但 spin-down k-point 仍然从头同步整个数组到设备。

**修复**: 根据 `isk[ik]` 判断自旋通道：
```cpp
if(PARAM.inp.nspin == 2 && this->isk[this->ik] == 1) {
    syncmem_complex_h2d_op()(vu_device, dftu->get_eff_pot_pw(0) + half_size, half_size);
}
```

**文件**: `source/source_pw/module_pwdft/op_pw_proj.cpp:cal_ps_dftu`

### 4. nspin=1/2 kernel 中 m2 负索引越界

**Bug**: nspin=1/2 kernel 中 `ip_m[ip2]` 可能返回 -1（非关联轨道），导致 `index_mm = m1 * tlp1 + m2` 为负数，访问 `vu_iat` 越界。

**修复**: 添加 `if(m2 < 0) continue;` 检查。

**文件**: `source/source_pw/module_pwdft/kernels/onsite_op.cpp`

### 5. `dftu_pw.cpp` 完全重写 — nspin 分支缺失

**Bug**: port 的 `dftu_pw.cpp` 被损坏为只使用 nspin=4 路径，缺少：
- `is` (spin) 跟踪
- nspin=1/2 vs nspin=4 分支
- `initialed_locale` 守卫
- `uom_array` 管理
- nspin=2 的 VU 和 energy 计算
- `set_locale` 调用

**修复**: 完全重写 `dftu_pw.cpp` 以匹配 zdy-tmp 参考实现。

**文件**: `source/source_lcao/module_dftu/dftu_pw.cpp`

---

## 测试状态

| 测试 | 描述 | 状态 | ETOT |
|------|------|------|------|
| 06_PW_SPIN_S2_Z | PW S2 无 DFTU | ✅ PASS | -6807.73 eV |
| 07_PW_SPIN_S4_XYZ | PW S4 无 DFTU | — | — |
| 08_PW_DFTU_S2_Z | PW DFTU S2 | ⚠️ 部分通过 | DS1=-6795.74 eV ✅, DS2 crash ❌ |

### Test 08 进展

| 阶段 | 崩溃点 | 说明 |
|------|--------|------|
| 修复前 | DS1 (对角化第1步) | TMAG=10^28，VU 从随机 psi 计算 |
| 修复1后 | DS2 (对角化第2步) | TMAG≈0 正常，但 DS2 仍崩溃 |
| 修复2后 | iter 2 初始化 | ps 清零修复消除了对角化中的数值暴增 |
| 修复3+4+5后 | DS2 (对角化第2步) | TMAG=10^19，OnsiteProj 所有调用通过 |

**DS1 验证**:
- TMAG = -4.78e-06 ✅ (接近 0)
- ETOT = -6795.74 eV (vs reference -6792.33 eV, 差异 < 4 eV)
- VU = 0 ✅ (iter 1 时正确)
- ps_norm = 0 ✅ (每次调用正确清零)
- OnsiteProj act() 调用全部完成 ✅ (act=50-55 通过)

---

## 当前问题: iter 2 SCF 崩溃

### 症状

```
iter=1: 正常完成 (DS1-DSn 全部通过)
iter=2: 崩溃 (signal 6, Aborted)
```

### 已确认排除的因素

| 因素 | 验证方式 | 结论 |
|------|----------|------|
| OnsiteProj 所有调用 | 详细 debug 打印显示所有 act() 完成 | 正确 ✅ |
| cal_occ_pw 调用 | debug 显示 iter=2 时正确调用 | 正确 ✅ |
| VU 计算 | eff_pot_pw 被正确更新 | 正确 ✅ |
| OnsiteProj 复制构造 | 修复后正确复制 dftu 指针 | 正确 ✅ |

### 关键发现

通过详细的文件级 debug 打印（300+ 行输出）确认:
1. **OnsiteProj 的 ~200 次调用全部完成** - 包括 init、vu_device sync、kernel 调用
2. **cal_occ_pw 在 iter=2 被正确调用** - eff_pot_pw 被正确更新
3. **崩溃发生在 OnsiteProj 之外** - 所有 CAL_PS_DFTU "kernel complete" 打印都出现了

### 疑似根因

**崩溃发生在 charge density 更新或 mixing 阶段**，而不是 DFTU/OnsiteProj 部分。

可能的原因:
1. `psiToRho` 电荷密度计算问题
2. Broyden mixing 的内存访问问题
3. MPI 同步问题

### 调试建议

1. 在 `iter_finish` 或 `hamilt2rho_single` 中添加 debug 打印
2. 检查 `pelec->psiToRho()` 的返回值
3. 验证 MPI reduce 操作的同步性

---

## 修改文件清单

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `source_pw/module_pwdft/dftu_pw.cpp` | 添加 iter=1 istep=0 guard + DFTU 检查 | ✅ 已编译 |
| `source_pw/module_pwdft/op_pw_proj.cpp` | ps 清零 + spin-down VU 偏移 + debug | ✅ 已编译 |
| `source_pw/module_pwdft/kernels/onsite_op.cpp` | m2 负索引检查 | ✅ 已编译 |
| `source_lcao/module_dftu/dftu_pw.cpp` | 完全重写，恢复 nspin 分支 | ✅ 已编译 |
| `source_lcao/module_dftu/dftu.h` | cal_occ_pw 声明添加 istep 参数 | ✅ 已编译 |
| `source_esolver/esolver_ks_pw.cpp` | 调用 iter_init_dftu_pw 包装器 | ✅ 已编译 |

### 测试 24 结果

**测试**: `tests/17_DS_DFTU/24_LCAO_DS_S2_Z`
**状态**: ❌ 崩溃
**崩溃点**: MPI_Allreduce 期间 (heap corruption)
**说明**: 与 test 08 崩溃模式相似，表明问题可能是系统性的

---

## 下一步 TODO

### P0 - 修复 iter 2 SCF 崩溃

- [ ] 在 `iter_finish` 或 `hamilt2rho_single` 中添加 debug 打印
- [ ] 检查 `pelec->psiToRho()` 的返回值
- [ ] 验证 MPI reduce 操作的同步性
- [ ] 对比 zdy-tmp 在 iter 2 的完整数据流

### P1 - 验证修复

- [ ] Gate 1: 编译通过（0 error）
- [ ] Gate 3: test 08 完整通过
- [ ] Gate 3: 全量 17_DS_DFTU 测试

### P2 - 代码审查

- [ ] 清理所有调试打印
- [ ] Gate 4: 代码风格检查
- [ ] Gate 4: 无未使用的 include

---

*文档位置: /root/abacus-dftu-pw-port/docs/dftu-pw-port-debug-progress-20260422.md*
