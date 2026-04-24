# DFT+U PW nspin=2 调试记录

## 问题描述

nspin=2 的 PW+DFT+U 测试 (222_PW_DFTU_S2_Z) 在 SCF 第3次迭代时发散，能量爆炸到 e+12 量级。

## 已修复的问题

### Bug 1: dftu_pw.cpp npol 硬编码
- **位置**: `source/source_lcao/module_dftu/dftu_pw.cpp:53,56`
- **问题**: `ib*2*nkb` 硬编码为 2，应使用 `ib*npol*nkb`
- **修复**: 改为 `ib*npol*nkb`

### Bug 2: op_pw_proj.cpp vu_begin 计算错误
- **位置**: `source/source_pw/module_pwdft/op_pw_proj.cpp:272`
- **问题**: `vu_begin += tlp1 * tlp1 * 4` 硬编码为 4
- **修复**: 改为 `vu_begin += tlp1 * tlp1 * npol * npol`

### Bug 3: nspin=2 自旋向下 vu_device 同步不完整
- **位置**: `source/source_pw/module_pwdft/op_pw_proj.cpp:297-328`
- **问题**: nspin=2 时，自旋向下通道只同步了前半部分 vu_device，但 kernel 访问了未初始化的后半部分
- **修复**: 为自旋向下通道单独同步 vu_device，仅同步有效数据大小

### Bug 4: nspin=2 locale 累加未区分自旋通道
- **位置**: `source/source_lcao/module_dftu/dftu_pw.cpp:27-31,60-76,121-137`
- **问题**: 所有 k 点的 locale 都累加到 `[0][0]`，自旋向下 k 点应累加到 `[0][1]`
- **修复**: 添加 `is` 变量，根据 `ik >= nk/2` 判断自旋索引

## 当前状态

### 测试结果
| 测试用例 | 状态 | 备注 |
|---------|------|------|
| 223_PW_DFTU_S4_XY (nspin=4 PW) | ✅ PASS | 需 mpirun -np 4 |
| 202_LCAO_DFTU_S2_Z (nspin=2 LCAO) | ✅ PASS | |
| 220_PW_SPIN_S2_Z (nspin=2 无 DFT+U) | ✅ PASS | |
| 222_PW_DFTU_S2_Z (nspin=2 PW+DFT+U) | ❌ FAIL | DS3 发散 |

### 调试数据
- Psi: nk=8, nbands=28, npol=1
- 自旋索引: ik=0,1,2,3 → is=0 (spin-up), ik=4,5,6,7 → is=1 (spin-down) ✓
- vu 值物理合理:
  - iter=2: vu_up[0]=-0.1331, vu_dn[0]=0.0886 (iat=0)
  - iter=3: vu_up[0]=-0.1597, vu_dn[0]=0.0979 (iat=0)

### 待排查问题
1. **nspin=2 PW+DFT+U SCF 发散** - vu 值正确但 SCF 在第3次迭代爆炸
   - 可能原因: DFT+U 势在 Hamiltonian 中的应用方式有误
   - 对比: zdy-tmp 收敛到 -6792.33 eV (45 iterations)
   - 对比: 纯 nspin=2 PW 收敛到 -6807.73 eV

## TODO

- [ ] **高优先级**: 排查 nspin=2 PW+DFT+U 发散根因
  - [ ] 对比 zdy-tmp 和当前代码的 vu_device 应用过程
  - [ ] 检查 nspin=2 时 k 点分布和 Psi 存储结构
  - [ ] 验证 cal_occ_pw 的 locale 并行归约是否正确
  - [ ] 检查 Hamiltonian 中 DFT+U 势的矩阵应用

- [ ] 运行完整集成测试套件确认修复无回归

- [ ] 清理调试代码和注释

## 参考
- zdy-tmp 参考代码: `/root/abacus-zdy-tmp`
- 收敛的 zdy-tmp 能量: -6792.33 eV (222_PW_DFTU_S2_Z)
