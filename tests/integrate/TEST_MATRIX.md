# Integration Test Matrix — DFT+U / DeltaSpin / PW

> 仓库: `/root/abacus-dftu-pw-port` (branch: `feat/dftu-pw-port`)
> 最后更新: 2026-04-18

## 测试覆盖矩阵

| ID | 功能 | nspin | 磁矩方向 | 基组 | SOC | 状态 | 备注 |
|---|---|---|---|---|---|---|---|
| 815 | DFT+U | 2 | AFM(z) | PW | ✗ | ✅ PASS | 815_PW_DFTU_S2 |
| 816 | DFT+U | 1 | FM | PW | ✗ | ❌ BUG | nspin=1 崩溃（zdy-tmp 同样有问题） |
| 099 | DFT+U | 4 | xyz | PW | ✓ | ✅ PASS | 099_PW_DJ_SO |
| 160 | DFT+U | 4 | xyz | PW | ✓ | ✅ PASS | 160_PW_DJ_PK_PU_SO |
| 54 | DFT+U(LCAO) | 2 | FM(z) | LCAO | ✗ | ✅ PASS | 54_NO_PK_PU |
| 55 | DFT+U(LCAO) | 1 | 无 | LCAO | ✗ | ✅ PASS | 55_NO_PK_PU_S1 |
| 56 | DFT+U(LCAO) | 4 | xyz | LCAO | ✓ | ✅ PASS | 56_NO_PK_PU_SO |
| 53 | DFT+U(LCAO) | 2 | z+URamp | LCAO | ✗ | ✅ PASS | 53_NO_PK_URAMP |
| **250** | **DeltaSpin** | **2** | **AFM(z)** | **PW** | **✗** | **待测** | **新** |
| **251** | **DeltaSpin** | **4** | **xyz** | **PW** | **✗** | **待测** | **新** |
| **252** | **DFT+U+DS** | **2** | **AFM(z)** | **PW** | **✗** | **待测** | **新** |
| **253** | **DFT+U+DS** | **4** | **xyz** | **PW** | **✗** | **待测** | **新** |

## 新增测试说明

### 250: PW DeltaSpin nspin=2
- 体系: bcc Fe, 2 atoms, AFM (z方向)
- 约束: Fe1=+2μB, Fe2=-2μB (z方向)
- 验证: DeltaSpin 独立工作（无DFT+U），nspin=2 路径

### 251: PW DeltaSpin nspin=4
- 体系: bcc Fe, 2 atoms, FM (xyz方向)
- 约束: Fe1=(1,1,1)μB, Fe2=(1,1,1)μB
- 验证: DeltaSpin 独立工作，nspin=4 non-collinear 路径

### 252: PW DFT+U + DeltaSpin nspin=2
- 体系: bcc Fe, 2 atoms, AFM (z方向)
- U=5eV, d-orbital, AFM 约束
- 验证: DFT+U 和 DeltaSpin 共存，occupation mixing + spin constraint

### 253: PW DFT+U + DeltaSpin nspin=4
- 体系: bcc Fe, 2 atoms, FM (xyz方向)
- U=5eV, d-orbital, FM 三维约束
- 验证: 最复杂场景 — DFT+U + DeltaSpin + non-collinear

## 数值比对标准

| 物理量 | 阈值 | 说明 |
|---|---|---|
| 总能量 | 1e-6 eV | dftu-pw-port vs zdy-tmp |
| 力 | 1e-4 eV/Å | 逐分量比较 |
| 应力 | 1e-4 kbar | 逐分量比较 |
| 原子磁矩 | 1e-3 μB | 逐分量比较（nspin=4 时 xyz 三个分量） |
| DFT+U 能量 | 1e-6 eV | 如果启用 DFT+U |

## 测试运行方法

```bash
cd /root/abacus-dftu-pw-port/tests/integrate
# 运行单个测试
OMP_NUM_THREADS=2 /root/abacus-dftu-pw-port/build/abacus_2p > out.log 2>&1
# 或者用 Autotest.sh
bash Autotest.sh -a ../../build/abacus_2p -n 2 -r "250|251|252|253"
```

## 参考基准

- zdy-tmp 二进制: `/root/abacus-zdy-tmp/build/abacus`
- 相同 test case 在 zdy-tmp 运行后生成 result.ref
- dftu-pw-port 运行结果与 result.ref 比对
