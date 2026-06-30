# Phase 0 开发记录：CSZ 投影基确定

## 日期
2026-06-30

## 目标
实现 Complete Single-Zeta (CSZ) 投影基的自动确定，为 DeltaQS 电荷约束提供完备的投影空间。

## 实现方案

### 初始方案（已废弃）
尝试从赝势的 PP_PSWFC 段解析价电子构型：
- 解析 (n, l, f) 三元组
- 对每个 l 累加 occupation → n_e^l
- 计算 n_ζ^l = ⌈n_e^l / (2l+1)⌉

**问题发现**：ABACUS 在 `read_pp.cpp` 中过滤相对论 j=l+0.5 通道时，**不更新 occupations**，导致 pp.oc 数据错乱。

### 最终方案
1. 从赝势 header 读取 zv（总价电子数，可靠）
2. 从轨道文件读取每个 l 的 zeta 数（atoms[it].l_nchi[l]）
3. **使用轨道文件中所有可用 zeta** 作为 CSZ 基

**理由**：
- 轨道文件设计时已考虑覆盖价电子空间
- 使用所有 zeta 确保完备性
- 多余的极化函数增加灵活性，无害

## 文件变更

### 新增文件
| 文件 | 说明 |
|------|------|
| `source/source_lcao/module_deltaqs/upf_valence_parser.h` | CSZ 基确定接口 |
| `source/source_lcao/module_deltaqs/upf_valence_parser.cpp` | 实现 |
| `source/source_lcao/module_deltaqs/CMakeLists.txt` | 编译配置 |
| `source/source_lcao/module_deltaqs/test/CMakeLists.txt` | 测试编译配置 |
| `source/source_lcao/module_deltaqs/test/test_upf_valence_parser.cpp` | 单元测试 |
| `source/source_lcao/module_deltaqs/test/test_upf_parser_standalone.cpp` | 独立测试（未使用） |

### 修改文件
| 文件 | 变更 |
|------|------|
| `source/source_lcao/CMakeLists.txt` | 添加 module_deltaqs 子目录 |
| `CMakeLists.txt` | 链接 deltaqs 库 |
| `source/source_lcao/module_deltaspin/deltaqs.cpp` | 调用 CSZ 基确定 |

## 测试验证

### Fe 测试（4s2p2d1f 轨道文件）
```
Element: Fe (Z_val = 16)
l   orbital_zetas  csz_zetas  proj_orbitals
0        4            4            4
1        2            2            6
2        2            2           10
3        1            1            7
Total CSZ projection orbitals: 27
```

**验证**：
- ✅ 正确读取 zv=16
- ✅ 正确读取轨道文件 zeta 数
- ✅ CSZ 使用所有可用 zeta
- ✅ 总投影轨道数 = 4+6+10+7 = 27

### O 测试（2s2p1d 轨道文件）
```
Element: O (Z_val = 6)
l   orbital_zetas  csz_zetas  proj_orbitals
0        2            2            2
1        2            2            6
2        1            1            5
Total CSZ projection orbitals: 13
```

**验证**：
- ✅ 正确读取 zv=6
- ✅ 正确处理无 f 轨道的情况
- ✅ 总投影轨道数 = 2+6+5 = 13

## 关键发现

### ABACUS 相对论通道过滤
**位置**：`source/source_cell/read_pp.cpp:237-243`

```cpp
if(pp.lchi[nb] != 0 && std::abs(pp.jchi[nb] - pp.lchi[nb] - 0.5)<1e-6)
{
    new_nwfc--;  // 移除 j=l+0.5 通道
}
```

**影响**：
- 过滤后 nchi 减少（Fe: 6→4）
- 波函数被平均（正确）
- **occupations 未更新**（bug）

**结论**：不能依赖 pp.oc 数据，必须使用替代方案。

## 下一步
Phase 1：实现 CSZ 投影算符，使用确定的 zeta 数构建 pre_hr_csz。
