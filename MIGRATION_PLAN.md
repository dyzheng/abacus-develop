# DFT+U PW Porting 分批迁移计划

## 参考代码
- zdy-tmp: `/root/abacus-zdy-tmp` (zdy/tmp 分支, HEAD=19ade1859)
- 目标: `/root/abacus-dftu-pw-port` (feat/dftu-pw-port 分支)
- 上游: `/root/abacus-develop`

## 目录映射
| zdy-tmp | develop |
|---|---|
| `source/module_hamilt_pw/hamilt_pwdft/` | `source/source_pw/module_pwdft/` |
| `source/module_hamilt_lcao/module_dftu/` | `source/source_lcao/module_dftu/` |
| `source/module_hamilt_lcao/module_deltaspin/` | `source/source_lcao/module_deltaspin/` |
| `source/module_hamilt_lcao/hamilt_lcaodft/operator_lcao/` | `source/source_lcao/module_operator_lcao/` |
| `source/module_esolver/` | `source/source_esolver/` |
| `source/module_elecstate/` | `source/source_estate/` |
| `source/module_hamilt_lcao/hamilt_lcaodft/` | `source/source_lcao/` |

## API 映射
| zdy-tmp | develop |
|---|---|
| `ModuleDFTU::DFTU` | `Plus_U` |
| `GlobalV::NSPIN` | `PARAM.inp.nspin` |
| `GlobalV::KPAR` | `PARAM.inp.kpar` |
| `GlobalC::ucell` | 参数传递 `const UnitCell& cell` |
| `psi_p->npol` | `psi_p->get_npol()` |
| `FS_Nonlocal_tools` | `Onsite_Proj_tools` |
| `GlobalV::NPROC_IN_POOL` | `PARAM.globalv.nproc_in_pool` |

## 已完成 (5 commits, ~1000 行 diff)
- `onsite_projector.h/cpp` nspin=1/2 基础适配
- `op_pw_proj.cpp` nspin=1/2 VU 修复
- `kernels/onsite_op.cpp/cu/hip.cu` npol=1 GPU/DCU 支持

## 迁移分批

### Batch 1: PW OnsiteProjector 核心 (4 files, ~500 行 diff)
- `onsite_projector.h` (diff=20/159)
- `onsite_projector.cpp` (diff=152/672)
- `op_pw_proj.h` (diff=27/106)
- `op_pw_proj.cpp` (diff=278/460)

### Batch 2: PW Force & Stress (10 files, ~2500 行 diff)
- `forces.h` (85/157), `forces.cpp` (758/719)
- `forces_onsite.cpp` (36/81)
- `kernels/force_op.h` (104/339), `kernels/force_op.cpp` (156/438)
- `stress_func.h` (89/283), `stress_onsite.cpp` (57/124)
- `kernels/stress_op.h` (117/508), `kernels/stress_op.cpp` (255/741)
- `stress_pw.h` (39/52), `stress_pw.cpp` (91/196)
- `kernels/onsite_op.cpp` (34/117)

### Batch 3: DFTU LCAO 核心 (8 files, ~1900 行 diff)
- `module_dftu/dftu.h` (335/358), `dftu.cpp` (206/485)
- `dftu_pw.cpp` (174/359), `dftu_force.cpp` (411/579)
- `dftu_hamilt.cpp` (139/172), `dftu_occup.cpp` (447/572)
- `module_operator_lcao/dftu_lcao.h` (27/142), `dftu_lcao.cpp` (120/556)
- `module_operator_lcao/op_dftu_lcao.h` (14/49), `op_dftu_lcao.cpp` (21/82)

### Batch 4: DeltaSpin 模块 (8+ files, ~800 行 diff)
- `module_deltaspin/spin_constrain.h` (110/291)
- `cal_mw.cpp` (185/172), `cal_mw_from_lambda.cpp` (282/543)
- `module_operator_lcao/dspin_lcao.h` (27/160), `dspin_lcao.cpp` (37/531)
- **新文件**: `cal_h_lambda.cpp`, `cal_mw_helper.cpp`, `sc_parse_json.cpp`
- **新测试**: `test/cal_h_lambda_test.cpp`, `test/cal_mw_helper_test.cpp`, `test/init_sc_test.cpp`

### Batch 5: ESolver + ElecState (10 files, ~4200 行 diff)
- `esolver_ks_pw.cpp` (1553/447)
- `esolver_ks_lcao.cpp` (1612/572)
- `esolver_ks.cpp` (1017/349)
- `elecstate_pw.cpp` (276/563), `elecstate_lcao.cpp` (193/90)
- `fp_energy.h` (23/75), `fp_energy.cpp` (72/127)
- `elecstate_energy.cpp` (257/360), `elecstate_energy_terms.cpp` (41/58)
- `elecstate.h` (125/166)
- **新文件**: `module_charge/density_matrix.h`, `density_matrix.cpp`

### Batch 6: LCAO 基础 (6 files, ~2200 行 diff)
- `hamilt_lcao.cpp` (434/587), `hamilt_lcao.h` (151/179)
- `FORCE_STRESS.cpp` (921/979)
- `spar_u.h` (11/32), `spar_u.cpp` (29/247)
- `spar_hsr.cpp` (526/426)
- `module_operator_lcao/operator_lcao.cpp` (77/303)

## 依赖关系
```
Phase 1: Batch 1 + Batch 3 (并行, 无依赖)
    ↓           ↓
Phase 2: Batch 2 + Batch 4 (并行, 各依赖上一阶段)
    ↓           ↓
Phase 3: Batch 5 + Batch 6 (并行, 依赖 Phase 2)
```

## 验收标准
1. 编译通过 (cmake --build . -j$(nproc))
2. 相关单元测试 PASS
3. 相关集成测试 PASS
4. 代码审查通过
