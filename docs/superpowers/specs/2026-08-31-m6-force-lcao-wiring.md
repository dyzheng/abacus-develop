# 2026-08-31: Task 2.5.4——LCAO 力接线（FORCE_STRESS + 同一 constraint_force 核）

> 上游：`docs/superpowers/plans/2026-08-31-task25-m6-force-detail.md` 2.5.4。
> 架构声明验证点：LCAO 不写第二份力代码——调 **同一个** 2.5.2 核
> `constraint_force`，仅密度指针不同（LCAO 读 pw_rhod 网格上的 chr.rho）。
> 零新依赖/算法/积分框架（显式禁止清单不变）。

## 测试计划（失败测试先行）

1. 集成冒烟（LCAO 约束 H₂O，`tests/02_NAO_Gamma/212_NAO_constraint_h2o`，
   `relax_nmax 1` 触发 cal_force，`mpirun -np 2`）：
   - μ≠0 相：`test_force` 打印 `#CONSTRAINT  FORCE (Ry/Bohr)#` 且非零；
   - μ=0 参考相：同块**恒零**；
   - 外环 CONVERGED、μ* 与 Q_ref 与二期评审实测吻合（交叉印证接线在
     正确的密度/权重场上运行）。
2. 核级失败测试不重复：`constraint_force`（2.5.2 四测试）与
   `compute_force`（2.5.3 三测试）已锁定；LCAO 侧为薄钩子（编排 + 累加 +
   打印），与 PW 侧逐行同构，集成级验证即可。

## 测试设置

- 集成：`tests/02_NAO_Gamma/212_NAO_constraint_h2o/`（ecutwfc 20/ecutrho 80、
  15 Å 盒、gamma-only、target +0.1 e on O）；INPUT 追加 `test_force 1` +
  `calculation relax` + `relax_nmax 1`；`orbital_dir`/`pseudo_dir` 改绝对路径。
  二进制 `build/abacus_basic_para`（LCAO+MPI 配置）。
- 密度来源：`pelec->charge->rho`（LCAO 的 pw_rhod 网格密度，与约束环
  `ConstraintLoop::init(..., this->pw_rhod, ...)` 构建的共享权重场同网格；
  NCPP 下 pw_rhod ≡ pw_rho，`setup_pwrho.cpp:55`）。

## 结果

| 项 | 判据 | 实测 |
|---|---|---|
| 构建（FORCE_STRESS.cpp + abacus_basic_para 重链） | 无警告无错误 | 通过 |
| LCAO μ≠0 相力块 | 非零 | O z=+0.1090、H1/H2 x=∓0.1072、z=+0.0830（Ry/Bohr） |
| LCAO μ=0 参考相力块 | 恒零 | 全零（精确 0.0000000000） |
| 外环 | CONVERGED | μ*=−0.2193、Q_ref=6.407956559（二期评审实测 −0.2193 / 6.40796，吻合） |
| 驻点守卫 | 收敛态不触发 | 无"unconverged"记录 |
| 既有 LCAO 约束用例回归 | 不破坏 | 212_NAO 冒烟全程 EXIT=0 |

## 分析

- **同一核验证**：`FORCE_STRESS::getForceStress` 内新增
  `ConstraintLoop::instance().compute_force(pelec->charge->rho, nspin,
  forcecon)`——与 PW 的 `cal_force_constraint` 走完全相同的
  `compute_force → constraint_force` 代码路径；差异仅密度指针与网格来源
  （PW `chr->rho` on pw_rhod vs LCAO `pelec->charge->rho` on pw_rhod，
  物理上同网格同数组）。"双基组同一核"由构造成立，无第二份力代码。
- **接线位置**：仿 dspin/deltap 先例——力分量矩阵在"begin calculate and
  output force"前与其他额外力并列构造；总力累加 `if (PARAM.inp.constraint)
  fcs += forcecon`（在 H_HK 块之后）；`test_force` 下以 Ry/Bohr 独立打印块
  （与 PW 块同名同单位，便于 2.6 逐位对拍）。
- **量级自检**：O z 0.1090 Ry/Bohr ≈ 2.80 eV/Å，与 μ*=−0.2193 Ry 同量级；
  H 原子受力非零符合物理（Becke 权重 w_O 依赖全部原子位置）。H1/H2 镜像
  对称，池归约无重复累加（-np 2）。
- **μ=0 恒零**：kernel 零乘子短路在 LCAO 通道同样生效——参考相 mu=0 时
  力块精确全零（即使密度、权重场、导数网格全部已构建，短路发生在乘子层）。
- 单位与口径：力核输出 Ry/Bohr（μ in Ry，ρ in e/Bohr³）；打印块保持
  Ry/Bohr（与 PW 一致），不并入 eV/Å 各分量块。

## 下一步

- 2.5.5 出口判据汇总：全部单测 + ctest 全绿 + PW/LCAO 双冒烟力打印非零且
  μ=0 恒零 + 文档日志 + Commit。
- Task 2.6 三判决（FD 协议、PW≡LCAO 逐位、力矩 FD）按 phase2 计划执行，
  力 FD 前置网格参数写死（ecutwfc=100/ecutrho≥400/scf_thr=1e-8）。
