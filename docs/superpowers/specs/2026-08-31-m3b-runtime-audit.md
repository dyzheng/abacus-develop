# 2026-08-31: M3b 命运判决——升格为运行时审计（Tr[W^α·DM] vs ∫w_αρ）

> 上游：Task 2.3 评审待办② / Task 2.6 附加议程。M3b（W^α HContainer 路径，
> 生产 v_eff 网格注入）此前为"ctest 持续执行的审计仪器、生产零调用"的悬置
> 资产。判决：**升格为运行时审计**（评审建议两选项之一），不留悬置资产。

## 判决依据

- 框架核心卖点是"观测量==注入算符"：生产走 v_eff 网格注入（PW/LCAO 同构），
  M3b 的 W^α=∫φ_μ w φ_ν dr 矩阵路径与网格路径在数学上同值，但此前只在
  单测里对拍（W↔直积、ΣW≡S）。
- 升格成本可控：一次每几何 Gint vlocal 构建（与一次 H 构建同量级）+ 一次
  同布局 HContainer 点积；换来生产级矩阵×网格对拍（运行时审计线）。
- 删除选项会失去唯一能持续暴露"网格口径漂移"的矩阵级仪器；评审措辞
  "不留悬置资产"两选项等价，选价值更高的升格。

## 实现

- `ConstraintInjectLCAO::build` 增 `paraV` 可选参数：MPI 下目标 HContainer
  携带 Parallel_Orbitals 分布（`HContainer(paraV, nullptr, &ijr_info)`），
  满足 Gint 核 `transferSerials2Parallels` 的要求——否则 -np>1 段错误
  （串行走 `hR.add` 无此依赖，单测未暴露）。
- `ConstraintInjectLCAO::trace`：同布局（nnr + 完整 IJR 结构双重校验）
  HContainer 的平面点积（ABACUS 能量约定：逐 (iat1,iat2,R) 块配对）；
  布局不匹配返回 false，调用方跳过而非静默读垃圾。
- `ConstraintInjectLCAO::audit_weighted_trace`：建 W^α → 逐 α 求
  Tr[W^α·DM]（MPI 下 reduce_pool 归约 rank 局部和）→ 与网格观测
  q_grid=∫w_αρ 的最大绝对偏差；null DM / 数量不匹配 / 布局不匹配 → −1
  （调用方打印 SKIPPED）。
- 接线：`ESolver_KS_LCAO::iter_finish` 约束外环到达 DONE 时每几何跑一次
  （`constraint_audit_done_` 标志，before_scf 重置）；通道→DM 模式映射：
  charge → 总 DM（nspin=2 时 switch_dmr(1)），spin → 磁化 DM
  （switch_dmr(2)）；打印 `[constraint] M3b runtime audit: ...`。
- 单测：`TraceAndAuditMatchFlatProduct`（trace==手算平面和；审计喂入自身
  trace 偏差<1e-12；null/count 守卫返回 −1）。

## 测试设置与结果

- 单测：constraint_inject_lcao 5/5 PASS；constraint ctest 11/11。
- 集成：212_NAO_constraint_h2o（ecutwfc 20/ecutrho 80，charge，+0.1 e on O）。
  - 串行：`max |Tr[W.DM] − ∫wρ| = 1.51262655734e-08 e`
  - MPI -np 2：`1.51262806725e-08 e`（两值一致到 ~1e-12 相对，差异为归约顺序）
- 分析：偏差 1.5e-8 e ≈ SCF 收敛残差量级（观察在混合后密度、DM 为混合前，
  末次迭代 drho<scf_thr 时两者差 O(drho·∫w)）——矩阵路径与网格路径在
  生产运行中逐位吻合，审计判据建议 <1e-5 e（远小于约束容差 1e-4）。

## 下一步

- Task 2.6 三判决（PW≡LCAO / 力 FD / 力矩 FD）+ 反假收敛/MPI；本审计线
  在 LCAO 高网格 FD 腿上随跑（每几何一次），为"观测量==注入算符"提供
  运行时持续证据。
