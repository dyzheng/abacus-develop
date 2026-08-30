# 2026-08-31 M2：约束读数 Q_α = ∫ w_α·ρ dr（constraint_observe）

## 1. Test plan
- `AtomicSuperposition`：ρ = Σ_I N_I·gauss_I（O:8e、H:1e、H:1e，σ=1.0 Bohr），
  20 Bohr 盒、h=0.5（40³）。断言：Σ_α Q_α == N_el（1e-8，FP 累积地板
  ~1e-10）；逐原子 Q_α 与高阶独立求积参考差 < 0.05 e（分账网格求积地板
  实测 ~0.035 e）；h=1.0 网格误差 > h=0.5 网格误差（向参考单调收敛）。
- `PointwiseDeltaReading`：单点网格 δ 密度 ρ(g*)=1/dV → Q_α ≡ w_α(g*)，
  20 个探针点，判据 1e-12（索引映射/ΔV/逐点权重的精确钉）。
- `ConstantDensitySum`：ρ≡1 → Σ_α Q_α == 盒体积（1e-8）。
- MPI `ObserveReduceConsistent`（test_mpi，4 rank）：ρ≡1 读数 reduce_pool
  后各 rank 位一致（allgather 比对），ΣQ == ω。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest；`MODULE_ESTATE_constraint_observe` +
  `MODULE_ESTATE_constraint_weight_grid_mpi`（mpirun -np 4）。
- 输入：H2O（O 盒心 (10,10,10)，H ±1.2 Bohr 镜像）20 Bohr 盒；
  PW_Basis 40³（h=0.5）/20³（h=1.0）；radii={1.5,0.5,0.5} Bohr。
- 参考求积：每原子一中心球面网格（baker 120×Rcut 8 + delley lmax=35），
  同一 w_becke_adjusted 配方，独立于 PW 网格。

## 3. Results
- 串行 3/3 PASS；MPI 2/2 PASS（1/2/4 rank 全过）。
- 分账灵敏度：h=0.5、σ=1.0 时 O 分账差 0.035 e（参考 7.0239 vs 读数
  6.9888），Σ 恒等于 N_el（两侧均精确到 1e-10）；h=1.0 误差更大（单调
  收敛成立）。
- 首次跑 10 Bohr 盒 + σ=1.0 时 sum rule 差 2.1e-4：根因=盒边界 ρ 尾
  ~3.7e-6 非周期，中点求积边界项泄漏；改 20 Bohr 盒后边界尾 ~2e-22，
  sum rule 回到 FP 地板。

## 4. Analysis
- 计划中"Q_I == N_I 精确（1e-8）"对 Becke 权重不成立：分账 = 网格求积
  量，误差 ~(h/σ)² 量级；精确钉是 sum rule（单位分解）+ 逐点 δ 读数。
  逐原子对拍降级为有界灵敏度 + 收敛性检验，口径基准留给 V2（Multiwfn
  细网格，判据 1e-4 e）。
- 自旋通道 m=ρ↑−ρ↓ 只预留接口（nspin==2 时读数取 ρ↑+ρ↓ 电荷通道），
  一期不接。

## 5. Next steps
- Task 4 (M4)：mu_solver 逐分量 secant + κ clamp + 翻号检测 + 熔断，
  mock Q(μ) 映射先行。
