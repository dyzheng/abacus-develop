# 2026-10-30 MBIS+ALM 接入 ABACUS 的真实集成方案

> 现状：独立模块 `module_mbis_alm`（MBIS 权重 + ALM + 力 + adapter + 标定）
> 已在真实 cube 上验证（LiCl/Fe2），10/10 独立测试通过。
> 本文给出把它接入 ABACUS `module_constraint` / ESolver 的可执行方案。

## 1. 架构决定

**复用现有 `module_constraint` 的注入/观测/ALM/外层循环**，只把"空间权重后端"
换成 MBIS：
- `WeightGrid` 增 `WeightType::MBIS`（**密度依赖**，需外层更新）；
- `ConstraintInject{PW,LCAO}`、`constraint_observe`、`constraint_alm` **不改**，用新权重；
- `mbis_alm` 核心保持独立；真实 adapter 是唯一耦合点。

## 2. 数据转换（ABACUS → `mbis_alm`）

| 需要 | 来源 | 转换 |
|---|---|---|
| `Grid.n` | `rho_basis->nx/ny/nz`（或 `nrxx` 分解） | 直接 |
| `Grid.dv[i]` | 晶格矢量 `a_i`（`latvec*lat0`） | `dv[i] = a_i / n_i`，`origin=0` |
| `Grid.origin` | — | 0 |
| `Atom.R` | `ucell.atoms[it].tau[ia] * lat0` | Bohr |
| `Atom.Zval` | `ucell.atoms[it].ncpp.zv` | 价电子数 |
| `rho_total` | `chr.rho[is][ir]` | nspin=1: `rho[0]`；nspin=2: `rho[0]+rho[1]` |
| `spin` | 同上 | `rho[0]-rho[1]` |

**网格约定核对**：先写一个 serial 单测，用 `PW_Basis` 生成网格并与
`tools/mbis_cube_io.h` 的 cube 坐标对照，确认 `dv_i=a_i/n_i` 的符号/顺序。

**MPI**：`chr.rho` 是 rank-local（`nrxx`）；`compute_mbis` 目前是 serial。
两条路线：
1. **gather→rank0→scatter**（实现简单，通信 O(Ng)，适合先跑通）；
2. 把 `compute_mbis` 改成 MPI-aware（pro-density 与求和 all-reduce，权重复用）。
建议先走 (1)，性能问题后置。

## 3. 接入点（`constraint_loop`）

1. 输入：`constraint_weight_type mbis`（新增）。
2. 每个 **外层 μ 步**开始时，用当前 `chr` 重建 MBIS 权重（密度依赖）：
   `wg.build_mbis(rho_total)` → 写入 `cw_/w_`（与 Becke/Hirshfeld 同布局）。
3. 之后沿用现有 `ConstraintInject`（`v_c = μ_eff w_A`）、`constraint_observe`
   （`N_A=∫w_Aρ`、`M_A=∫w_A m`）、`constraint_alm`（`mu_eff=mu+k(Q-t)`、外层更新）。
4. **k 标定**：在进入外层前用 `calibrate_alm` 的思路测 `χ=dQ/dmu`（固定乘子 solve），
   取 `k=κ/|χ|`；`kmin` 必须 < `1/|χ|`（上一轮教训）。

## 4. 力/应力

- **力**：`constraint_force`（模块，冻结参数、解析 `∂w/∂R`）给出约束力；
  或复用 `constraint_deriv` 的权重导数核（需把 MBIS 的 `∂w/∂R` 灌进去）。
  与常规 DFT 力相加。
- **应力**：同理，模块的应变导数核（待补，见 §6）；
  冻结 `ζ,N` 近似 + 可选 KL 参数 Pulay 项。
- **注意**：MBIS 的 `N,ζ` 随几何/密度变；严格力/应力含参数响应，近似可先省略。

## 5. 周期/对称

- 小/常规胞（bcc）用 `images`+`symmetrize`（模块已支持），否则会出现之前
  Fe2 的 17.8/14.2 破缺；**只对同种/等价原子**对称化。
- 大盒分子（LiCl）用 `images=0` 即可。

## 6. 待补模块件

- `constraint_stress`（MBIS 应变导数核）+ FD 测试；
- MPI-aware `compute_mbis`（或 gather 包装）；
- `PW_Basis/Charge` → `Grid/rho` 的 bridge（含网格约定单测）。

## 7. 测试计划

1. **网格转换单测**：serial `PW_Basis` → `Grid`，与 cube 路径对照。
2. **自由密度一致性**：Fe2/LiCl 在 `constraint_weight_type mbis` 下的 `N/M`
   应与 cube 结果一致（LiCl Li 2.0447；Fe2 16/16、±2.65）。
3. **约束运行**：Fe2/LiCl 电荷+自旋目标，ALM 收敛；对比 Lagrange/penalty。
4. **回归**：现有 23 个 constraint 测试 + 新增 10 个 module 测试全绿。
5. **k 标定**：验证 `χ` 谱与 k、`kmin` 的关系；异质体系（LiCl）ALM 应显著更快。

## 8. 风险

- **密度依赖权重** → 外层迭代收敛（frozen 近似 + 阻尼）；与近零模交互。
- **MPI 一致性** → gather/scatter 或 all-reduce 的求和审计（`Σ_A w_A=1`）。
- **kmin 安全**（`k|χ|∈(0,2)`）；**网格约定**（`dv_i=a_i/n_i`）易错。
- **性能**：MBIS KL 每次外层重建（O(N_at·N_g·iter)）；先跑通再优化。
- 与现有 Becke/Hirshfeld/DeltaSpin 的**数值口径不同**，不可混用。

## 9. 分阶段任务

1. `PW_Basis/Charge → Grid/rho` bridge + 网格约定单测（本机可做）。
2. `WeightGrid::build_mbis` + 自由密度一致性测试。
3. `constraint_loop` 的外层重建 + `constraint_weight_type mbis`。
4. Fe2/LiCl 约束 + ALM/k 标定验证。
5. 力/应力核 + FD 测试；MPI/性能。
``` 
