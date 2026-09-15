# 2026-09-15: 混合通道（charge+spin 同原子）力 FD —— LCAO 全轴（进行中，6/9 轴）

> 目标：力 FD 覆盖的**最后一个通道**——同一原子上 charge + spin 双约束（v2 列表）的
> 解析力是否等于能量导数。判据 0.0128555 eV/Å，不豁免。
> **状态：进行中（本地 12/18 腿后停止，交接另一台机器续跑）**，见 §5 交接清单。
> 前置：V1 自旋通道已于 2026-09-14 闭合（`2026-09-10-taskV1-spin-force-fd.md` §3.7）。

## 1. 测试计划（Test plan）

| # | 问题 | 判据 |
|---|---|---|
| Q1 | 混合通道（同一原子 charge+spin）解析力 = 能量导数？ | LCAO 全 9 轴 \|F_FD−F_ana\| < 0.0128555 eV/Å |
| Q2 | v2 多约束协议可冻结吗（多分量 t\*）？ | base 每条约束的 t\* 都被冻结；腿写回 v2 absolute 保留 kind/atoms |
| Q3 | 混合通道的净力指纹是否干净？ | Σ 补偿前力 ≈ 0（无 μw-Pulay 缺失特征） |

## 2. 测试设置（Test setup）

- **载体**：`tests/constraint_fd_force/cases/213_NAO_constraint_h2o_mixed`
  （213 几何、LCAO/gamma-only、nspin 2、O `mag 0.5`；v2 列表
  `{"constraints":[{"type":"charge","target":0.1,"atoms":[0]},{"type":"spin","target":0.1,"atoms":[0]}]}`）。
- **runner 扩展（入库 `1fd4fe35b`）**：
  1. **v2 多约束支持**：base 从首条 audit 逐分量取 `t*`；腿把 **每个分量** 的 target
     改写为冻结 `t*`（absolute 模式），保留 kind/atoms；`legs.tsv` 记 `mu_c/mu_s`；
  2. `SCF_NMAX`（默认 300 = 历史 INPUT 逐字节不变）；**混合通道必须放大**
     ——外环每步重跑一次 SCF，默认 300 会让 base 以
     `RUNNING (SCF ended before the outer loop converged)` 收尾（本轮实测）；
  3. **`ABA_CONSTRAINT_FIXED_MU` 是标量**（`std::fill(mu_.begin(), mu_.end(), v)`），
     无法逐分量冻结 μ ⇒ v2 场景**强制 `FIXED_MU=0`**（腿内重优化 μ，即电荷通道
     原本验证用的协议）并打印 notice；
  4. `std_checks` 块名自动发现（LCAO：OVERLAP/T_VNL/VL_dPHI/VL_dVL/EWALD/NLCC/SCC）。
- **命令（原样）**：
  ```sh
  TEST_FORCE=1 SCF_NMAX=1200 ABACUS=/root/abacus-develop/build_rel/abacus_basic_para \
    TAG=v1_lcao_mixed_fullaxis \
    CASE=$PWD/tests/constraint_fd_force/cases/213_NAO_constraint_h2o_mixed \
    bash tests/constraint_fd_force/tools/run_constraint_fd.sh lcao 0.005 4 2
  ```
- 环境：release 构建、np4、`MAXJOBS=2`（8 rank）、`OMP_NUM_THREADS=1`、R7 网格
  （ecutwfc 100 / ecutrho 400 / scf_thr 1e-8）、δ = 0.005 Bohr。

## 3. 结果（Results，**部分：12/18 腿 = 6/9 轴**）

- base：`E0 = −466.1435161607209920 eV`、`t* = 6.505555755 (charge) / 0.1 (spin)`、
  `μ* = −0.2256411084 / −0.0936213988 Ry`；base 墙钟 ~16 min。
- base 标准检查：Σ CONSTRAINT z = +7.237745 eV/Å（诊断量，非判据）；
  **Σ 补偿前 z = −0.001633 eV/Å ≈ 0**（无 μw-Pulay 缺失特征）；
  一致自证 `max|pre−compen−printed| = 2.154e-06 eV/Å`。
- 已归档腿：`0_0 / 0_1 / 0_2 / 1_0 / 1_1 / 1_2`（每腿 992–1621 s ≈ 16–27 min）。

| 原子轴 | F_FD (eV/Å) | F_ana (eV/Å) | \|d\| (eV/Å) | 富余 | 判定 |
|---|---|---|---|---|---|
| O-x | −0.0000141596 | −0.0000459404 | 3.178e-5 | 404× | PASS |
| O-y | −0.0000132746 | −0.0000452320 | 3.196e-5 | 402× | PASS |
| **O-z** | −2.8081114004 | −2.8065575071 | **1.554e-3** | **8.3×** | PASS |
| **H1-x** | −2.3221877709 | −2.3210530686 | **1.135e-3** | **11.3×** | PASS |
| H1-y | −0.0000001077 | +0.0000226161 | 2.272e-5 | 566× | PASS |
| H1-z | +1.4032772845 | +1.4032787493 | 1.465e-6 | 8776× | PASS |
| **H2-x / H2-y / H2-z** | —— | —— | —— | —— | **未跑（本地停止）** |

## 4. 分析（Analysis）

- **已测 6 轴全部 PASS**，最大残差 1.554e-3（O-z，8.3× 富余）、次大 1.135e-3（H1-x，11.3×）；
  小力轴（O-x/O-y/H1-y，F_ana ~5e-5）残差 ~3e-5，绝对量远低于判据。
- **误差结构**：混合通道残差（~1.5e-3）比自旋通道（1.8e-4）大约一个量级，但仍留 8–11× 富余。
  与"腿内重优化 μ + 腿收敛到 res≈1e-4"的能量噪声一致（双腿 E 各带 ~1e-4 eV 量级噪声 ⇒
  F 误差 ~0.02 eV/Å 上限），属协议成本而非力核缺陷；混合通道**无法**用 fixed-μ 压缩噪声，
  这是 `ABA_CONSTRAINT_FIXED_MU` 标量性的直接后果（登记为工具限制，M4 求解器不动）。
- **净力指纹**：base Σ 补偿前 = −0.0016 eV/Å ≈ 0，与电荷/自旋通道同量级 ⇒ 混合折叠
  （M6 per-α channel）**没有** μw-Pulay 型缺失。
- **成本**：重优化 μ ⇒ 每腿 16–27 min，全 18 腿 ≈ 3 h（重设计 §3 的 "1–1.5 h" 基于 fixed-μ，
  对混合通道不适用）。**真杠杆仍是载体/协议，不是求解器或构建类型。**

## 5. 下一步（Next steps，交接清单）

**A. 续跑本算例（另一台机器）**
```sh
# 方案 1（推荐，产出单一干净产物）：整轮重跑，TAG 复用会覆盖部分产物
TEST_FORCE=1 SCF_NMAX=1200 ABACUS=<repo>/build_rel/abacus_basic_para \
  TAG=v1_lcao_mixed_fullaxis CASE=$PWD/tests/constraint_fd_force/cases/213_NAO_constraint_h2o_mixed \
  bash tests/constraint_fd_force/tools/run_constraint_fd.sh lcao 0.005 4 2   # ~3 h
# 方案 2（省时，补齐 3 轴；每次会重算 base ~16 min）
for ax in 2_0 2_1 2_2; do
  ONLY=$ax TEST_FORCE=1 SCF_NMAX=1200 TAG=v1_lcao_mixed_rest ... run_constraint_fd.sh lcao 0.005 4 2
done
```
交回后：把 9/9 表写回本 spec §3、更新总览（§3.1/§3.11 同族）、dev-log 追加一轮，再 commit。

**B. PW 混合冒烟（2 轴，同族另一半）**
```sh
ONLY=0_2 KS_SOLVER=dav_subspace TEST_FORCE=1 SCF_NMAX=1200 TAG=v1_pw_mixed_oz \
  CASE=$PWD/tests/01_PW/213_PW_constraint_h2o_mixed ... run_constraint_fd.sh pw 0.005 4 2
# 再跑 ONLY=1_0（H1-x）
```

**C. 之后**：约束 relax 单步冒烟（几何驱动，验证跨离子步 λ/μ 生命周期）；
III-1 δ≈0 加密；4b + II-1 重锚定；阶段 B 立项评审。

---

## 本轮记录

- 本轮 = 混合通道力 FD 的 LCAO 全轴（进行中）+ runner v2 支持。本地在 12/18 腿后停止，
  产物已归档 `tests/constraint_fd_force/results/v1_lcao_mixed_fullaxis/`（含 `STATUS.md`）。
