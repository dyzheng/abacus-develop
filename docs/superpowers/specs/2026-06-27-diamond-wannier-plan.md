# Diamond Wannier Center 对比：计划文档

> **日期**: 2026-06-27
> **目标**: 直接对比 DeltaP 的逐能带 Berry phase γ_n 与 Wannier90 的 Wannier center ⟨r_n⟩

---

## 1. 体系参数

| 参数 | 值 |
|------|-----|
| 结构 | Diamond (C), FCC, 2 atoms |
| 晶格常数 | lat0 = 6.1 Bohr, latvec = [[-0.5,0,0.5],[0,0.5,0.5],[-0.5,0.5,0]] |
| 体积 | Ω = 56.745 Bohr³ |
| K-mesh | 4×4×4 = 64 k-points, symmetry=-1 |
| nocc | 4 (C: 4 valence e × 2 atoms / 2 = 4) |
| LCAO | C_lda_8.0au_100Ry_2s2p1d.orb (nwl=2, nproj_SMO=9, nproj_NAO=14 per atom) |
| gdir | 3 (a3 = [-0.5, 0.5, 0] × 6.1 = [-3.05, 3.05, 0] Bohr) |

**注意**: a3 = (-3.05, 3.05, 0) 无 z 分量。gdir=3 的极化沿 a3 方向（xy 平面内），不是沿 z。

## 2. 对比方案

### 2.1 量与单位

| DeltaP 量 | Wannier90 量 | 关系 | 单位 |
|-----------|-------------|------|------|
| γ_n = arg(λ_n) | γ_n^W = Im ln ∏_k [U†MU]_nn | γ_n ≈ γ_n^W (模 2π) | 无量纲 |
| (a3/2π) × γ_n | ⟨r_n⟩ (from .wout) | 应相等 (模 a3) | Bohr |
| P_elec = prefactor × Σ γ_n | P = -e/Ω × Σ ⟨r_n⟩ | 应相等 | e/bohr² |

其中:
- a3 = |a3| = √(3.05² + 3.05²) = 4.3134 Bohr
- prefactor = -0.5 × a3 / (2π × Ω) = -0.5 × 4.3134 / (2π × 56.745) = -0.006046

### 2.2 需要运行的测试

| 测试 | 内容 | 文件 |
|------|------|------|
| diamond_ref | 参考结构 (C1=0.875,0.875,0.875; C2=0.125,0.125,0.125) | 已有 (01_lcao) |
| diamond_ref_deltap | NSCF with berry_phase + deltap(wannier) + towannier90 | 需运行 |
| diamond_disp | 位移结构 (C1 z: 0.875→0.885) | 已有 (01_lcao_disp) |
| diamond_disp_deltap | NSCF with berry_phase + deltap + towannier90 | 需运行 |

### 2.3 预期结果

**参考结构** (中心对称):
- P = 0 (berry_phase 和 DeltaP 都应为 0)
- γ_n 应成对出现 ±γ (反演对称)
- Wannier center 应关于原胞中心对称

**位移结构**:
- P ≠ 0 (打破反演对称)
- berry_phase 给出 P_total
- DeltaP 给出 P_elec (ratio 应 ≈ 0.97)
- Wannier90 给出 8 个 Wannier center 位移
- 总 Wannier center 位移 = -Ω × P_elec / e

### 2.4 关键验证点

1. **P_elec ratio** (DeltaP/berry): 应 ≈ 0.97 (已知 BaTiO3 的 ratio)
2. **γ_n vs ⟨r_n⟩**: 逐能带对比 (模 a3)
3. **Σ γ_n vs arg(det(W))**: sum rule (精确)
4. **逐原子 P^I**: BaTiO3 无法对比，diamond 可以对比 (2 个 C 原子)

### 2.5 Wannier90 .wout 解析

Wannier90 输出 8 个 Wannier center (num_wann=8, 含 disentanglement)。
其中 4 个对应占据态 (valence), 4 个对应非占据态 (conduction)。

区分方法: 通过 Wannier center 位置与原子位置的接近程度判断。
- 接近 C1 (0.875,0.875,0.875) 的 4 个 = valence
- 接近 C2 (0.125,0.125,0.125) 的 4 个 = valence
- 或者: 从 .eig 文件看哪些 Wannier 函数能量 < Fermi

**注意**: DeltaP 只用 4 个占据能带 (nocc=4)。Wannier90 用 8 个 Wannier 函数。
需要取 Wannier90 的 4 个占据 Wannier center 来对比。

## 3. .mmn 对比 (已有数据)

之前已从 .mmn 文件计算了 Berry phase:
- .mmn det (15 bands) = 1.200 (BaTiO3)
- 对 diamond: .mmn det (4 bands) ≈ 0 (中心对称)

可以直接从 .mmn 计算 Wilson loop 特征值，与 DeltaP 对比。
两者用**相同的重叠矩阵** (如果 DeltaP 的 S_dk 与 .mmn 一致)。

**但**: DeltaP 用 berry_phase 相位+位置修正, .mmn 用 Wannier90 的 unkdotkb。
两者相位约定不同，不能直接对比 γ_n。只能对比 det(W) (= 总 Berry phase)。

