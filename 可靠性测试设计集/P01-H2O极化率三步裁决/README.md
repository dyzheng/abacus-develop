# P01　H₂O 极化率三步裁决（旗舰测试）

> 系列：DeltaP ｜ 优先级：P0 ｜ **阻塞状态：被 F1/F2 换算链阻塞（判定结果为备忘录定稿前工作值）**

## 1. 目的

裁决"DeltaP 能否正确计算极化率"。同盒、同基组、同泛函下：能量有限差分（efield，无 γ 换算，定义级裁判）给出 α_ref；DeltaP total 约束 ±λ 扫描（raw γ）经 F1/F2 换算链给出 α_DeltaP；两者一致即裁决。

## 2. 目录结构

```
P01-H2O极化率三步裁决/
  README.md
  run.sh                 # 主工作流：阶段1裁判 + 阶段2被裁 + 阶段3裁决
  cases/h2o/{STRU,KPT}   # H2O 实验几何，30 Bohr 立方盒，Γ 点
  runs/                  # 运行产物（run.sh 自动生成）
    efield/ef_*/         # 阶段1: efield_amp = ±0.001/±0.0005/0
    deltap/lam_*/        # 阶段2: lambda = ±0.08/±0.02/0
    results.txt          # 汇总与判定表
```

## 3. 用法

```bash
./run.sh
# 可覆盖环境变量：
ABACUS=/path/to/abacus NPROC=4 PSEUDO_DIR=... ORBITAL_DIR=... ./run.sh
```

- `ABACUS`（默认 `/root/abacus-develop/build/abacus_basic_para`）
- `PSEUDO_DIR`（默认 `/root/pporb/apns-pseudopotentials-v1`）
- `ORBITAL_DIR`（默认 `/root/pporb/apns-orbitals-efficiency-v1`）
- `NPROC`（默认 1；>1 时用 `mpirun -np`）

## 4. 判据

| 项 | 判据 | 说明 |
|---|---|---|
| 裁决 | \|α_DeltaP − α_ref\| / α_ref ≤ 10% | 核心裁决项 |
| 反对称 | ±λ 反对称偏差 < 5% | λ=±0.02 与 ±0.08 两对 |
| efield 线性 | 四点线性 R² ≥ 0.99 | E 对 δ² 最小二乘 |
| 充分性 | α_ref ∈ [7,10] a.u. | 越界仅 WARNING 不 FAIL |

末尾打印 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

换算链（run.sh 头部常量区，备忘录定稿前工作值）：

- F1：`E_field(Ha/Bohr) = -π·λ/(2·A_BOHR)`，`A_BOHR=30.0`
- F2：`α_DeltaP = -(A_BOHR/π)·dγ/dE_field`
- 能量 FD：`α_ref = 2·[E(+δ)+E(−δ)−2E(0)]/δ²`（E 单位 Ry、δ 单位 Ha，系数 2 为 Ry→Ha）

## 5. 已知风险（设计文档 §7）

- 负 λ SCF 失败是既有失败模式（R1）：本脚本负 λ 用 mixing_beta=0.3 缓解；若确认负 λ 存在物理性失稳，需改用正 λ 单侧 + 能量 FD（F3 记账）路径。
- 若裁决失败，排查序：DFPT 第三裁判 → SMO 投影完备性实测 → HK 项符号/权重 → 基组敏感性（QZDP）。
- **F1/F2 换算系数未最终核实**：α_ref 公式的符号依赖 dip_cor 能量记账约定（脚本取绝对值进入判据）；α_DeltaP 亦取绝对值对比。判定前打印 `WARNING: F1/F2 换算链待备忘录定稿（阻塞项）`。本测试不过，一切 DeltaP 响应类结论"待裁决"。
