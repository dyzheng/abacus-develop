# P 系列测试算例与工作流构建设计

> 日期：2026-07-30 ｜ 分支：feat/deltap ｜ 依据：`可靠性测试设计集/P01–P18` 设计文档 + `DeltaP测试说明.md`
> 状态：设计已获用户批准（2026-07-30），进入实现

## 1. 目标

为 `可靠性测试设计集/` 中 P01–P18 共 18 个 DeltaP 测试构建**可执行的算例与工作流脚本**，每个测试一个子目录，自包含。

## 2. 目录结构

```
可靠性测试设计集/
  PXX-<名称>/
    README.md        # 目的、算例清单、判据、阻塞状态、用法
    run.sh           # 主工作流：生成算例目录→批量运行→提取→判据→PASS/FAIL 表
    cases/           # 检入输入文件（INPUT 模板、STRU、KPT、target.dat 等）
    runs/            # 运行产物（gitignore）
```

## 3. 公共约定（18 个 run.sh 统一）

### 3.1 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ABACUS` | `/root/abacus-develop/build/abacus_basic_para` | 可执行文件 |
| `PSEUDO_DIR` | `/root/pporb/apns-pseudopotentials-v1` | 赝势目录 |
| `ORBITAL_DIR` | `/root/pporb/apns-orbitals-efficiency-v1` | 轨道目录 |
| `NPROC` | `1` | MPI 进程数（>1 时用 mpirun） |

### 3.2 提取与判定

- 每次运行前 `rm -rf OUT.*`（无缓存原则）
- 提取键：`[rawG] Σγ_raw`、`E_KohnSham`（running_scf.log）、`TOTAL-FORCE`、`[DeltaP-PW] γ_total`
- 收敛巡检：`grep -l "SCF IS NOT CONVERGED"`，未收敛点子目录在判定表中标记 FAIL/INVALID
- 判据用 awk 数值比较；末尾打印逐项 PASS/FAIL 表，最后 `SUMMARY: n/n PASS`，退出码 0/1
- 被阻塞测试（依赖 F1/F2 换算链或 P01 结论）：run.sh 完整可跑，判定段打印
  `WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）`；换算系数集中在脚本头部常量区
  （`F1_PI_OVER_A`、`F2_SPIN_FACTOR` 等），便于备忘录定稿后单点修改

### 3.3 F1/F2 换算常量（备忘录定稿前的工作值）

- `E = -(π/a)·λ`：λ 单位 Ry → E(Ha) = π·λ/(2a)，a 为 gdir 方向盒边长（Bohr）
- 偶极：`μ(e·Bohr) = (a/π)·Σγ_raw`（含离子项时由代码输出总极化），`1 D = 0.393430 e·Bohr`
- 极化率 FD：`α = [E(+δ)+E(−δ)−2E(0)]/δ²`（E_KohnSham 单位 Ry，δ 单位 Ha，结果乘 2 得 Ha 制 a.u.）

> **⚠️ LCAO 极化率基组重定标（2026-08-13 D2 实测补注，评审要求）**：
> h2o1（30 Bohr 盒，O 2s2p1d/H 2s1p 紧缩基）锯齿场实测 **α_LCAO = 3.02 Bohr³**，
> 而实验 α(H₂O)=9.8 Bohr³——**紧缩 LCAO 基低估极化率 ~3.3×**（无弥散函数的
> 已知行为；κ_phys=(eL/π)²/α 相应为 60.4 Ry/rad² 而非按实验值推的 18.6）。
> **P01/P05 等一切"α 判据"必须用同基组参考值（或能量 FD 自洽 α），不得直接
> 拿实验 α 判 LCAO 的数**；大窗外推需先做基组收敛（加弥散/换 TZDP+）标定。

### 3.4 分子标准设置

- H₂O 实验几何：r_OH=0.9572 Å、∠HOH=104.52°、C2 轴沿 z
  （O 在盒中心，H 在 z=+0.5858 Å、x=±0.7571 Å）
- 盒 30 Bohr（15.873 Å），PBE，Γ 点，scf_thr 1e-8（能量 FD 类），LCAO/TZDP 级轨道
- 其余分子几何：CH₄ r=1.087 Å 正四面体；CO r=1.128 Å；NH₃ r=1.012 Å、∠HNH=106.7°；
  HF r=0.917 Å；H₂S r=1.336 Å、∠=92.1°

### 3.5 可用赝势/轨道（已盘点）

- 赝势 `/root/pporb/apns-pseudopotentials-v1`：O.upf H.upf C.upf N_ONCV_PBE-1.0.upf F.upf
  S.upf Na.upf Cl.upf Si.upf B.PD04.PBE.UPF Ba_ONCV_PBE-1.0.upf Ti_ONCV_PBE-1.2.upf
- 轨道 `/root/pporb/apns-orbitals-efficiency-v1`：全部所需元素 2s2p1d 级（H 2s1p）
- 基组层级（P05/P07）：可用层级不足时按层级目录占位，缺文件则 SKIP 并打印说明

## 4. 逐测试内容矩阵

| 测试 | 算例 | 工作流 | 判据 | 阻塞 |
|---|---|---|---|---|
| P01 | H₂O 30 Bohr：efield ±0.0005/±0.001（dip_cor=1）+ DeltaP ±0.02/±0.08 Ry + λ=0 | 能量 FD→α_ref；raw γ→dγ/dλ→α_DeltaP；反对称性与 R² | \|Δα\|/α_ref≤10%，R²≥0.95 | F1/F2 标注 |
| P02 | H₂O：LCAO λ=0 三方向（如需）+ PW berry_phase gdir=1/2/3 | raw γ→μ，γ_total→μ，对比实验 1.855 D | 通道互差≤0.02 D，与实验≤0.05 D | 无 |
| P03 | H₂O：λ=±0.005/±0.01 扫力；O 位移 ±0.005 Å berry_phase FD | dF/dλ→Z*，位移 FD→Z*，中和检查 | 互差≤0.05，\|ΣZ*\|≤0.05 | P01 |
| P04 | CH₄/CO/NH₃/HF/H₂S 五分子，LCAO λ=0 + PW berry_phase | 逐分子 μ，对 CCSD(T) 表 | MAE≤0.03 D，CH₄≤0.01 D，CO 符号 | 无 |
| P05 | 五分子：efield FD + DeltaP ±0.02/±0.08（DZP+TZDP 层级） | 逐分子 α 双通道 | \|Δα\|≤10%（5/5） | P01 |
| P06 | H₂O L=12/15/18/21/24 Å：λ=0 μ + α 双通道 | 收敛曲线、平台值 | 15 Å：\|Δμ\|≤3%、\|Δα\|≤5% | α 部分 P01 |
| P07 | H₂O：LCAO 层级 × PW ecut 40/60/80/100 | μ+α 双收敛曲线 | TZDP→QZDP ≤2%/3%；ecut80→100 ≤1%/2% | 无 |
| P08 | H₂O：λ=±0.01/±0.02/±0.04/±0.08 + 0 九点 | raw γ 线性/反对称/SCF 迭代数巡检，λ_crit 定位 | 反对称<5%，R²≥0.98 | 无 |
| P09 | H₂O + h-BN：三次全新运行 × OMP 1/4/8 | raw γ、E 逐位对比 | Δγ≤1e-6 rad，ΔE≤1e-8 Ha | 无 |
| P10 | H₂O：λ 七点 vs efield 七点 | E_phys=E_KS−λγ 换算，双曲线斜率/截距/κ 对比 | 斜率差≤10% | P01 |
| P11 | h-BN：DeltaP ±λ gdir=3，k 6×6×4 与 9×9×6 | dγ/dλ→α→ε=1+4πα/V | 与 berry_phase FD ≤5%，k 收敛≤2% | P01 |
| P12 | NaCl/Si：DeltaP ±λ 力 + 位移 ±0.005 Å berry_phase FD | dF/dλ→Z* 两通道 | 互差≤0.05，NaCl≈±1，Si≈0 | P01 |
| P13 | BaTiO₃ 四方相：Ti z 位移 0→1 路径 ~9 点 | 逐点 raw γ 分支连续性 + Ps 三通道对比 | 三通道 Ps 互差≤10% | P09 |
| P14 | H₂O/NH₃/CH₄/Si：ΔP per-atom γ vs w90 WC | 总偶极对比；w90 未安装则该通道 SKIP | 总偶极≤0.02 D | 无 |
| P15 | H₂O/NH₃/CO：gdir=1/2/3 各做 ±λ + efield FD | 三分量 α，各向异性比 | 逐分量≤10%，比值差≤5% | P01 |
| P16 | H₂O PW：onsite_radius=6/10/14/20 Bohr × ±λ | 覆盖率(r)、dγ/dλ(r) 曲线，对比 LCAO | 收敛响应与 LCAO ≤10% | 无 |
| P17 | H₂O：固定 λ=±0.02/±0.04 relax vs 固定 E relax | 对比 Δr_OH、Δ∠ | ≤0.005 Å / ≤0.5°，符号一致 | P01 |
| P18 | H₂O：λ=0/0.02 与 E=0/E(λ) 各 out_chg | Δρ 逐点 Pearson 相关 | r≥0.99，RMS≤5% | P01 |

## 5. 实现分工

四个并行构建组：
- A 组（P0 旗舰）：P01 P02 P08 P09
- B 组（分子响应）：P03 P04 P05 P15
- C 组（收敛/等效）：P06 P07 P10 P16 P17 P18
- D 组（固体）：P11 P12 P13 P14

## 6. 验证计划

1. `bash -n` 全部 18 个 run.sh
2. 抽查 INPUT 关键字与当前分支参数表一致（deltap_* 已在源码确认）
3. 冒烟运行 P02（LCAO 通道）与 P09（H₂O 单次），确认工作流端到端可跑
4. 结果记录到本文档"实现结果"节 + deltap-development-log.md
