# P12　NaCl / Si Born 有效电荷（可执行算例）

> 系列：DeltaP ｜ 前置：P01（**阻塞中**）、P11 ｜ 设计文档：`../P12-NaCl-Si-Born有效电荷.md`

## 1. 目的

固体 Born 有效电荷 Z\* 是 Berry 相位方法的经典试金石：NaCl 近刚性离子（Z\*≈±1.0）、
Si 近零反常小值，覆盖强弱两极限。验证 DeltaP 的 dF/dλ 路径在体材料 k 点网格下
与传统位移 FD（berry_phase）一致。

## 2. 目录结构

```
P12-NaCl-Si-Born有效电荷/
  README.md
  run.sh              # 主工作流
  cases/
    STRU_NaCl         # 岩盐 fcc 初基胞（a=5.64 Å），Na(0,0,0) Cl(1/2,1/2,1/2)
    STRU_Si           # 金刚石 fcc 初基胞（a=5.431 Å），Si(0,0,0) (1/4,1/4,1/4)
    KPT               # 8×8×8
    INPUT_lcao.tmpl   # DeltaP LCAO（cal_force 1，genelpa）
    INPUT_pw.tmpl     # PW berry_phase + deltap_switch true（测 γ_total）
  runs/               # 运行产物（含 results.txt）
```

## 3. 用法

```bash
bash run.sh                  # 或 NPROC=4 bash run.sh
```

环境变量：`ABACUS` / `PSEUDO_DIR` / `ORBITAL_DIR` / `NPROC`。

## 4. 工作流

**通道①（DeltaP dF/dλ，LCAO/genelpa）**：每体系 λ=±0.005、±0.01 Ry 四点 SCF
（`cal_force 1`，负 λ 用 `mixing_beta 0.3`），从 `OUT.*/running_scf.log` 的
`TOTAL-FORCE` 块取两原子 F_z，中心差分 dF/dλ →（F1 换算）→ Z\*（双 λ 尺度，兼作线性巡检）。

**通道②（berry_phase 位移 FD，PW）**：阳离子（Na / 第一个 Si）沿笛卡尔 z 位移
±0.005 Å（fcc 初基矢下对应分数位移 (δ/a, δ/a, −δ/a)，保持 Δx=Δy=0），
PW `berry_phase 1, gdir 3` 读 `[DeltaP-PW] γ_total`，两点 FD → Z\*_zz
（电子项 + 离子项 Z_ION，常量区）。

## 5. 判据表

| 编号 | 判据 | 状态 |
|---|---|---|
| J1 | 全部算例 SCF 收敛且有力/γ 输出 | 有效（硬判） |
| J2 | Si \|Z\*_zz\| < 0.1（通道②，不依赖 F1） | 有效（硬判） |
| — | NaCl \|Z\*_Na\| ∈ [0.9,1.2]（通道②） | 软判，越界打印 WARNING |
| J3 | 两通道 Z\* 互差 ≤0.05 | **阻塞**（通道①需 F1） |
| J4 | 中和检查 \|ΣZ\*\| ≤ 0.05（通道① 双原子） | **阻塞**（同上） |

末尾 `SUMMARY: n/n PASS`，退出码 0/1；结果存 `runs/results.txt`。

## 6. 阻塞状态与换算链

本测试**阻塞于 P01**：通道① 的 Z\* = (dF/dλ)/(dE/dλ) 需要 F1 换算
（`E=−π·λ/(2a)`，fcc 初基胞 gdir=3 方向有效盒长取 a/√3，待备忘录固体专项核实）。
J3/J4 打印 `WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）`，不计入 SUMMARY。
通道② 不经过 λ，J2 与 NaCl 区间为有效判据。

换算常量集中在 run.sh 头部（`FORCE_CONV`、`Z_ION_NA/SI`、每体系 `L3_BOHR`/`F1_LAM2E`），
备忘录定稿后单点修改。

## 7. 风险与注意（抄自设计文档 §7）

- Si 小 Z\* 是符号/数值噪声敏感区，双侧差分 + 高位 SCF 阈值（已设 scf_thr 1e-8）；
- NaCl 离子性强，检查分支选择稳定性（越界 WARNING 即为此设）。

补充（工程层面）：

- `TOTAL-FORCE` 提取按 "标签+三列数值" 启发式解析 running_scf.log，
  首次实跑后若格式不符需微调 `get_fz()` 的正则；
- 通道② 电子项符号约定待备忘录核实，当前 Z\* 报告值含离子项
  （Na +1、Si +4，取赝势价态，常量区可改）；
- 通道① 两 λ 尺度（±0.005/±0.01）斜率一致性仅打印，用于发现非线性/收敛问题。
