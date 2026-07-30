# P11　h-BN 介电常数 ε∞（可执行算例）

> 系列：DeltaP ｜ 前置：P01（**阻塞中**）｜ 设计文档：`../P11-hBN介电常数.md`

## 1. 目的

把响应验证从分子推进到周期固体：用 DeltaP total 约束在多 k 点网格下测 h-BN 面外（c 轴）
电子极化响应 dγ/dλ，换算介电常数 ε∞=1+4πα/V，并与 PW berry_phase 位移 FD 通道交叉核对，
检验 HK 修正项（k 空间规范协变项）在真 k 点网格下的正确性。

## 2. 目录结构

```
P11-hBN介电常数/
  README.md           # 本文件
  run.sh              # 主工作流：生成 runs/ → 批量运行 → 提取 → 判定
  cases/
    STRU              # h-BN 六方原胞（a=2.512 Å, c=6.692 Å, 2B+2N, Direct）
    KPT_664           # 6×6×4
    KPT_996           # 9×9×6（k 收敛对照）
    INPUT_lcao.tmpl   # DeltaP LCAO 模板（占位：SUFFIX/LAMBDA_INIT/MIXING_BETA）
    INPUT_pw.tmpl     # PW berry_phase + deltap_switch true 模板（测 γ_total）
  runs/               # 运行产物（run.sh 生成，含 results.txt）
```

## 3. 用法

```bash
bash run.sh                      # 串行
NPROC=4 bash run.sh              # MPI 并行
ABACUS=/path/to/abacus bash run.sh
```

环境变量：`ABACUS` / `PSEUDO_DIR` / `ORBITAL_DIR` / `NPROC`（默认值见 run.sh 头部）。

## 4. 工作流

**通道①（DeltaP，LCAO/genelpa）**：对 k 网格 6×6×4 与 9×9×6 各做 λ=±0.005、±0.01 Ry
四点 SCF（`deltap_gdir 3` 沿 c，`lambda_step 0.0` 冻结 λ，负 λ 用 `mixing_beta 0.3`），
读 `[rawG] Σγ_raw` → 中心差分得 dγ/dλ（两个 λ 尺度，兼作线性巡检）→
α=(F2/F1)·dγ/dλ → ε_c=1+4πα/V_cell。

**通道②（PW berry_phase FD）**：B 原子沿 c 位移 ±0.005 Å（取 B 层 z=1/4 那个原子），
PW `berry_phase 1, gdir 3` + `deltap_switch true` 读 `[DeltaP-PW] γ_total`，
两点位移 FD 得 dγ/du → Z*_zz(B)（**Z\* 路径**，见 §6 说明）。

## 5. 判据表

| 编号 | 判据 | 状态 |
|---|---|---|
| J1 | 全部算例 SCF 收敛且有 [rawG]/[DeltaP-PW] 输出 | 有效（硬判） |
| J2 | 两 k 网格 dγ/dλ 相对差 ≤2%（比值判据，不依赖 F1/F2） | 有效（硬判） |
| J3 | 两通道 ε_c 差 ≤5% | **阻塞**（F1/F2 备忘录未定稿） |
| J4 | 与文献区间相符（面外 ε∞≈3，标注 PBE 系统误差） | **阻塞**（同上） |
| — | 线性巡检：±0.005 与 ±0.01 两尺度斜率一致性 | INFO（打印不判） |
| — | 各向异性标注：仅测面外分量；文献面内(~4.5)>面外(~3) | INFO（打印不判） |

末尾输出 `SUMMARY: n/n PASS`，退出码 0/1；结果存 `runs/results.txt`。

## 6. 阻塞状态与换算链

本测试**阻塞于 P01**：λ→E 的 F1 换算（`E=−π·λ/(2a)`，a 取 gdir 方向盒边长）与
γ→μ 的 F2 自旋因子尚未定稿，凡涉及 ε 绝对值的判据（J3/J4）一律打印
`WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）`，不计入 SUMMARY。

所有换算常量集中在 run.sh 头部常量区（`F1_LAM2E`、`F2_GAMMA2MU`、`C_BOHR`、`V_BOHR3`、
`Z_ION_B`），备忘录定稿后单点修改即可。

通道② 选用 **Z\* 路径** 而非 efield 路径：周期 PW 下无宏观电场可直接施加，
故由位移 FD 得 Z\*_zz(B) 作为独立交叉核对量；由 Z\* 反演 ε∞ 需要简正模信息，
该反演关系连同 F1/F2 一并在备忘录中定稿后再启用 J3。

## 7. 风险与注意（抄自设计文档 §7）

- 体材料 λ 窗口换算中 a 取有效晶格尺度，需在备忘录中固体专项核实；
- 关注 HK 修正项的 k 并行确定性（与 P09 联动）。

补充（工程层面）：

- 通道② 位移 FD 的 Z\* 符号约定（电子项 ±）待备忘录核实，当前仅记录数值；
- 负 λ 的 SCF 收敛性较差，已默认 `mixing_beta 0.3`，仍不收敛的点会被 J1 标记 INVALID；
- h-BN 为层状材料，c=6.692 Å 含层间空隙，V_cell 用全原胞体积（含空隙），
  若备忘录改用有效层体积，ε 数值会变（常量在头部单点修改）。
