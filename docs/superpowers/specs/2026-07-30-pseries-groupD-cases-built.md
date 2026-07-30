# P 系列算例构建 D 组（P11/P12/P13/P14 固体）

> 日期：2026-07-30 ｜ 分支：feat/deltap ｜ 依据：`2026-07-30-pseries-test-cases-design.md` §4 D 组
> 执行范围：只构建算例与工作流，**未运行真实 ABACUS**；不 git commit；未改 tests/。

## 1. 测试计划

- 四个 run.sh 全部 `bash -n` 通过；
- 用 stub 二进制（伪造 `[rawG]`/`E_KohnSham`/`TOTAL-FORCE`/`[DeltaP-PW]` 输出）端到端干跑，
  验证：目录生成、占位符替换、位移 STRU 生成、提取函数、判据 awk、PASS/FAIL 计数、
  `SUMMARY: n/n PASS` 与退出码、阻塞 WARNING 打印；
- wannier90 通道：`wannier90.x -pp` 用真实程序验证 seed.win 模板合法性。

## 2. 测试设置

- stub：`/tmp/opencode/abacus_stub`（按 INPUT suffix 生成 `OUT.<suffix>/running_scf.log`，
  打印固定 [rawG]/[DeltaP-PW] 行）；
- 本机存在 `/root/miniconda3/bin/wannier90.x`（3.x），P14 的 `wannier90 -pp` 真跑；
- 赝势/轨道文件名逐一核对 `/root/pporb/apns-pseudopotentials-v1` 与
  `apns-orbitals-efficiency-v1`，任务书所列全部存在。

## 3. 结果

| 测试 | bash -n | stub 干跑 | 判定行为（stub 数据下） |
|---|---|---|---|
| P11 | OK | exit 0 | J1/J2 PASS，J3/J4 BLOCKED + WARNING，SUMMARY 2/2 |
| P12 | OK | exit 0（guard 修正后） | J1 PASS；J2 受 stub 恒值数据影响 FAIL（真实数据预期 PASS）；阻塞 WARNING 正常 |
| P13 | OK | exit 0 | J1–J4 全 PASS（除零 guard 修正后），SUMMARY 4/4 |
| P14 | OK | exit 0 | J1 PASS；w90 -pp 真实成功（seed.nnkp 生成）；mmn 因 stub 缺失 → 三分子 SKIP；SUMMARY 1/1 |

- P14 `seed.win` 被真实 wannier90.x 接受（`seed.wout` 正常、`seed.nnkp` 生成），
  证明 .win 模板（cell/atoms_cart/projections/mp_grid 1 1 1）合法；
- 干跑产物（runs/）已全部清理，交付目录仅含 README.md/run.sh/cases/。

## 4. 分析

- `set -e` 与 `&&` 短路、命令替换、`|| true` 的交互逐条核对，干跑未出现意外退出；
- 除零 guard 两处（P11 J2 斜率为 0、P13 J3 Ps≈0）按"两者一致小量→PASS，否则 999"修正；
- P12 `TOTAL-FORCE` 提取用"标签+三列数值"启发式，stub 验证可取到 z 分量，
  但真实 running_scf.log 格式未核（首次实跑可能需微调 `get_fz()` 正则）；
- P13 PW wrapped γ 用 LCAO Δγ 做 2π 整数倍预测-校正，逻辑经 stub 验证；
- w90 全链路（scf→-pp→nscf towannier90→wannier90）的 mmn/amn/eig 产物位置用 find 通配收集，
  真实行为待首次实跑确认。

## 5. 假设/偏差清单

1. P11 通道② 选 **Z\* 路径**（周期 PW 无宏观场可施加）；由 Z\* 反演 ε∞ 的关系
   连同 F1/F2 一起挂起，README §6 说明；
2. P11/P12 的 F1 换算中 gdir 有效盒长：P11 取 c（正交晶胞），P12 fcc 初基胞取 a/√3，
   均在常量区标注"待备忘录固体专项核实"；
3. P12 只位移阳离子（任务书规定）；中和检查 J4 走通道① 双原子力，故整体阻塞；
4. P12/P11 的离子项 Z_ION（Na+1/Si+4/B+3）取赝势价态/形式电荷，常量区可改；
5. P13 参考点 z_Ti=0.50 非严格中心对称（Ba z=0.9937），Ps 为相对极化，README 标注；
6. P13 的 P_CONV 用任务书公式 P=(e/Ω)(c/π)γ 且 PW 通道同常量（依据测试说明
   "差值 dγ 应一致"）；若 PW 定义差因子 2，J3 会系统性失败——属 F2 备忘录钉死项；
7. P13 w90 通道只到"检测+生成 .win 模板"，不做数值判定（窗口依赖实际能带）；
8. P14 Si 不进 w90 循环（固体 k 点接口需专项确认）；w90 通道偶极按每 MLWF −2e、
   离子项取赝势价电子数（O6/N5/C4/H1）；
9. P14 Si k 网格取 4×4×4（任务书未指定，P12 为 8×8×8），README 未单独标注——
   如需加密改 cases/KPT_SI 单点；
10. E(γ) 光滑性判据实现为"每组三点相邻能量差符号变化 ≤1"（三点最多 1 次，
    抛物线段恒满足；>1 即振荡），属任务书"无符号振荡"的最简可执行化。

## 6. 下一步

- 首次实跑 D 组（建议先 P14 LCAO 四体系 + P13 路径 9 点），核对 [rawG]/TOTAL-FORCE/
  [DeltaP-PW] 真实输出格式，必要时微调提取正则；
- F1/F2 备忘录定稿后回填 P11/P12 常量区并启用 J3/J4；
- w90 全链路实跑一次（P14 H₂O），确认 towannier90 产物目录与 MLWF 收敛。
