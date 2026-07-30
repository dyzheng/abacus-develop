# P 系列测试算例与工作流构建——本轮记录

> 日期：2026-07-30 ｜ 分支：feat/deltap ｜ 关联设计：`2026-07-30-pseries-test-cases-design.md`

## 1. 测试计划

- 为 P01–P18 全部 18 个测试构建自包含算例目录（README.md + run.sh + cases/）
- 验证：`bash -n` 全部脚本；真实 ABACUS 冒烟验证提取键与换算链

## 2. 测试设置

- 四个并行构建组：A(P01/P02/P08/P09)、B(P03/P04/P05/P15)、C(P06/P07/P10/P16/P17/P18)、D(P11/P12/P13/P14)
- 冒烟：H₂O 30 Bohr 盒、PBE、LCAO（O 2s2p1d / H 2s1p）、Γ 点、`abacus_basic_para`

## 3. 结果

### 3.1 构建产出
- 18 个测试目录全部建成，各含 README.md（中文：目的/结构/用法/判据/阻塞状态/风险）+ run.sh（自动判据 PASS/FAIL + 退出码）+ cases/
- `bash -n`：18/18 通过
- 各组建组文档：`2026-07-30-pseries-group{A,B,C,D}-cases-built.md`

### 3.2 冒烟发现与修复（本轮关键）

| # | 发现 | 证据 | 修复 |
|---|------|------|------|
| 1 | **gamma_only 必须为 0**：DeltaP LCAO 路径只支持 multi-k（complex），gamma_only=1 时静默跳过 γ 输出 | 源码 `esolver_ks_lcao.cpp:832` 警告 + 冒烟无 rawG 输出 | 确认全部 39 处 INPUT 模板 gamma_only=0（无需改） |
| 2 | **KPT 沿 gdir 需 ≥2 点**：Wilson 环 k-string 无法从 1 k点构建 | 冒烟 `Gamma 1 1 1` 无 rawG；改 `1 1 2` 后正常输出；参考算例 deltap_h2o_polarizability 亦用 1 1 2 | 全部分子 KPT 改 `1 1 2`；P02/P05/P15 多方向测试 run.sh 改为按 gdir 动态生成 KPT（2 1 1 / 1 2 1 / 1 1 2） |
| 3 | **F2 自旋因子实测确认 = 2**（nspin=1）：Σγ_raw=−12.718 rad → unwrap(−π,π] = −0.1514 → ÷2 → μ=1.838 D，与实验 1.855 D 差 0.017 D（判据 0.05 D 内） | 冒烟实测 | P01/P02/P04/P05/P06/P10/P11/P14/P15 常量区写入 F2（除子或乘 0.5），附实测证据注释；raw γ→μ 一律先 unwrap |
| 4 | **F1 公式源码确认**：`E_eff_au = −λ·π/(2a)`（Ha），`esolver_ks_lcao.cpp:823` | 源码 | 各脚本常量区 F1 公式与此一致，无需改 |
| 5 | 本机 ecutwfc=100 + scf_thr=1e-8 的 LCAO 分子算例单点 >15 min（192³ FFT、串行 genelpa） | 冒烟超时 | 未改测试设置（生产机器上合理）；README 建议 HPC 环境运行；冒烟用 ecut50/scf_thr 1e-6 50 s 收敛 |

### 3.3 冒烟数据
```
gamma_only=1, KPT 1 1 1:  无 rawG 输出（发现 #1/#2）
gamma_only=0, KPT 1 1 2:  [rawG] Σγ_raw=-1.271779e+01  γ0=-5.4219 γ1=-3.6480 γ2=-3.6480
                          SCF 20 步收敛（ecut50, thr 1e-6），总耗时 50 s
换算: μ = (30/π)·unwrap(−12.7178)/2 × 2.541746 = 1.838 D  vs 实验 1.855 D → PASS 裕量内
```

## 4. 分析

- 发现 #2 是致命级：若不修，所有分子 DeltaP 算例跑不出 γ。冒烟测试在构建阶段拦截了该问题。
- F2=2 的实测确认使 P02/P04 平衡偶极类判据从"形式可跑"升级为"数值预期可过"；响应类（α）换算链仍保留"待备忘录定稿"WARNING，但常量区工作值已有实测依据。
- PW 通道（berry_phase、ΔP-PW）未在本机冒烟（ecut80 30 Bohr 盒成本过高）；提取键已对 tests/ 既有真实输出样例逐一核对（各组建组文档记录）。

## 5. 下一步

1. 在 HPC 环境完整跑 P02（四通道）与 P09（确定性）两个无前置 P0 测试
2. P08 九点扫描，确定 λ 线性窗口 → 喂 P01
3. F2/F3 备忘录定稿：冒烟证据（μ=1.838 D、源码 F1）已可支撑 F1/F2 定稿，剩余 F3（frozen-λ 能量记账）
4. PW 通道冒烟（小盒/低 ecut 验证提取键端到端）
5. 各组文档中记录的假设/偏差（groupA–D 文档）逐条复核关闭
