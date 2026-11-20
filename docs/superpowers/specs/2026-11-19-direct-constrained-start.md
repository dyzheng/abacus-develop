# 2026-11-19 LiF 直接约束启动与中性/离子态对比：开发与当前结果汇总

> **当前状态速览**
> - 18 Å，ionic 直接约束链：3.0–5.25 Å 收敛（Q≈2.000，E −840.60→−838.69 eV）；
> - 18 Å，δ-0 参考：1.2–3.0 Å 收敛，4.5/6.0 Å 自由 SCF 失败；
> - 20 Å：原子初值直接启动全部失败；free SCF + direct start 在 r=3.0 成功
>   （E=−840.5290 eV）；3.25–10.0 Å 未完成；
> - 中性态（3/7）和两态交点尚未获得。

## 1. 动机

拉伸 LiF（r2SCAN，18 Å 盒）在 r=4.5/6.0 Å 时，**μ=0 的 reference SCF**
出现 charge sloshing：120 步不收敛、E_gap ~0.1 eV、Drho 振荡。
要比较“中性原子 Li⁰+F⁰”与“离子 Li⁺+F⁻”两种价态的能量并找交点，
当前流程会先跑自由 reference SCF，正好卡在这一步。

方案：允许**跳过 reference SCF**，从第一次 SCF 起就处于 constrained 相，
用 target 文件给出的 `initial_mu` 作为初始乘子；外循环仍把 μ 调到
absolute target。这就是“直接约束启动”。

## 2. 实现

### 2.1 新输入

- `constraint_start_constrained`（Boolean，默认 False）：
  - 跳过 μ=0 reference SCF；
  - 要求 `constraint_target_mode=absolute`；
  - 要求 `constraint_branch_tol=0`（没有 reference 能量）；
  - 不支持 `constraint_alloc_switch`。
- target 文件每个 constraint 可给 `initial_mu`（数值；nspin4 spin 可为向量）：
  ```json
  {"constraints":[
    {"type":"charge","target":2.0,"atoms":[0],"initial_mu":1.10},
    {"type":"charge","target":8.0,"atoms":[1],"initial_mu":-1.38}]}
  ```
  `initial_mu` 只是初值，不是 fixed；外循环继续调整 μ。

### 2.2 初始化与权重

`ConstraintLoop::init` 在直接启动时：

- `mu_ = initial_mu`（缺省 0），`targets_ = spec.target`（absolute）；
- `phase_ = CONSTRAINED`，`mu_eff_ = mu_`，第一轮 SCF 就注入 `μ w`；
- 不再等第一次 observe 才建 MBIS 权重。

新增 `ConstraintLoop::build_initial_mbis_weights(rho, nspin)`：
从初始密度（`init_chg` 的 restart 或 atomic）求和自旋、调用
`WeightGrid::build_mbis`，并置 `mbis_constrained_built_=true`。
PW（`esolver_ks_pw::before_scf`）和 LCAO（`esolver_ks_lcao`）在
`constraint_start_constrained` 时于 `ConstraintLoop::init` 后调用它。

这样第一次 `inject_potential` 用的就是**初始密度的 MBIS 权重 + initial_mu**，
而不是零权重/零势；第一次 observe 也不会重新冻结另一套权重。

## 3. 测试

环境：r2SCAN（Libxc 5.1.7，`ENABLE_LIBXC=ON`），18 Å 盒，Li 9,9,9，
F 9,9,9+r；`ecut 40/160`，nspin1，nbands6；`device gpu`，`OMP=1`；
absolute targets `2/8`（Li⁺/F⁻）；`scf_nmax=200`，
`constraint_kappa_min=0.05`，`constraint_step_max=0.5`，
mixing 0.2，smearing 0.005。

| r (Å) | 初始 μ (Li,F) | 最终 μ | Q_Li | Q_F | residual | FINAL_ETOT (eV) | wall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3.00 | 0.457, −0.470 | 1.102, −1.379 | 2.000606 | 7.999013 | 9.9e-4 | −840.6012 | 5m25 |
| 4.50 (from 3.0) | 1.102, −1.379 | 0.159, −0.516 | 1.99752 | 8.00206 | 2.5e-3* | −839.1914 | 8m01 |
| 4.50 (rerun) | 0.159, −0.516 | 0.293, −1.064 | 2.000600 | 7.999008 | 9.9e-4 | −839.1052 | 2m52 |
| 4.75 | 0.293, −1.064 | 0.293, −1.064 | 2.000375 | 7.999232 | 7.7e-4 | −838.9520 | 1m18 |
| 5.00 | 0.293, −1.064 | 0.293, −1.064 | 2.000255 | 7.999352 | 6.5e-4 | −838.8119 | 0m59 |
| 5.25 | 0.293, −1.064 | 0.293, −1.064 | 2.000175 | 7.999428 | 5.7e-4 | −838.6881 | 1m25 |

\* 4.5 从 3.0 直接跳 1.5 Å，SCF 收敛但 outer loop 在 150 步内只到
2.5e-3；用它的密度/μ 再跑一次（200 步）即到 1e-3。

失败记录：

- 4.5→5.0 **0.5 Å 跳变**：使用 4.5 的密度/μ 直接启动，SCF 不收敛
  （FINAL_ETOT −161.9 eV，坏态）。改为 0.25 Å 步长（4.75→5.0）后，
  两次都在 1–1.5 min 内 CONVERGED。
- 5.5 的链式运行被手动中止（没有 5.5 结果）。
- 中性态（3/7）和两态交点尚未测试。

## 4. 分析

1. **直接约束启动有效**：4.5/4.75/5.0/5.25 的 SCF 都收敛，而原来 4.5/6.0
   的自由 reference SCF 不收敛。第一轮 SCF 就带约束势，避免了中性/离子
   态之间的 sloshing。
2. **需要小步长 warm start**：0.5 Å 跳变（4.5→5.0）失败，0.25 Å
   （4.75→5.0）成功；每一步用上一步的密度/tau 和最终 μ 作初值。
3. **离子态被精确钉住**：Q≈2.0002/7.9994，residual<1e-3；总 MBIS 电荷
   ≈9.9996（10 Bohr cutoff 的 deficit ~4e-4）。
4. **能量趋势**：r=4.5→5.25 的离子态能量单调上升
   （−839.105 → −838.688 eV），符合分离极限方向；3.0 的 fully ionic
   约束态比实际 ~0.96 离子性的基态高约 0.25 eV（约束态，不是基态）。
5. **交点预期**：LiF 的离子态在平衡附近和分离极限都应更低
   （IP(Li)≈5.39 eV, EA(F)≈3.40 eV，渐近 ~2 eV）；4.5–7.0 Å 很可能
   **没有中性/离子交点**。需要中性态（3/7）的数据确认。

## 5. 更多距离测试脚本

`/tmp/opencode/lif_scan/lif_direct_chain.py`（已准备，未入库）：

- 从任意一个收敛的直接启动目录继续；
- 每一步复制上一步的 charge/tau restart，解析上一步 final μ 作为
  `initial_mu`，写新的 target/INPUT，GPU 单线程运行；
- 解析 `CONSTRAINT_AUDIT`、`final status`、`FINAL_ETOT`，首个失败即停；
- 支持 `--target 2.0,8.0`（离子）或 `3.0,7.0`（中性）。

继续离子态 5.5–7.0：

```bash
python3 /tmp/opencode/lif_scan/lif_direct_chain.py \
  --start-dir /tmp/opencode/lif_scan/r5p25_ionic2 \
  --start-suffix lif_r5p25_ionic2 \
  --distances 5.50 5.75 6.00 6.25 6.50 6.75 7.00 \
  --target 2.0,8.0 --tag ionic
```

中性态需要先产生一个中性 start（例如在 r=3.0 用
`--target 3.0,7.0`、猜一个相反的 initial μ，跑通后再用脚本链式扩展）。

## 6. 代码改动

- `input_parameter.h` / `read_input_item_other.cpp`：
  `constraint_start_constrained`。
- `constraint_io.{h,cpp}`：`ConstraintSpec::initial_mu*` 与 v2 JSON 解析。
- `constraint_loop.{h,cpp}`：直接启动初始化、`build_initial_mbis_weights`。
- `esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp`：启动后构建初始 MBIS 权重。
- `constraint_io_test.cpp`：`initial_mu` 解析单测（21/21 PASS；其它
  constraint/MBIS 单测通过）。

## 7. 下一步

1. 用脚本跑完离子态 5.5–7.0；
2. 产生中性态 start 并跑 3.0–7.0；
3. 比较两态 `FINAL_ETOT`，若存在交点则用二分/插值细化；
4. 若 0.25 Å 仍失败，尝试更小步长、更大 `scf_nmax`、不同 smearing，
   或缩到 15–16 Å 盒（需说明镜像影响）。

## 8. 20 Å 盒尝试与当前最新状态（2026-11-19 更新）

### 8.1 为什么改用 20 Å

18 Å 盒无法容纳 r=10 Å（Li 在 9，F 会到 19），而且不同盒子的绝对能量不能
直接比较。按用户要求改用 **20 Å 盒**：

- Li 固定在 `(10,10,5)`，F 在 `(10,10,5+r)`；
- r=10 时 F 在 15，左右各留约 5 Å 真空；
- 20 Å 下 r=10 的最近周期镜像间距约 10 Å（镜像库仑 ~1.4 eV），
  定量交点仍建议更大盒；本轮按 20 Å 执行。

### 8.2 20 Å：原子初值 + 猜 μ 的直接启动全部失败

在 20 Å/r=3.0 直接用 `init_chg atomic` + 直接约束启动，尝试多组 μ：

| 价态 | initial μ (Li,F) | SCF | Q_Li | Q_F | FINAL_ETOT (eV) | wall |
|---|---|---:|---:|---:|---:|---:|
| ionic | 1.1017, −1.3794 | ❌ | 2.018297 | 7.981313 | −834.8304 | 1456 s |
| ionic | 0.5, −1.0 | ❌ | 2.018214 | 7.981398 | −857.8638 | 1362 s |
| ionic | 1.5, −1.5 | ❌ | 2.019413 | 7.980195 | −839.9288 | 1362 s |
| ionic | 0.0, −0.5 | ❌ | 2.018260 | 7.981352 | −818.7905 | 1447 s |
| neutral | −1.0, 1.5 | ❌ | — | — | −872.8364 | 951 s |
| neutral | −1.5, 1.0 | ❌ | — | — | −600.0954 | 1016 s |
| neutral | −0.5, 1.0 | ❌ | — | — | −806.8052 | 1076 s |
| neutral | −2.0, 2.0 | ❌ | — | — | −716.5866 | 987 s |
| neutral | 0.0, 1.0 | ❌ | — | — | −163.3992 | 1019 s |

均无 `#SCF IS CONVERGED#`；能量大幅跳动，说明原子初值 + 猜 μ 在 20 Å
不足以稳定 SCF。**不是 direct start 本身失效，而是初值太差**。

### 8.3 20 Å：先用自由 SCF 产生初值，再做直接约束启动

在 20 Å/r=3.0 先跑一次**无约束**自由 SCF（`constraint=false`，
`init_chg=atomic`）：

- **CONVERGED**，`FINAL_ETOT = −840.8044866886 eV`，wall **8m53s**。

再用这次自由 SCF 的 charge/tau 作为初值，做 ionic absolute target 2/8 的
直接约束启动（initial μ = 1.1017, −1.3794）：

- **CONVERGED**，final μ = 1.1235, −1.6241；
- Q_Li = **2.000593896**，Q_F = **7.999017425**；
- residual = **9.8×10⁻⁴ e**；
- `FINAL_ETOT = −840.5289605241 eV`，wall **3m25s**。

结论：**20 Å 必须先有一个自由 SCF 的收敛密度，再做直接约束启动**；
直接用原子初值不行。

### 8.4 当前脚本状态

`lif_compare.py` 已按此逻辑更新：

1. `ensure_free`：在 `r_start` 先跑自由 SCF，缓存 `free_r3`；
2. `start_state`：用 free 密度/tau + `initial_mu` 猜测建立 ionic/neutral start；
3. `chain_state`：从 `r_start` 按 0.25 Å 链式到 10.0 Å，复制上一步
   charge/tau 和最终 μ；
4. 首个 SCF 不收敛即停；输出 `result.txt` / `summary.csv`。

`lif_compare.py --execute` 的 20 Å 离子态链式运行在 r=3.0 之后被中止
（环境临时目录随后被清理）；因此：

- **20 Å 目前只有 r=3.0 的 ionic 结果（−840.5290 eV）**；
- 20 Å 的 3.25–10.0、中性态（3/7）和交点**还没有结果**；
- 18 Å 的 ionic 直接链结果仍为本文 §3 的 3.0–5.25；
- 18 Å 的 δ-0 参考：1.2–3.0 收敛（Q≈2.038/7.961），4.5/6.0 失败。

> 注意：`/tmp/opencode/lif_scan` 下的脚本和原始日志已被环境清理；
> 本文保留了结果和脚本逻辑，需要时可在仓库 `tools/` 下重建。

### 8.5 当前分析

1. **direct constrained start 本身有效**：18 Å 在 3.0–5.25 全部收敛；
   20 Å 在 free 密度初值下 3.0 也收敛。
2. **20 Å 的瓶颈是初值**：原子密度 + 猜 μ 失败；先自由 SCF 再约束启动
   成功。后续每个 0.25 Å 步都应继承前一步的密度/μ。
3. **中性态仍是难点**：20 Å 的 5 组猜 μ 均未收敛；需要从中性初值
   （例如把 ionic free 密度先驱动到 3/7，逐步调 μ）或换价态 start。
4. **交点未找到**：物理上 LiF 离子态在分离极限应更低
   （IP−EA≈2 eV），4.5–10 Å 很可能没有交点；但仍需中性态数据确认。
5. **盒子影响**：20 Å 在 r=10 的镜像间距 10 Å，定量交点需更大盒；
   当前结果只能作为定性趋势。

### 8.6 下一步

1. 重建 `lif_compare.py`（或放入 `tools/`），用 free 密度启动 ionic
   3.25→10.0，记录完整离子态 PES；
2. 从 ionic free 密度出发，用更宽的 initial μ 猜测 + 直接启动找到
   neutral (3/7) start，再链式到 10.0；
3. 比较两态 `FINAL_ETOT`，若存在交点用二分/插值细化；
4. 若 20 Å 镜像影响明显，重复 26 Å 盒并用同一流程。
