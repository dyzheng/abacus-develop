# DeltaP Force/Stress T7-c：B-7（H_HK 解析力）实现与排查记录

> 2026-08-03 · 依据 `2026-08-02-deltap-force-stress-t7b.md` §6 Next steps 第 1 项执行。
> 结论先行：**B-7 主体已实现并进入 TOTAL-FORCE（x/y 分量精确吻合），但 z 方向
> 出现一个均匀 −2.24e-3 Ry/Bohr 的异常偏移，来源排查进行中**。E_HK 数值本身
> 已通过 FD 交叉验证（∂E_HK/∂R FD ≈ +2.86 eV/Å，与 T7-b 隔离推断 2.64 吻合）。

---

## 1. Test plan

1. 推导 H_HK 解析力公式（冻结 C 的 Hellmann–Feynman 形式）；
2. 在 `deltap::DeltaP` 中实现 `compute_hk_force()`（串行，nproc==1 + nrow==ncol 守卫）；
3. 接入 esolver `cal_force` + `FORCE_STRESS`（静态存储，使 TOTAL-FORCE 打印包含 F_HK）；
4. 验证：
   - E_HK 数值合理性（与 H_HK 矩阵元量级、λ、T 一致）；
   - F_HK 与 TOTAL-FORCE 的对应关系（逐分量核对）；
   - 组① O1-z FD 复验（残差应下降 B 项量级）。

## 2. 公式（冻结 C 的 HF 力）

H_HK(k_j) = sym[(i/2)·S_dk(k_j,k_{j+1})·C(k_{j+1})·W_eff·C†(k_j)]，
W_eff[n] = Σ_I λ_I Σ_lm |D_I^{(j)}[lm][n]|²（D 为 k_j 处 SMO 投影，见
`deltap_wannier.cpp compute_hk_correction`）。

利用 C 正交归一（C_L†C_L = I）与 W/f 对角性：
E_HK = Σ_j Re[Tr(H_sym(k_j)·DM_{k_j})] = Re[(i/2)·Σ_j Σ_n f_n W_n (C_L† S_dk C_R)_{nn}]

冻结 C（SCF 收敛时 ∂E'/∂C = 0，HF 定理成立）：
F_Jα = −∂E_HK/∂R_Jα = −Re[(i/2)·Σ_j Σ_n f_n ( W_n·∂T_nn/∂R + ∂W_n/∂R·T_nn )]

- ∂T/∂R = C_L†(∂S_dk/∂R)C_R：S_dk 条目对 bra/ket 原子的导数用
  `snap(cal_deri=1)` 导数块（∂ov/∂R_bra=+g，∂ov/∂R_ket=−g），外加 bra 侧
  相位项 ∂phase/∂R_bra = −2πi·dkv_α/lat0·phase（S_dk 相位含 −dk·τ_bra）；
- ∂W/∂R = 2Σ_I λ_I Σ_lm Re[D*·conj(∂S_k/∂R)·C]，∂S_k/∂R 用投影集
  （first-ζ per l）导数块，S_k 相位 exp(2πi k·R) 无原子位置依赖。

实现：`deltap.h` 新增 `compute_hk_force()` 声明；`deltap_wannier.cpp` 实现
（~280 行，串行全矩阵；MPI nproc>1 时 WARNING 跳过，与 compute_hk_correction
串行-only 状态一致）。输出 `force_out`（nat*3，Ry/Bohr）+ `e_hk_out`（Ry）。

## 3. 接入方案（含两次排错）

### 3.1 问题①：F_HK 不进 TOTAL-FORCE 打印

初版在 `ESolver_KS_LCAO::cal_force` 中于 `fsl.getForceStress(...)` **之后**
直接 `force(iat,a) += f_hk[...]` —— 但 TOTAL-FORCE 打印发生在
`FORCE_STRESS.cpp:684`（getForceStress 内部），先于 esolver 的加法，因此
打印不含 F_HK（relax 实际使用的矩阵含，但日志报告不含）。

**修法**：镜像既有 `store_lambda_for_force` 模式 —— `deltap_lcao.h/.cpp`
新增静态 `s_stored_hk_force/s_stored_e_hk`（+ `store_hk_force_for_force` /
`get_stored_hk_force`）；esolver `cal_force` 计算后存储；`FORCE_STRESS.cpp`
deltap 汇总块（`fcs += force_deltap` 处，打印之前）追加 f_hk。

### 3.2 问题②：store 时机太晚（static 为空）

esolver 的 compute+store 块最初在 getForceStress **之后** → FORCE_STRESS
读到空 static（`[fsdbg] hk_force size=0` 实证）。

**修法**：把 compute+store 块移到 getForceStress 调用之前（psi/pelec 在
cal_force 时已收敛可用）。修复后 TOTAL-FORCE 正确变化（x/y 精确吻合）。

### 3.3 问题③：z 方向均匀 −2.24e-3 Ry/Bohr 偏移 —— 已定位机制（净力修正）

**症状**：启用 f_hk 加法后，TOTAL-FORCE 相对禁用基准的变化：
- H1-x +0.0627 eV/Å = f_hk[3]×25.71 **精确吻合**；
- H2-x −0.0627 eV/Å = f_hk[6]×25.71 **精确吻合**；
- O1-z +0.1035 eV/Å ≠ f_hk[2]×25.71 = +0.1612（差 −0.0577）；
- H1-z/H2-z −0.0518 ≠ f_hk[5]×25.71 = +0.0059（差 −0.0577）。

即 **actual = f_hk + uniform_z(−2.24e-3 Ry/Bohr)**，x/y 无此偏移。

**已排除**：
- f_hk 存储/读取不一致（FORCE_STRESS 加法点打印 f_hk 与 cal_force 打印逐位一致）；
- 加法代码问题（机械 `fcs += f_hk[iat*3+i]`）；
- compute_hk_force 调用副作用改 SCF（禁用加法后 TOTAL-FORCE 与基线逐位一致，
  且 E'/λ 15 位一致）；
- psi fix_k 残留（cal_dm_psi 内部自行 fix_k）。

**根因（已锁定）**：`FORCE_STRESS.cpp` 力汇总循环末尾对 **fcs 做净力修正**：
`sum += fcs(iat, i); ... fcs(iat, i) -= sum/nat;`（gate/efield 关闭时，源码
560 行附近）。我的 f_hk z 分量**不满足力守恒**：

```
Σ_z f_hk = 6.2696e-3 + 0.2293e-3 + 0.2293e-3 = 6.7282e-3 Ry/Bohr
sum/nat  = 6.7282e-3 / 3 = 2.2427e-3 Ry/Bohr   ← 正好等于观察到的 uniform 偏移
Σ_x f_hk = 0  → 修正为 0 → x/y 精确吻合 ✓
```

**排查证据链**（均为逐位可重复实验，二进制/目录见 §7）：
1. 启用 vs 禁用 f_hk 加法（仅 `if(false)` 差异）：f_hk、force_deltap 逐位
   一致；x/y preprint 差 = f_hk；z 差 = f_hk − 2.2427e-3 uniform；
2. 同一二进制两次运行 preprint 逐位一致（无随机性）；
3. 禁用 compute_hk_force 调用后 preprint 回到基线（-3.00775012e-2），
   证明 compute_hk_force 无污染副作用；
4. ASAN 全程 0 错误（排除越界写/UAF）；
5. 数学闭合：uniform 偏移 ≡ sum/nat 修正（非 bug、非未初始化读）。

**结论**：TOTAL-FORCE 的 z 偏移是 ABACUS 净力修正的正常行为；**真正待修问题
是 f_hk 的 z 分量不满足平移不变性（Σ_J F_Jz ≠ 0）**。x/y 守恒（Σ=0）说明
∂T/∂R、∂W/∂R 的 x/y 链正确；z 不守恒的候选根因：
- (a) S_dk 相位项 ∂phase/∂R_bra = −2πi·dkv_α/lat0·phase 在整体平移下无
  对应补偿（bra+ket 项残留 −2πi·dkv/lat0·phase·ov），E_HK 若平移不变则该
  项应有 C 侧或 u_k 相位补偿；
- (b) E_HK 定义本身（compute_hk_correction 的 S_dk 相位约定）在整体平移下
  非不变，则均匀分量应丢弃（ABACUS 修正恰好如此）——需数值区分 (a)/(b)：
  跑整体平移 ±δ 的冻结-λ SCF 看 E_HK 是否不变。

## 4. 测试结果

### 4.1 E_HK 数值与 FD 交叉验证（关键）

| 量 | 值 |
|----|-----|
| E_HK(base, λ*) | +2.5838e-3 Ry |
| E_HK(O1-z +δ, 组① 冻结 λ) | +2.0256e-3 Ry |
| E_HK(O1-z −δ, 组① 冻结 λ) | +3.1391e-3 Ry |
| ∂E_HK/∂R FD | −(2.0256−3.1391)e-3/0.01 Bohr = **+0.1114 Ry/Bohr = +2.86 eV/Å** |
| T7-b 隔离推断 B | ≈ +2.64 eV/Å |

E_HK FD ≈ 隔离推断（差 0.2 eV/Å 属 C-响应/λ* 差异），说明 E_HK 记账与
H_HK 构造自洽；解析 F_HK 是否等于 ∂E_HK/∂R 的冻结-C 部分需待 3.3 解决后
复验（冻结-C 与重收敛-C 的差 = ∂E_HK/∂C·dC/dR，属预期差异）。

### 4.2 f_hk 分量（base h2o1，Ry/Bohr）

```
O1: (4.07e-12, 2.56e-12, +6.2696e-3)
H1: (+2.4372e-3, -1.76e-12, +2.2933e-4)
H2: (-2.4372e-3, -8.0e-13, +2.2933e-4)
```

∑F_x = 0 ✓（O1/H1/H2 x 分量守恒）；z 分量 ∑ = +6.7e-3 ≠ 0 —— 与 3.3 的
uniform 偏移并存，表明 z 链尚未自洽（守恒破坏正是待查项的表现）。

### 4.3 TOTAL-FORCE（base，含 F_HK）

```
O1  (3.3e-6, 2.0e-6, -0.6698)     H1 (-0.2583, -1e-6, +0.3349)     H2 (+0.2582, -1e-6, +0.3349)
```

x/y 与 "基准确 + f_hk" 逐位一致；z 差 uniform −0.0577 eV/Å（见 3.3）。

### 4.4 组② 完整矩阵

已补入 `2026-08-02-deltap-force-stress-t7b.md` §3.5：5/9 FAIL，失败模式与
组① 一致，`script_exit=1`。

## 5. 代码改动（T7-c B-7，临时调试代码待清理）

- `source/source_lcao/module_deltap/deltap.h`：`compute_hk_force` 声明；
- `source/source_lcao/module_deltap/deltap_wannier.cpp`：实现（含 `#if 1`
  临时 hkdbg 打印）；
- `source/source_lcao/module_operator_lcao/deltap_lcao.h/.cpp`：静态
  `s_stored_hk_force/s_stored_e_hk`；
- `source/source_esolver/esolver_ks_lcao.cpp`：cal_force 中 compute+store
  （getForceStress 之前）+ `[DeltaP HK-force]` 诊断行；
- `source/source_lcao/FORCE_STRESS.cpp`：fcs 汇总块追加 f_hk（当前含
  `if (false)` 临时禁用、`[fsdbg*]` 打印，待清理）；
- `tests/deltap_fd_force/README.md`：双组协议文档（T7-b 收尾）。

## 6. Next steps

1. **修复 f_hk 平移不变性**：先做整体平移 ±δ（冻结 λ）数值实验区分
   (a)/(b)（E_HK 是否平移不变）；若 (a) 则补齐 ∂T/∂R 的补偿项（S_dk 相位
   导数与 C/u_k 相位的配对），若 (b) 则在文档记录 E_HK 相位约定并让均匀
   分量由 ABACUS 净力修正处理；修复后 f_hk 应满足 Σ_J F_Jα = 0 且
   TOTAL-FORCE 逐分量 = 基线 + f_hk（无 uniform 修正差）；
2. 清理全部临时调试代码（hkdbg/fsdbg/`&& false` 禁用调用），恢复
   `[DeltaP HK-force]` 单行诊断；
3. 组① O1-z FD 复验：残差应从 +5.33 下降 B 项（~+2.6 eV/Å 量级，实际值以
   修复后为准）→ 若收敛到 ~+2.7（≈ A2+C 剩余）则 B-7 关闭；
4. A2（∂τ/∂R）+ τ 单位决策（B-6 BLOCKER）后重跑组①/组② 全矩阵；
5. 应力路径 S1（H_HK 应力未实现，本轮回合记录 LIMITATION）；
6. MPI：H_HK 力串行-only（与 compute_hk_correction 一致），文档标注。

## 7. 数据文件

- `/tmp/fd_t7b/b7/base/`：with/disabled 构建的 base 运行（run.log 含
  hkdbg/fsdbg 诊断与 TOTAL-FORCE）；
- `/tmp/fd_t7b/b7/base2|base3/`：禁用版（if(false)）两次运行，preprint
  逐位一致（-3.00775012e-2 O1-z）；
- `/tmp/fd_t7b/b7/base4|base5/`：启用版两次运行，preprint 逐位一致
  （-2.60506902e-2 O1-z）；`base6/`：禁用 compute_hk_force 调用（preprint
  回基线）；
- `/tmp/fd_t7b/b7/asan/`：ASAN 运行（0 错误）；
- `/tmp/deltap_lcao.cpp.abs_tau.bak` 等为 T7-b 临时构建备份（工作树已还原
  基线）。

### 3.4 用户评审补充（2026-08-03 二轮）：机制确认 + 判据实验设计

- 确认 §3.3 定位：`FORCE_STRESS.cpp:595-601` 均值扣除 `fcs -= sum/nat`，
  f_hk z 不守恒（Σ=+6.7e-3）→ −2.24e-3 均摊；x/y Σ=0 不受影响。
- **真 bug 唯一候选：ΣF_HK,z ≠ 0（平移不变性破坏）**，两种解读：
  - (a) 实现漏项：∂S_dk/∂R 仅 bra 侧相位导数（相位约定 exp(2πi(dk·R−dk·τ_bra))），
    闭合 link/ket 侧未微分相位在均匀平移下不抵消——只沿 dk 方向出现，与症状同构；
  - (b) 半物理：均匀平移 Δ 下 S_dk 获得全局相位 exp(−2πi·dk·Δ)，E_HK 随之
    变化——Berry 相位随原点移动的真实物理，补偿项正是未实现的 C 项
    （+λ·dγ/dR，T7-b 估 ~0.24 eV/Å，与 6.7e-3 Ry/Bohr 同量级）。若 (b)，
    f_hk 不守恒不是错，缺 C 才是错。
- **判据实验**：
  - ① 全体原子同移 δz，对 E_HK 做 FD：若 ∂E_HK/∂uniform ≠ 0 且
    ≈ −d(escon)/d(uniform) → (b) 成立（f_hk 无错，补 C 项）；
  - ② gdir 改 1 重跑：uniform 偏移应跟到 x 方向；若仍在 z 则另有 z 专属 bug。
- **FD 测试失败原因分解**（用户评审）：
  - 噪声预算：判据 5e-4 Ry/Bohr × 2δ=0.01 Bohr ⇒ 能量噪声须 < 5e-6 Ry；
  - ecutwfc=50 + ecutrho=200（默认 4×）下 egg-box 噪声底 ~6e-3 eV/Å（纯 SCF
    对照实测），已达判据一半——判据级判定不可信，大残差结论（隔离实验）稳健
    （同网格差分、一阶相消）；
  - T7-c 收尾验收处方：LCAO ecutwfc=100 + 显式 ecutrho≥400 重跑；PW FD
    （O4，未做）ecutwfc 80–100；scf_thr 建议 1e-8；
  - 其余失败：B-7 未闭环（主导）、B-6 τ 单位（两端错 L 倍）、A2/C 缺失、
    组② 协议缺陷（target=0 而 γ*≈(−5.5,−3.6,−3.6) 约束未激活 + γ 分支交换
    → 建议 target 改 γ*(base) 组② 才有判决意义）。
