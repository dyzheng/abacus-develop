# 电荷+自旋同时约束的可行性实测（H₂O，nspin=2）

> 问题：当前版本能否同时约束电荷和自旋？方式：三组 H₂O 实测（/tmp，串行，15 Å 盒，ecutwfc=20/ecutrho=80）。无代码改动。

## 结论速览

| 场景 | 当前支持？ | 实测 |
|---|---|---|
| 同 run 多约束（同一类型、多片段） | ✅ 支持 | A/B 两组均 CONVERGED |
| **同 run 混合 charge+spin** | ❌ **不支持** | 输入无法表达；守卫实证拒绝 |

## 测试 A：电荷双片段（type=charge，nspin=2）

`{"targets":[0.1,-0.1],"atoms":[[0],[1,2]]}`（O +0.1 e，H 片段 −0.1 e）：

- **CONVERGED**（少外步）：c[0] res=+1.66e-5、c[1] res=−1.66e-5（均 <1e-4）；
- μ_O=−0.0881、μ_HH=+0.0881 Ry（纯电荷转移对的镜像乘子，物理合理）。

## 测试 B：自旋双片段（type=spin，nspin=2）

`{"targets":[0.1,-0.05],"atoms":[[0],[1]]}`（O +0.1 μB，H1 −0.05 μB）：

- **CONVERGED，但用了 47 个外步**（单自旋约束仅 3 步）：c[0] res=−6.0e-5、c[1] res=+5.3e-5；
- μ_O=−0.07236 Ry（与单约束 μ*=−0.07234 一致）、μ_H1≈−5.0e-5 ≈ 0；
- 分析：两约束近共线——H1 权重区小，m 读数主要由 O 区贡献，H1 靶点几乎被 O 约束"顺带"满足；轨迹前 4 步残差振荡（+0.0036→−0.0090→−0.0149），是对角 secant 在近共线约束对下的预期行为（T-7p 教训同类），最终自愈未触熔断。**多自旋约束可用但收敛成本显著高。**

## 测试 C：混合输入守卫

`constraint_type charge_spin` → before_scf **WARNING_QUIT**（EXIT=1）：
`constraint_type="charge_spin" is not implemented in phase 2 (only "charge" and "spin")`。

代码事实：`ConstraintConfig.type` 为 run 级单值，`channel_from_type(cfg_.type)` 在 observe/inject/force 四处全局调用——混合约束在当前 schema 下无法表达。

## 实现混合约束的改动评估（框架内，无新算法）

M2/M3a/M6 均已按 channel 参数化，M4 逐分量与 channel 无关。所需改动：
1. 靶文件 JSON 增加逐约束 type 字段（如 `{"targets":[...],"atoms":[...],"types":["charge","spin"]}`），M7 解析+守卫（spin 型约束要求 nspin=2）；
2. ConstraintLoop 持有逐约束 channel 数组，observe/inject/force 三处按分量取 channel（注入时 charge 分量同号、spin 分量 ±μ 叠加进同一 v_eff）；
3. 审计行/输出标注每分量类型。
估计 1–2 天含测试。建议立项为 Task 2.4.1 或三期项（非当前判决门阻塞项）。

---

## 本轮记录

- 实验轮，无代码改动。用例在 /tmp/mixed/{a_charge,b_spin,c_guard}（未入库；如需保留可按 P4 先例注册 tests/）。
