# commit 3（C-29 修复）+ inner_thr 三档标定 严格评审

> 评审对象：commit 098091b4d + tests/deltap_inner_thr/ 证据（未提交）+ 文档写回。
> 本轮无代码改动。

## 裁定：✅ 两项均通过。批准 test(constraint) 提交；MODULE_IO 陈旧期望值建议**单独小 commit** 而非折叠。

## 一、commit 3 核实

098091b4d 在案（13 文件：守卫、金丝雀测试、定位 spec、评审文档、证据、日志）。

## 二、inner_thr 扫描核实

| 声称 | 核实 |
|---|---|
| 数据表（212: 42/44/38/39；213: 117/63/62/64；MgO: 381/94/107/129） | summary.txt 与 mgo_crosscheck.txt 逐项一致 |
| 正确性（μ* 差 ≤0.34%、ΔE ≤4.2e-7 eV，8 run 全 CONVERGED） | 与审计文件一致——门控是纯成本旋钮的结论成立 |
| "1e-4 甜点被 MgO 证伪" | mgo_crosscheck 实测 1e-3=94 / 1e-4=107 / 1e-5=129——收紧门控在 MgO 上确实变差（内步 31→22 但每步更贵）。**自我假设被实测推翻并如实降级为"按体系调参"——这正是测量驱动纪律的样板** |
| C-29 二次独立确认 | MgO 三个点全部走"跨基组热启动"（修复前的崩溃模式）且 corrupt=0——修复有效性在多体系多档位成立 |
| 过程纪律 | 中断后全部重跑取净数据；1e-3 点逐位复现已提交数（OMP=1 纪律生效的直接证据） |

## 三、批复

1. **批准 test(constraint) 提交**：tests/deltap_inner_thr/（120 KB，无 SCF dump，合规）+ spec + 手册 §5.9/开发者指南 §3.4 写回；
2. **MODULE_IO 陈旧期望值**：建议**单独小 commit**（`fix(test): update stale sc_scf_thr_mode expectations`）——域不同（deltaspin 默认值 vs 约束工作），独立提交便于回溯；内容=3 目标期望值改 "immediate"（阈值/语义一并核对）；
3. FeO 对照继续暂缓（维持）。

## 四、阶段状态备注

优先级队列 ① C-29 ✅ ② inner_thr 标定 ✅ 均已闭合。开放项刷新：MODULE_IO 测试卫生（已批）、FeO（暂缓）、4b 半径敏感性（待锚点）、II-1 重锚定（a)+(c)）、阶段 B 立项评审（数据已齐：μ 耦合表 + 47/15 步 + INNER 收益曲线 + inner_thr 标定）。

---

## 本轮记录

- 评审轮，无代码改动。MgO 三档数据与"甜点证伪"逐项亲验；C-29 跨基组模式 corrupt=0 二次确认。
