# 阶段 A Task A0：混合约束 schema 决策记录（纸面）

> 计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md`（已批准，日志条目 2026-09-08 (8)）。
> 本 Task 为纯纸面决策轮，无代码改动、无计算。产出 = 决策定稿记录，供 A1 的 spec 直接并入引用。
> 硬前置状态复核：**Task 2.6 判决门未闭合**（见 §5），故本轮止于 A0，不动 A1 代码。

---

## 1. 测试方案（本轮范围与验证对象）

A0 无失败测试先行环节——它是后续 A1 失败测试的"契约源"。本轮"验证"= 一致性核查：
- 决策内容与已批准计划 A0 三条目逐字一致（不新增/不缩水/不改向）；
- 决策与架构重设计评审（2026-09-08-unified-multi-constraint-redesign-review.md）的 P2/P5 约束相符：
  2.6 门未闭合前不写代码；schema 迁移须保 3 个旧用例 result.ref 链零修改；
- 决策落到现有代码事实可执行（字段名/JSON 键/守卫点已对照源码逐条核实）。

A1 启动时将依据本记录固化的断言清单写失败测试（计划的 G1 门）：
- `MixedConstraintListParsing`：charge+spin 两条 → 每条 `ConstraintSpec{kind,atoms,chan,target,mu_max}`
  正确（含 chan 由 kind 工厂推导）；旧格式 + run 级 type → 自动转换 + deprecation 标志；
  atoms 嵌套/平铺、缺省 atoms、逐约束 mu_max 缺省回退 run 级。
- `MixedGuards`：spin 约束 + nspin=1 → ERROR；`{"type":"dipole"}` → ERROR not implemented；
  空 `constraints` → ERROR；atoms 越界 → ERROR；重复 kind+atoms 组合 → WARNING（近共线预警）。

## 2. 依据与现状（Setup）

- 代码事实（已核对）：
  - `ConstraintConfig`（constraint_io.h）：`type`/`target_mode`/`mu_max`/`thr` 均为 run 级单值；
    `targets` 为 `vector<ConstraintTarget>{value, atoms}`——混合类型在现 schema 下无法表达。
  - JSON 旧格式（v1）解析 `parse_target_file`：顶层 `"targets"` + 可选顶层 `"atoms"`
    （嵌套片段或扁平列表；缺省 fragment i = atom i）。
  - INPUT 参数（read_input_item_other.cpp / input_parameter.h）：`constraint`、`constraint_type`
    （默认 charge）、`constraint_weight_type`（becke）、`constraint_target_file`、
    `constraint_target_mode`（delta/absolute）、`constraint_mu_max`（5.0 Ry）、`constraint_thr`。
  - channel 约定（constraint_observe.h / constraint_inject_pw.h）：charge 通道 Q=∫w(ρ↑+ρ↓)、
    v_eff 两自旋均 +μw；spin 通道 m=∫w(ρ↑-ρ↓)、v_eff↑+μw / v_eff↓-μw（ΔSpin 语义，μ=-λ）。
- 旧用例现状：`tests/01_PW/211_PW_constraint_h2o`、`212_PW_constraint_h2o_spin`、
  `02_NAO_Gamma/212_NAO_constraint_h2o` 全部使用 v1 JSON `{"targets":[0.1],"atoms":[[0]]}`，
  INPUT 带 run 级 `constraint_type`（charge/charge/spin…按实况）。这三文件在 A 阶段**零修改**。
- 评审链：重设计评审 P5"schema 迁移需含 3 旧用例 result.ref 链"；P2"阶段 A 代码必须在 2.6 后启动"。

## 3. 决策定稿（Results）

### D1：`target_mode` 保持 run 级（偏离重设计 §1）

- `ConstraintSpec` **不含** mode 字段；`constraint_target_mode`（delta|absolute）维持 run 级 INPUT。
- 理由：charge+spin 混合的常态是双 delta（电荷相对中性基准的偏移 + 自旋磁矩增量），
  无"同 run 内一条 delta 一条 absolute"的真实需求 → YAGNI + 减小状态面。
- 向后兼容：未来出现需求时给 schema 加可选 `"mode"` 键 + spec 字段即可，属增量不破坏。
- 守卫不变：run 级 absolute 的校准尺度 WARNING 沿用现有实现；混合 mode 的写法不存在（schema 无此表达）。

### D2：JSON 新格式 + v1 自动转换 + deprecation

新格式（v2）——唯一新入口，顶层 `constraints` 数组：

```json
{"constraints": [
  {"type": "charge", "target": 0.1, "atoms": [0]},
  {"type": "spin",   "target": 0.1, "atoms": [0]}
]}
```

逐约束字段契约：

| 键 | 必填 | 类型/语义 | 守卫 |
|---|---|---|---|
| `type` | 是 | `"charge"` / `"spin"` | 其它值（含 `"dipole"`）→ ERROR not implemented |
| `target` | 是 | number；语义随 run 级 target_mode（delta/absolute） | 数合法性沿用现有 |
| `atoms` | 否 | 扁平 int 数组（每约束一个片段）；缺省 = 片段 [i]，i 为列表内序 | 越界 [0,nat) → ERROR；空数组 → ERROR |
| `mu_max` | 否 | number > 0；缺省回退 run 级 `constraint_mu_max` | ≤ 0 → ERROR |

- 旧格式（v1）检测：顶层含 `"targets"` 键且无 `"constraints"` 键 → 走现有
  `parse_target_file` 语义（nested/flat atoms、缺省 atoms=atom i），自动转换到内部
  `ConstraintSpec` 列表：`kind = channel_from_type(run 级 constraint_type)`、
  `mu_max = run 级 constraint_mu_max`；置 deprecation 标志 → configure 成功后打一条 WARNING
  （建议改写为 v2；每个 run 仅一次）。
- **同现规则**：顶层同时含 `constraints` 与 `targets` → ERROR（不猜测语义）。
- **run 级 `constraint_type` 作用域收窄为 v1-only**：v2 下逐约束 type 胜出；若此时 run 级
  `constraint_type != "charge"`（默认值）→ 打 WARNING "per-constraint types supersede
  constraint_type"——不静默忽略显式设置。
- **deprecation 输出契约**：WARNING 不得进入 result.ref 比较口径（3 旧用例逐位复现是 G6 硬门）。
  实现时若主输出流会污染结果比对则走独立输出流（实现细节 A1 定，A6 回归把关）。
- **3 旧用例零修改**：v1 路径必须保持现有行为逐位不变（INPUT 与 JSON 均不动）。

### D3：`mu_max` 逐约束保留（保留重设计这一条）

- 每条约束独立 `mu_max`（软通道/硬通道熔断上限可不同）；`ConstraintSpec` 含 `mu_max` 字段。
- 理由：M4 熔断/融合已按分量处理，逐约束 cap 是纯数据接线；软约束需要大 cap、硬约束需要
  小 cap 熔断是真实差异。
- 不逐约束化的参数明确列出：`target_mode`（D1）与 `thr`（保持 run 级单值）——
  计划 ConstraintSpec `{kind, atoms, chan, target, mu_max}` 不含二者，照此执行。

### 数据模型契约（D4，A1 实现蓝本）

```cpp
enum class ConstraintKind { Charge, Spin };        // Dipole 明确不在 A 阶段（守卫拒绝）

struct ChannelProfile {                            // 由 kind 工厂一次性推导
    int read_up;                                   // charge: +1, spin: +1
    int read_dn;                                   // charge: +1, spin: -1
    double inj_up;                                 // charge: +1, spin: +1
    double inj_dn;                                 // charge: +1, spin: -1
};

struct ConstraintSpec {
    ConstraintKind kind;
    std::vector<int> atoms;                        // Becke 片段（A 阶段无第二类权重，不设 weight_id）
    ChannelProfile chan;                           // 工厂推导，禁手填
    double target;                                 // delta/absolute 语义由 run 级 target_mode 解释
    double mu_max;                                 // 缺省取 run 级 constraint_mu_max
};
```

- `configure_from_inputs` 输出 `std::vector<ConstraintSpec>`（取代 run 级 type 全局态）。
- 内部顺序 = JSON 顺序（v1 转换 = targets 顺序）；审计行继续 c[0..N-1]，A4 起加 `kind=` 标签。
- 关键不变量（沿用重设计 §1）：observe / inject / force 三处共享同一条 (w, chan) 对；
  charge 分量只进总势、spin 分量只进差势（逐点断言是 A3 的 G3 门）。
- 与重设计的偏差登记：a) 去掉逐约束 `TargetMode mode`（D1）；b) 去掉 `weight_id` 整数
  （A 阶段权重恒为 Becke 原子/片段，`atoms` 即定义；偶极 z 权重未实现故无需第二类权重源）。

## 4. 分析（Analysis）

- 决策 1（run 级 mode）vs 逐约束 mode：重设计把 mode 放进 spec 是"统一模型"的洁癖，
  实际引入混合 mode 的校准语义组合（delta 电荷 + absolute 磁矩？）无用户案例。守住 run 级
  使守卫面、文档面、测试面都不膨胀；偏离已获计划批准。
- 决策 2（v2 数组 + v1 自动转换）：P5 的核心诉求是"3 旧用例 result.ref 链零修改"。
  自动转换 + deprecation WARNING 满足迁移零成本，同时让新用户不再需要
  `constraint_type` 全局开关理解（逐约束 type 自解释）。同现 ERROR 与 type 冲突 WARNING
  保证"不静默错跑"纪律不因迁移破功。
- 决策 3（mu_max 逐约束保留）：三决策中唯一"保留重设计"项。它是软/硬通道真实的运行参数
  差异，且 M4 已按分量处理，接线成本≈0；若也退回 run 级，未来加回时旧 JSON 用户会得到
  静默行为变化（cap 变严格/放宽），现在落位最便宜。
- 风险与缓解：v2 解析是自研 JSON 子集解析器扩展（现 parse_target_file 同风格）——对象数组
  解析是本解析器首次碰"对象"，A1 实现需先扩解析器（仍拒绝任意 JSON，仅本 schema 白名单），
  失败测试先行锁定结构错误路径；deprecation WARNING 的输出流选择以 G6 三旧用例 result.ref
  逐位复现为验收，不预设方案。

## 5. 下一步（Next Steps）

- **Task 2.6 判决门未闭合**（本轮复核，开放项）：
  - 2.6.2：LCAO R7 pinned-μ 18 腿全量重跑（修复后二进制）未做；残余∝|Q-t| 档位检查（1e-5 档）未做；
  - 2.6.1：PW≡LCAO 同密度 Q 读数对拍 + μ* 差 + M3b trace 对拍未做；
  - 2.6.3：力矩 FD（μ=-λ 换算写死）未做；
  - 2.6.4：自旋反假收敛集成级 + LCAO 4-rank 数值一致未做；
  - 2.7：判决门闭合文档未写（力仍属"未验收"口径）。
  已有 PASS 证据：PW 18 腿 9/9（判据 0.0128555，不豁免）；LCAO O-z fixed-μ 三腿可证伪验证
  |d|=0.000364（35× 富余）+ 全轴净力≈0；plain LCAO 对照 PASS。
- **A1 启动条件** = 2.6 门闭合。建议两条待批路径：
  - 路径 1（轻量收口，匹配"不跑重测试"约束）：2.6.4 反假收敛 1 条集成 + LCAO 4-rank 单腿 +
    2.6.1 对拍 1 对（PW/LCAO 各 1 次 SCF）+ 2.6.3 力矩 FD 单腿，估 ~0.5-1 h 墙钟 np4；
  - 路径 2（原判据全量）：LCAO 18 腿 + 档位 + torque + 4-rank 全量，估 2-4 h 墙钟 np4 并行。
  两者均以 2.7 闭合文档收口（净力/补偿前力入 FD 标准检查、F_ana 补偿前后双值入档——
  rootcause-review 的两条纪律项）。
- 路径未定时，A0 已完成：本记录即 A1 spec 的 schema 蓝本；A1 的失败测试清单见 §1。

---

## 本轮记录

- 纸面决策轮，无代码改动、无计算。更新：本 spec + 计划 A0 勾选 + 开发日志条目 (9)。
