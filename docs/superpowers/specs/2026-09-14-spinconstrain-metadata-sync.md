# Spin-constrain 元数据/注释默认值同步（`fix(input)` 小件）

日期：2026-09-14（上一轮 `2026-09-14-moduleio-test-hygiene.md` 发现项的批复落地）

## 1. 测试计划

**目标**：把 spin-constrain 相关参数的**文档元数据**与**代码注释**同步到结构体真值
（`input_parameter.h`），消除 `abacus --help` 与生成文档向用户打印错误默认值的问题。
**零运行期行为变化**（只动 `item.default_value` / 描述文本 / 注释）。

批复的四处漂移（用户亲验）+ 本轮排查追加：

| # | 位置 | 旧（错） | 真值（`input_parameter.h`） |
|---|---|---|---|
| 1 | `read_input_item_other.cpp:174` `sc_scf_thr` metadata | `"1.0e-3"` | `10`（`:602`） |
| 2 | `read_input_item_other.cpp:193` `sc_scf_thr_mode` metadata + 描述文本 | `"threshold"` + “threshold (default)” | `"immediate"`（`:603`） |
| 3 | `spin_constrain.h:47-53` 参数说明注释块 | `nsc (default 50)`、`sc_scf_thr ... Default: 1e-3`、`"threshold" (default)` | `nsc`=5、`sc_scf_thr`=10、`sc_scf_thr_mode`="immediate" |
| 4 | `read_input_item_other.cpp:162` `sc_drop_thr` metadata | `"1.0e-2"` | `1e-3`（`:605`） |
| 5 | （本轮排查追加）`esolver_ks_lcao.cpp:583` 注释块 | `"threshold" (default)` | `"immediate"` 为默认 |

**通过判据**：

1. `abacus --help sc_scf_thr|sc_scf_thr_mode|sc_drop_thr` 打印的 `Default:` 等于结构体真值；
2. 元数据一致性脚本在 spin-constrain 块内 **0 条真漂移**；
3. `ctest -R "MODULE_IO|constraint"` 仍 56/56（零行为变化）；
4. 全仓库无残余 `threshold (default)` / `sc_scf_thr ... 1e-3` 类默认值陈述（历史 spec/plan 除外）。

## 2. 测试设置

- 构建：`build_rel/abacus_basic_para`（Release，当前 HEAD，改动后增量重建）；`build/`（Debug，单测）。
- 命令：
  - 漂移扫描：python 脚本解析 `read_input_item_other.cpp` 的 `Input_Item("x")` →
    `item.default_value` 与 `read_sync_*(input.x)`，再取 `input_parameter.h` 同名成员初值比较；
  - 用户可见面：`OMP_NUM_THREADS=1 ./abacus_basic_para --help <param>`（改前/改后各一次）；
  - 回归：`ctest -R "MODULE_IO|constraint"`；
  - 生成文档链：`./abacus_basic_para --generate-parameters-yaml` +
    `python docs/generate_input_main.py`（**仅在 /tmp 试跑，未入库**，见 §4）。

## 3. 结果

改动（3 个源文件，全部为 metadata/注释，**无逻辑改动**）：

- `source/source_io/module_parameter/read_input_item_other.cpp`：`sc_drop_thr` `"1.0e-2"`→`"1.0e-3"`；
  `sc_scf_thr` `"1.0e-3"`→`"10"`；`sc_scf_thr_mode` `"threshold"`→`"immediate"` 并把描述中的
  “threshold (default)” 移到 “immediate (default)”。
- `source/source_lcao/module_deltaspin/spin_constrain.h`：注释块 `nsc (default 5)`、
  `sc_scf_thr ... Default: 10`、`sc_scf_thr_mode: "immediate" (default, ...)`。
- `source/source_esolver/esolver_ks_lcao.cpp`：`sc_scf_thr_mode` 注释块把默认标到 immediate。

`--help` 实证（前 → 后）：

```
sc_scf_thr        Default: 1.0e-3            -> 10
sc_scf_thr_mode   Default: threshold         -> immediate
                  "- threshold (default)"    -> "- immediate (default)"
sc_drop_thr       Default: 1.0e-2            -> 1.0e-3
```

其余验证：元数据扫描在 spin-constrain 块内 0 条真漂移（`sc_mag_switch`/`sc_thr`/
`sc_direction_only` 的 "False" vs "false"、"1.0e-6" vs "1e-06" 为格式等价，非漂移）；
`ctest -R "MODULE_IO|constraint"` **56/56** 仍全绿。

## 4. 分析

- 根因：`5735ea673`（`sc_strategy` 简化）把 `nsc` 的元数据一起改了、却漏改
  `sc_scf_thr`/`sc_scf_thr_mode` 的元数据与 `spin_constrain.h` 注释；`sc_drop_thr`
  的元数据则从更早的统一参数化提交起就没对上。三个面（结构体默认 / 元数据 / 注释）
  因此长期互相矛盾，其中**元数据面是用户可见的**（`--help` 与生成文档都读它）。
- **未落地的部分（需单独裁定，本轮如实上报）**：`input-main.md` 的修正链路是
  `abacus --generate-parameters-yaml > parameters.yaml` → `generate_input_main.py`，
  而 `docs/parameters.yaml` 自 2026-05 起就没再生成过。用当前 HEAD 重建它会**首次公开
  48 个参数**（28 个 `deltap_*` + 12 个 `constraint_*` + `sc_strategy`/`sc_acceleration_*` 等），
  其中 `constraint_*` 正是在开发中的框架 —— 这属于“发布哪些未公开参数”的项目级决定，
  不属于本小件，故**未动 `parameters.yaml` 与 `input-main.md`**（重建结果留在 `/tmp` 备查）。
  另注：`parameters.yaml` 头部写明 “Do not edit manually”，不能手工局部改。
- 另一处**手写**用户文档漂移（本轮只上报）：`docs/advanced/scf/spin.md` 的 DeltaSpin
  参数表仍写 `nsc`=100、已删除的 `sc_scf_nmin`、`sc_scf_thr`=1.0e-4、`sc_drop_thr`=1.0e-2，
  且 `sc_lambda_strategy` 仍列 `linear_response`/`augmented_lagrangian`/`hybrid_delayed`
  （现仅 `bfgs`/`linear_scan`）—— 需要一次独立的 DeltaSpin 指南刷新。

## 5. 下一步

1. **待裁定**：是否重建 `docs/parameters.yaml` + `input-main.md`（会首次公开 48 个参数，
   含 `deltap_*`/`constraint_*`）；命令已备好；
2. **待裁定**：刷新手写指南 `docs/advanced/scf/spin.md`（≥6 处陈旧行，含已删除参数）；
3. 回到开放队列：4b 半径敏感性（待锚点）、II-1 重锚定、阶段 B 立项评审；FeO 暂缓。
