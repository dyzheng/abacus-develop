# MODULE_IO 测试卫生：同步陈旧 `sc_scf_thr` / `sc_scf_thr_mode` 默认期望

日期：2026-09-14（优先级队列开放项「MODULE_IO 测试卫生」，用户批复为独立小 commit）

## 1. 测试计划

**目标**：消除 3 个长期红灯的 MODULE_IO 单元测试目标——自 2026-05-27 的
`5735ea673 feat(deltaspin): simplify execution strategy with sc_strategy parameter`
起，它们的期望值与**有意变更**的默认值不同步：

| 测试目标 | 失败断言 | 期望（陈旧） | 实际（当前默认） |
|---|---|---|---|
| `MODULE_IO_input_test_para`（#241，np=1） | `read_input_ptest.cpp:436/437` | `1e-3` / `"threshold"` | `10` / `"immediate"` |
| `MODULE_IO_input_test_para_4`（#242，np=4） | 同上（同一二进制） | 同上 | 同上 |
| `MODULE_IO_read_item_serial`（#281） | `read_input_item_test.cpp:1701` | `"threshold"` | `"immediate"` |

**通过判据**：

1. 3 个目标全部转绿；
2. `ctest -R "MODULE_IO|constraint"` 56 项全绿（此前 53/56）；
3. **零运行期行为改动**——只改测试断言，不动 `input_parameter.h` 默认值与任何逻辑；
4. 全仓库确认没有第 4 处同类陈旧断言。

**范围裁定**：`5735ea673` 的默认值变更是上游有意设计（commit message 明列
`sc_scf_thr: 1e-3 -> 10`、`sc_scf_thr_mode: threshold -> immediate`，属于
`sc_strategy` 简化的一部分），因此以**结构体默认值为真值**、修测试期望，而不是回退默认值。

## 2. 测试设置

- 二进制：`build/`（Debug，`BUILD_TESTING=ON`，当前 HEAD），`cmake --build . --target
  MODULE_IO_input_test_para MODULE_IO_read_item_serial -j 14`。
- 用例数据：`source/source_io/test/support/INPUT` —— 该文件设了 `sc_mag_switch`/`sc_thr`/
  `nsc`/`nsc_min`/`alpha_trial`/`sccut`，但**没有** `sc_scf_thr` / `sc_scf_thr_mode`，
  故读完 INPUT 后这两项保持结构体默认值（`read_input_ptest.cpp` 断言的是该默认值）。
- `read_input_item_test.cpp::Item_test2` 用默认构造的 `Parameter param;` 直接断言结构体默认值。
- 对照证据：`git log -L 602,603:source/source_io/module_parameter/input_parameter.h` 与
  `git show 5735ea673 -- source/source_io/module_parameter/read_input_item_other.cpp`。
- 单测不涉及多线程数值，未加 `OMP_NUM_THREADS`。

## 3. 结果

修改（2 个文件、3 行断言，仅测试）：

- `source/source_io/test/read_input_ptest.cpp:436-437`：`1e-3`→`10.0`，`"threshold"`→`"immediate"`；
- `source/source_io/test_serial/read_input_item_test.cpp:1701`：`"threshold"`→`"immediate"`。

```
1/3 Test #241: MODULE_IO_input_test_para ........   Passed    0.36 sec
2/3 Test #242: MODULE_IO_input_test_para_4 ......   Passed    0.42 sec
3/3 Test #281: MODULE_IO_read_item_serial .......   Passed    0.17 sec
100% tests passed, 0 tests failed out of 3

ctest -R "MODULE_IO|constraint":  100% tests passed, 0 tests failed out of 56
```

改前红灯原文（用于回溯）：`param.inp.sc_scf_thr` Which is: 10 vs 0.001；`sc_scf_thr_mode`
Which is: "immediate" vs "threshold"。

第 4 处同类断言排查：`rg 'sc_scf_thr_mode, "threshold"|sc_scf_thr, 1e-3' source/ tests/`
只命中上述 3 处（其余 `sc_scf_thr_mode == "threshold"` 命中是 `+if` 分支判断，不是期望值）。

## 4. 分析

- **根因**：`5735ea673` 改了结构体默认值却漏改对应测试期望，两个文件自此长期红灯；
  这些是**测试卫生**问题，与约束框架工作无关（用户据此要求独立 commit 便于回溯）。
- **无行为风险**：断言是"缺省标签 → 结构体默认值"的读取结果，改为当前默认即为正确语义。
- **发现但本轮未改（同族、需单独裁定）**：`source/source_io/module_parameter/read_input_item_other.cpp`
  的**文档元数据**仍是旧值——`sc_scf_thr` 的 `item.default_value = "1.0e-3"`（L174）、
  `sc_scf_thr_mode` 的 `item.default_value = "threshold"`（L193），而 `5735ea673` 把
  `nsc` 的元数据从 `"100"` 同步成了 `"5"`、唯独漏了这两条 ⇒ `abacus --help` 与
  由 `parameters.yaml` 生成的 `docs/advanced/input_files/input-main.md` 会继续显示旧默认值。
  同族还有 `sc_drop_thr`（元数据 `"1.0e-2"` vs 结构体 `1e-3`）与
  `source/source_lcao/module_deltaspin/spin_constrain.h:52` 的注释
  `"threshold" (default, ...)`。**本轮按批复只做测试期望同步，元数据/注释留给后续小件**。

## 5. 下一步

1. （可选小件）同步 `sc_scf_thr` / `sc_scf_thr_mode`（及 `sc_drop_thr` + `spin_constrain.h`
   注释）的文档元数据到 `10` / `"immediate"` / `1e-3`，并重新生成 `input-main.md`；
2. 回到开放项：4b 半径敏感性（待锚点）、II-1 重锚定、阶段 B 立项评审（输入数据已齐）；
3. FeO 对照维持暂缓（用户已批复）。
