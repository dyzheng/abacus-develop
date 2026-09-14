# MODULE_IO 测试卫生 + 元数据漂移新发现 评审

> 评审对象：39146832e（item 1）+ 0625f8a87（item 2，独立小 commit）+ 新发现（sc_scf_thr/sc_drop_thr 元数据陈旧）。
> 本轮无代码改动。

## 裁定：✅ 两 item 通过；新发现批准为**独立小 commit**（`fix(input)`，用户面元数据）。

## 核实

- 两 commit 在案（item 2 仅 3 处断言改动 + spec + 日志，零运行时行为变化，符合"test-only、可回溯"的要求）；
- `ctest -R "MODULE_IO|MODULE_ESTATE_constraint"` 亲测 **56/56 全绿**（前为 53/56）；无第四处陈旧点（亲验失败点仅这 3 个）；
- 根因链属实：5735ea673 改结构默认值（1e-3→10、threshold→immediate）未同步断言。

## 新发现裁定（批准修复，独立小 commit）

用户面漂移属实（亲验）：
- `read_input_item_other.cpp:174` metadata `default_value="1.0e-3"` vs 结构 `sc_scf_thr=10`（input_parameter.h:602）；
- `:193` metadata `"threshold"` 且描述文本写 "threshold (default)" vs 结构 `"immediate"`（:603）；
- `spin_constrain.h:50-52` 注释块同步陈旧（"Default: 1e-3"、"threshold (default)"）；
- `sc_drop_thr` metadata "1.0e-2" vs 结构值（修复时一并核对 input_parameter.h 真值后改正）。

影响面：`abacus --help` 与生成文档（docs/advanced/input_files/input-main.md）持续打印错误默认值——**这是用户可见的错误文档，不是纯注释问题**，建议按 `fix(input): sync spin-constrain metadata defaults with struct defaults` 独立提交（只动 metadata/注释/生成文档，零运行时行为变化）。

## 开放队列确认

4b 半径敏感性（待锚点）、II-1 重锚定、阶段 B 立项评审（输入已齐）；FeO 继续暂缓。

---

## 本轮记录

- 评审轮，无代码改动。56/56 亲测；元数据漂移四处逐一核实（2 处 metadata + 注释块 + sc_drop_thr）。
