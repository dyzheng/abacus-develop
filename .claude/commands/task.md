# /task — 加载并执行 PR task instruction

加载指定 PR 的 task instruction 并开始执行。用法: `/task <PR编号>`

参数 $ARGUMENTS 指定 PR 编号（如 1, 2, 3...）。

## 执行步骤

1. 读取 task instruction 文件：
   `/root/pm-agent/projects/f-electron-scf/tasks/PR-$ARGUMENTS-task-instruction.md`

2. 读取验收规范：
   `/root/pm-agent/projects/f-electron-scf/plans/review-and-testing-spec.md`

3. 解析 task instruction 中的以下关键信息：
   - 需要修改的文件列表
   - 每个文件的具体修改内容
   - 前置依赖（确认已满足）
   - 测试要求（MANDATORY section）

4. 在开始编码前，先读取所有需要修改的文件和相关的参考文件（zdy-tmp 对应文件）

5. 按照 task instruction 中的修改顺序逐个执行：
   - 头文件声明 → 源文件实现 → CMakeLists 更新
   - 每完成一个逻辑变更后立即编译验证（Gate 1）

6. 代码修改完成后，按顺序执行：
   - 编写单元测试（Gate 2 要求的所有 test case）
   - 创建集成测试 case（Gate 3 要求的所有 case）
   - 运行完整的 4-Gate 验收（调用 /verify）

7. 验收通过后，输出验收证据报告

## 重要提醒

- 严格按照 task instruction 执行，不要自行扩大修改范围
- 遇到 task instruction 中的矛盾或遗漏时，必须暂停并询问用户
- 需要修改非指定文件时，必须说明原因并请求批准
- 所有 4 个 Gate 必须通过才能声称任务完成
- 不要在验收证据不完整时声称任务完成
