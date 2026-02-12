# /accept — 生成验收证据报告

读取最近的验收日志文件，生成格式化的验收证据报告。

## 执行步骤

1. 读取以下日志文件（如果存在）：
   - `/tmp/build.log` — 编译日志
   - `/tmp/unit_test.log` — 单元测试日志
   - `/tmp/integration_test.log` — 集成测试日志

2. 从日志中提取关键信息：
   - Gate 1: error 数量、warning 数量、二进制文件大小
   - Gate 2: 测试总数、通过数、失败数、新增 test case 列表
   - Gate 3: 回归测试结果、新增 case 状态

3. 运行 Gate 4 代码审查检查（`git diff` 分析）

4. 输出完整的验收证据报告：

```
## 验收证据

### Gate 1: 编译
- 编译结果: PASS / FAIL
- 编译日志: /tmp/build.log
- error 数量: <N>
- warning 数量: <N>
- 生成二进制: <路径和大小>

### Gate 2: 单元测试
- 新增测试文件: <列表>
- 新增测试用例: <列表>
- 测试结果: X passed, Y failed
- 测试日志: /tmp/unit_test.log

### Gate 3: 集成测试
- 回归测试结果: PASS / FAIL
- 新增 case: <列表>
- 测试日志: /tmp/integration_test.log

### Gate 4: 代码审查
- [ ] 逻辑一致性
- [ ] API 适配
- [ ] 无 debug print
- [ ] 无 WIP 代码
- [ ] 无未使用 include
- [ ] 风格一致性

### 总结
- 验收状态: PASS / FAIL
- 失败项: <列表或"无">
```

5. 如果所有 Gate 通过，建议用户 commit。
6. 如果有 Gate 失败，列出需要修复的问题。
