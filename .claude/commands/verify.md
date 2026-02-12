# /verify — 执行完整的 4-Gate 验收流程

对当前工作执行完整的 4-Gate 验收。必须按顺序通过所有 Gate。

## 执行步骤

### Gate 1: 编译验证

1. 运行编译：
```bash
cd /root/abacus-dftu-pw-port/build
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
```
2. 检查 build.log 中 `error:` 的数量（必须为 0）
3. 确认生成了可执行文件 `abacus`
4. 如果编译失败，停止验收，报告错误并修复

### Gate 2: 单元测试

1. 识别本次修改涉及的模块测试目录
2. 运行单元测试：
```bash
ctest --test-dir /root/abacus-dftu-pw-port/build/<测试目录> -j4 --output-on-failure 2>&1 | tee /tmp/unit_test.log
```
3. 确认所有测试 PASS
4. 确认新增/修改的 public 方法都有对应的 test case
5. 确认新增的代码分支都有对应的 test case
6. 如果缺少测试，停止验收，列出需要补充的测试

### Gate 3: 集成测试

1. 读取当前 PR 的 task instruction，找到需要运行的回归测试和新增 case
2. 运行回归测试：
```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a /root/abacus-dftu-pw-port/build/abacus -n 4 -r "<case正则>" 2>&1 | tee /tmp/integration_test.log
```
3. 确认所有回归测试 PASS
4. 确认新增 case 目录存在且文件完整
5. 如果回归测试失败，停止验收，报告偏差

### Gate 4: 代码审查

对本次修改的所有文件执行以下检查：

1. 用 `git diff` 查看所有变更
2. 逐项确认：
   - [ ] 修改与 zdy-tmp 参考代码逻辑一致
   - [ ] API 适配正确（Plus_U/PARAM.inp/参数传递）
   - [ ] 无遗留 debug print（搜索 std::cout, printf, std::cerr）
   - [ ] 无 WIP 代码（搜索 TODO, FIXME, HACK）
   - [ ] 无未使用的 include
   - [ ] 函数签名与 develop 风格一致

### 输出验收报告

所有 Gate 通过后，输出完整的验收证据（格式见 CLAUDE.md）。

如果任何 Gate 失败，输出：
- 失败的 Gate 编号和原因
- 需要修复的具体问题列表
- 建议的修复方案
