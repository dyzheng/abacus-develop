# CLAUDE.md — f-electron-scf 移植项目

本仓库是 ABACUS 的 worktree，分支 `feat/dftu-pw-port`，用于将 DFT+U PW 和 DeltaSpin 功能从 `abacus-zdy-tmp` 移植到 `abacus-develop`。

继承上游 `/root/abacus-develop/CLAUDE.md` 的所有构建、测试、代码风格规范。以下为本项目的额外约束。

---

## 强制验收规则（4-Gate）

**任何代码变更必须按顺序通过以下 4 个 Gate，缺一不可。不得以任何理由跳过。**

### Gate 1: 编译 — MUST PASS

```bash
cd /root/abacus-dftu-pw-port/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON .. 2>&1 | tee /tmp/cmake.log
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
```

- `build.log` 中零 `error:` 行
- 生成可执行文件 `abacus`
- 必须保留编译日志

### Gate 2: 单元测试 — MUST PASS

```bash
ctest --test-dir /root/abacus-dftu-pw-port/build/<相关测试目录> -j4 --output-on-failure 2>&1 | tee /tmp/unit_test.log
```

- 所有现有单元测试 PASS
- 新增/修改的 public 方法必须有对应 gtest 用例
- 新增的代码分支（如 nspin=1 vs nspin=4）必须有对应 test case
- 不允许以"依赖链重"为由跳过单元测试 — 提取 static helper 来测试
- 必须保留测试日志

### Gate 3: 集成测试 — MUST PASS

```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a <abacus路径> -n 4 -r "<case正则>" 2>&1 | tee /tmp/integration_test.log
```

- 所有相关现有集成测试 PASS
- 新增功能需要新建集成测试 case（INPUT, STRU, KPT, result.ref, README）
- 必须保留测试日志

### Gate 4: 代码审查 — MUST PASS

- 修改与 zdy-tmp 参考代码逻辑一致
- API 适配正确（Plus_U 而非 DFTU，PARAM.inp 而非 GlobalV，参数传递而非 GlobalC）
- 无遗留 debug print（std::cout/printf/std::cerr 调试输出）
- 无 WIP 代码（TODO/FIXME/HACK 中的临时代码）
- 无未使用的 include
- 函数签名与 develop 风格一致

---

## 执行者行为约束

### 禁止行为

1. 禁止声称"编译通过"但不提供编译日志
2. 禁止以"依赖链重"、"mock 成本高"为由跳过单元测试
3. 禁止以"需要后续 PR"为由跳过回归测试
4. 禁止在验收证据不完整时声称任务完成
5. 禁止修改验收标准以降低要求

### 必须行为

1. 编译后必须检查 build.log 中的 error 数量
2. 新增代码必须有对应的单元测试
3. 提交前必须运行相关回归测试
4. 必须填写完整的验收证据（见下方模板）
5. 遇到测试失败必须修复，不得标记为"已知问题"跳过

---

## 与用户的交互规则

### 何时必须暂停并询问用户

1. **API 设计选择**：当 zdy-tmp 和 develop 的实现方式有本质差异时（如 mixing 策略），必须向用户说明差异并请求决策
2. **测试失败无法修复**：同一错误重试 2 次仍失败时，输出根因分析并请求指导
3. **发现 task instruction 中的矛盾或遗漏**：不要自行假设，向用户确认
4. **需要修改非 task instruction 指定的文件**：说明原因并请求批准
5. **回归测试出现非预期偏差**：报告偏差数值和可能原因，请求判断

### 何时可以自主执行

1. task instruction 中明确指定的代码修改
2. 编译错误的修复（在 2 次重试限制内）
3. 单元测试的编写和调试
4. 集成测试 case 的创建（按 task instruction 规格）
5. 代码风格调整（clang-format 等）

### 任务完成时的报告格式

任务完成后必须输出以下验收证据，缺失任何一项即视为未完成：

```
## 验收证据

### Gate 1: 编译
- 编译结果: PASS / FAIL
- 编译日志: /tmp/build.log
- error 数量: 0
- 生成二进制: <路径和大小>

### Gate 2: 单元测试
- 新增测试文件: <文件路径列表>
- 新增测试用例: <TestSuite.TestCase 列表>
- 测试结果: X tests passed, 0 failed
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
```

---

## 项目特定信息

### 目录映射（zdy-tmp → develop）

| zdy-tmp | develop |
|---|---|
| `module_hamilt_lcao/module_dftu/` | `source_lcao/module_dftu/` |
| `module_hamilt_lcao/module_deltaspin/` | `source_lcao/module_deltaspin/` |
| `module_hamilt_pw/hamilt_pwdft/` | `source_pw/module_pwdft/` |

### API 映射

| zdy-tmp | develop |
|---|---|
| `ModuleDFTU::DFTU` | `Plus_U` |
| `GlobalV::NSPIN` | `PARAM.inp.nspin` |
| `GlobalV::KPAR` | `PARAM.inp.kpar` |
| `GlobalV::NPROC_IN_POOL` | `PARAM.globalv.nproc_in_pool` |
| `GlobalC::ucell` | `const UnitCell& cell`（参数传递） |
| `psi_p->npol` | `psi_p->get_npol()` |
| `FS_Nonlocal_tools` | `Onsite_Proj_tools` |

### 参考仓库

- worktree: `/root/abacus-dftu-pw-port`（本仓库）
- 上游 develop: `/root/abacus-develop`
- zdy-tmp 参考: `/root/abacus-zdy-tmp`
- PM agent: `/root/pm-agent/projects/f-electron-scf/`

### 任务指令位置

`/root/pm-agent/projects/f-electron-scf/tasks/PR-<N>-task-instruction.md`

### 构建命令

```bash
cd /root/abacus-dftu-pw-port/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
```
