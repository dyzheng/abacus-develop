# /gate — 运行指定的验收 Gate

运行单个验收 Gate。用法: `/gate <gate_number>`

参数 $ARGUMENTS 指定 Gate 编号（1-4）。

## Gate 定义

### 如果参数是 1（编译验证）

```bash
cd /root/abacus-dftu-pw-port/build
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
```

检查 build.log 中 `error:` 数量。报告：编译结果、error 数量、warning 数量、生成二进制路径。

### 如果参数是 2（单元测试）

1. 询问用户要测试哪个模块目录（或自动检测 git diff 涉及的模块）
2. 运行：
```bash
ctest --test-dir /root/abacus-dftu-pw-port/build/<测试目录> -j4 --output-on-failure 2>&1 | tee /tmp/unit_test.log
```
3. 报告：通过/失败数量、失败的 test case 名称和原因

### 如果参数是 3（集成测试）

1. 询问用户要运行哪些 case（或从 task instruction 中读取）
2. 运行：
```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a /root/abacus-dftu-pw-port/build/abacus -n 4 -r "<case正则>" 2>&1 | tee /tmp/integration_test.log
```
3. 报告：通过/失败的 case、偏差数值

### 如果参数是 4（代码审查）

1. 运行 `git diff --stat` 和 `git diff` 查看所有变更
2. 对每个变更文件执行检查清单：
   - 逻辑一致性（与 zdy-tmp 对比）
   - API 适配（搜索 GlobalV::, GlobalC::, DFTU:: 等旧 API）
   - debug print（搜索 std::cout, printf, std::cerr）
   - WIP 代码（搜索 TODO, FIXME, HACK）
   - 未使用 include
3. 报告：每项检查的结果和发现的问题
