# 2026-08-31 M7：最小输入解析 + 语义守卫

## 1. Test plan
- `DisabledByDefault`：`constraint=false` 时配置为 DISABLED，不解析不报错。
- `DefaultsAndDeltaParsing`：`{"targets": [0.1, -0.1]}` → 2 个 fragment，
  缺省 `atoms` 时 fragment i 默认取原子 i；weight_type 缺省 becke、
  target_mode 缺省 delta、mu_max 缺省 5.0、thr 缺省 1e-4。
- `FragmentParsing`：嵌套 `[[0,1,2]]` 与平铺 `[2,0]` 两种 atoms 语法。
- `AbsoluteModeWarnsButRuns`：absolute 模式仅 WARNING 不中止（口径差异
  ~0.2-0.3 e vs ~e，R12）。
- `Guards`：weight_type=hirshfeld → ERROR（一期未实现，不静默跑）；
  type!=charge → ERROR；无 target → ERROR（禁隐式约束，DeltaP 4.3 教训）；
  mode 非法 → ERROR；fragment 数不匹配 → ERROR；原子下标越界 → ERROR。
- `MalformedJson`：非法 JSON 结构 → ERROR。
- 参数注册：`read_input_item_test` 的 Item_test2 校验 5 个新 INPUT 参数
  默认值（constraint=False、constraint_weight_type=becke、
  constraint_target_mode=delta、constraint_mu_max=5.0、
  constraint_thr=1.0e-4）。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest，`MODULE_ESTATE_constraint_io`、
  `MODULE_IO_read_input_serial`（read_input_item_test）。
- 输入：合成 JSON 字符串 + INPUT 默认值断言（无真实 SCF）。

## 3. Results
- 6/6 PASS（constraint_io）；read_input_item_test 2/2 PASS。
- 参数注册：`input_parameter.h` 新增 7 个字段（constraint、
  constraint_type、constraint_weight_type、constraint_target_file、
  constraint_target_mode、constraint_mu_max、constraint_thr）；
  `read_input_item_other.cpp` 注册 7 个 Input_Item（category=Constraint，
  availability 与 deltaspin 同风格）。

## 4. Analysis
- 守卫全部返回 ConfigStatus::ERROR + error 字符串，由调用方 WARNING_QUIT，
  杜绝"无 target 静默跑约束"与"未知权重静默退化"两类错误语义。
- absolute 模式保留为合法但显式告警：一期物理可达域以 delta 口径校准，
  absolute 目标（0.2-0.3 e 量级）与 delta 位移（~e 量级）不可混用，
  告警提示用户核对。
- 解析器为最小 JSON 子集（key 查找 + 递归数组解析），刻意不引入第三方
  JSON 依赖；结构错误一律返回 false 并附位置信息。

## 5. Next steps
- Task 6 (M3a)：`constraint_inject_pw` PW veff 注入 + 共享权重实例断言。
- Task 7 (M5)：记账 E_con + 审计行。
- Task 8：外环编排 + esolver_ks_pw 钩子 + H2O 冒烟。
