# 后端功能迭代计划

基于代码审查发现的问题，按优先级排列。

---

## P0 — 数据完整性 ✅ 已完成

### 1. ✅ 多步写操作缺少事务

所有多步写操作已用 `sqlite.transaction()` 包裹：responses、confirmations POST/PUT、import、differences PUT。导出了底层 `sqlite` 实例供事务使用。

### 2. ✅ 函证编号竞态条件

改为在事务内用 `SELECT MAX(confirmation_number)` 解析最大编号，避免并发重复。schema 中已加 `uniqueIndex` 约束。

### 3. ✅ 函证状态转换校验

confirmations PUT 增加了 `VALID_TRANSITIONS` 映射，校验 `currentStatus → newStatus` 合法性，非法转换返回 400。

### 4. ✅ AI 分析失败恢复

differences/analyze 外层 catch 中将状态回退为 "pending"。内层 AI JSON 解析也加了 try-catch，fallback 修复了除零风险。

---

## P1 — 数据精度与校验 ✅ 已完成

### 5. 金额浮点精度（部分完成）

差异阈值已提取到 `src/lib/config.ts` 的 `differenceTolerance`，可通过环境变量覆盖。金额存储类型（real → integer）改动较大，留待后续迁移。

### 6. ✅ AI 返回值 JSON 解析防护

新增 `safeJsonParse()` 工具函数，已应用到 sample-selection、report 路由。differences/analyze 内层也加了 try-catch。

### 7. ✅ 导入接口校验加固

- 文件大小限制（可配置，默认 10MB）
- 空工作簿/空 sheet 校验
- `columnMapping` JSON.parse 改用 `safeJsonParse`
- 返回 `skippedCount` 和 `totalRows`

### 8. ✅ ar-records 查询参数校验

- `selectionStatus` 和 `riskLevel` 改为枚举校验，非法值返回 400
- `minBalance`/`maxBalance` 已实现为 `gte`/`lte` 过滤
- 移除了未使用的 `querySchema` 和 `as any`

---

## P2 — 性能（部分完成）

### 9. N+1 查询 / 逐条插入（部分缓解）

事务包裹后 SQLite 同步模式下性能已大幅改善（单次 fsync）。sample-selection 的 O(n²) `.find()` 已改为 `Map` 查找。批量 INSERT 可进一步优化但优先级降低。

### 10. ✅ 数据库索引

schema.ts 已为所有外键列和常用查询列添加索引，confirmations 加了 `(projectId, confirmationNumber)` 唯一索引。

### 11. 无分页（待做）

需要前后端联动改造，留待后续迭代。

---

## P3 — 安全与健壮性 ✅ 已完成

### 12. ✅ Prompt injection 防护

新增 `sanitizeForPrompt()` 工具函数（截断 + 去换行 + 去特殊字符），已应用到 differences/analyze、sample-selection、report 三个 AI 调用路由的用户数据插值。

### 13. ✅ 硬编码配置提取

新增 `src/lib/config.ts`，以下值可通过环境变量覆盖：
- `AI_MODEL` — AI 模型名（默认 gpt-4o）
- `RISK_HIGH_BALANCE` / `RISK_MEDIUM_BALANCE` — 风险分类余额阈值
- `RISK_HIGH_AGING_RATIO` / `RISK_MEDIUM_AGING_RATIO` — 账龄占比阈值
- `DIFFERENCE_TOLERANCE` — 差异容忍阈值
- `CONFIRMATION_DUE_DAYS` — 函证到期天数
- `MAX_IMPORT_FILE_SIZE` — 导入文件大小限制

### 14. ✅ 统一错误处理

新增 `src/lib/api-error.ts` 的 `handleApiError()`，统一 ZodError → 400、未知错误 → 500 + console.error 日志。已应用到 import 和 ar-records 路由。

---

## P4 — 前端（后续迭代）

### 15. useEffect 依赖数组（待做）

补全依赖并加入 AbortController 取消请求。

### 16. ✅ report/route.ts 的自调用

已移除 `fetch(request.url.split(...))` 自调用，改为直接查询数据库。
