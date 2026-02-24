# 审计智能体 — 应收账款函证系统

AI 驱动的应收账款函证审计管理系统，面向中国审计事务所。

## 技术栈

- **前端：** Next.js 14 (App Router) + React 18 + Tailwind CSS v4 + Recharts
- **后端：** Next.js API Routes + Drizzle ORM + SQLite (better-sqlite3)
- **AI：** OpenAI GPT-4o（可配置）

## 快速开始

```bash
# 安装依赖
npm install

# 配置环境变量
cp .env.local.example .env.local
# 编辑 .env.local，填入 OPENAI_API_KEY

# 初始化数据库
npm run db:push

# 启动开发服务器
npm run dev
```

访问 http://localhost:3000

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENAI_API_KEY` | 是 | — | OpenAI API 密钥 |
| `AUDIT_DB_DIR` | 否 | `./data` | SQLite 数据库目录 |
| `AI_MODEL` | 否 | `gpt-4o` | AI 模型名称 |
| `RISK_HIGH_BALANCE` | 否 | `1000000` | 高风险余额阈值（元） |
| `RISK_MEDIUM_BALANCE` | 否 | `100000` | 中风险余额阈值（元） |
| `DIFFERENCE_TOLERANCE` | 否 | `0.01` | 差异容忍阈值（元） |
| `MAX_IMPORT_FILE_SIZE` | 否 | `10485760` | 导入文件大小限制（字节） |

## 审计工作流

```
创建项目 → 导入AR数据 → AI抽样选择 → 生成函证 → 登记回函 → 差异分析 → 生成报告
```

每个步骤对应一个页面（`/projects/[id]/*`）和一组 API 接口（`/api/*`）。

### 功能说明

**数据导入** — 支持 Excel/CSV 文件上传，自动识别列映射（模糊匹配中英文表头），自动风险分类。

**AI 抽样选择** — 基于审计准则和风险评估智能选择函证样本。AI 不可用时自动降级为规则引擎。

**函证管理** — 6 阶段状态流转：草稿 → 已生成 → 已发出 → 已收回 → 已核对 → 替代程序。状态转换有严格校验。

**差异分析** — AI 自动分析差异原因（在途款项、记账错误、期间差异等），提供置信度评分和处理建议。

**报告生成** — 自动汇总统计数据，AI 生成专业审计报告文字。

## 常用命令

```bash
npm run dev          # 开发服务器
npm run build        # 生产构建
npm run start        # 生产运行
npm run lint         # 代码检查
npm run db:push      # 推送 schema 到数据库
npm run db:studio    # 打开 Drizzle Studio
npm run db:generate  # 生成迁移文件
npm run db:migrate   # 执行迁移
```

## 项目结构

```
src/
├── app/
│   ├── api/              # 10 个 API 路由
│   └── projects/         # 页面组件（按工作流步骤组织）
├── components/ui/        # 通用 UI 组件
├── db/
│   ├── schema.ts         # 数据库 schema（7 张表）
│   └── index.ts          # 数据库初始化
└── lib/
    ├── ai-client.ts      # OpenAI 封装（重试 + 日志）
    ├── config.ts         # 集中配置管理
    ├── api-error.ts      # 统一错误处理
    └── utils.ts          # 工具函数
```
