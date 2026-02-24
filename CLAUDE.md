# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audit Agent (审计智能体) is an AI-powered accounts receivable (AR) confirmation audit system for Chinese auditing firms. Next.js 14 web application using OpenAI GPT-4o for intelligent analysis. The UI is entirely in Chinese.

## Commands

```bash
# Development
npm run dev              # Next.js dev server on port 3000
npm run lint             # ESLint

# Build
npm run build            # Next.js production build
npm run start            # Start production server

# Database
npm run db:generate      # Generate Drizzle migrations
npm run db:migrate       # Run migrations
npm run db:push          # Push schema directly to DB (dev shortcut)
npm run db:studio        # Open Drizzle Studio UI
```

## Environment

Requires `OPENAI_API_KEY` in `.env.local`. The SQLite database lives at `./data/audit.db` (or override with `AUDIT_DB_DIR` env var).

Additional config via env vars (see `src/lib/config.ts`): `AI_MODEL`, `RISK_HIGH_BALANCE`, `RISK_MEDIUM_BALANCE`, `DIFFERENCE_TOLERANCE`, `MAX_IMPORT_FILE_SIZE`, etc.

## Architecture

**Stack:** Next.js 14 (App Router) + React 18 + Tailwind CSS v4 + Drizzle ORM + SQLite (better-sqlite3)

**Path alias:** `@/*` maps to `./src/*`

### Key directories

- `src/app/api/` — 10 API route handlers (projects, ar-records, import, sample-selection, confirmations, responses, differences, differences/analyze, report)
- `src/app/projects/` — All page components, nested under `projects/[id]/` for each workflow step
- `src/components/ui/` — Reusable UI primitives (button, card, dialog, input, label, select, textarea, badge, loading) using class-variance-authority
- `src/db/schema.ts` — Drizzle ORM schema defining 7 tables: projects, arRecords, confirmations, responses, differences, aiLogs, importBatches
- `src/db/index.ts` — Database initialization (SQLite with WAL mode, foreign keys enabled). Exports both `db` (Drizzle) and `sqlite` (raw better-sqlite3 instance for transactions)
- `src/lib/ai-client.ts` — Centralized `aiChat()` function wrapping OpenAI with retry (exponential backoff, 3 attempts) and logging to aiLogs table
- `src/lib/config.ts` — Centralized configuration with env var overrides (AI model, risk thresholds, tolerance values, file size limits)
- `src/lib/api-error.ts` — Shared `handleApiError()` for consistent error responses (ZodError → 400, unknown → 500 with logging)

### Data flow

The app follows a linear audit workflow: Create Project → Import AR Data (Excel/CSV) → AI Sample Selection → Generate Confirmations → Register Responses → Analyze Differences → Generate Report.

Each step corresponds to a page under `src/app/projects/[id]/` and an API route under `src/app/api/`.

### AI integration

Three AI-powered features, all routed through `src/lib/ai-client.ts`:
1. **Sample selection** (`/api/sample-selection`) — Selects AR records for confirmation based on audit criteria; falls back to rule-based selection when AI is unavailable
2. **Difference analysis** (`/api/differences/analyze`) — Root cause analysis for discrepancies between book and confirmed amounts
3. **Report narrative** (`/api/report` POST) — Generates audit report text

All AI calls are logged to the `aiLogs` table with prompt, response, token usage, and status.

### Database

SQLite via better-sqlite3 + Drizzle ORM. Schema uses text IDs (UUIDs), text timestamps (ISO strings), and real numbers for monetary amounts. The confirmation workflow has 6 states: draft → generated → sent → received → reconciled → alternative_procedure.

### Validation

API routes validate requests with Zod schemas defined inline at the route level.

### UI patterns

All page components are client components (`"use client"`). State is managed with React hooks (no external state library). Toast notifications via Sonner. Charts via Recharts. The `cn()` utility in `src/lib/utils.ts` merges Tailwind classes via clsx + tailwind-merge.

