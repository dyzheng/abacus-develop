import { db, sqlite } from "@/db";
import { arRecords } from "@/db/schema";
import { eq, and, desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { aiChat } from "@/lib/ai-client";
import { safeJsonParse, sanitizeForPrompt } from "@/lib/utils";
import { z } from "zod";

const selectionSchema = z.object({
  projectId: z.string(),
  criteria: z.object({
    minBalance: z.number().optional(),
    includeHighRisk: z.boolean().optional(),
    includeRelatedParties: z.boolean().optional(),
    includeLongAged: z.boolean().optional(),
    samplePercentage: z.number().min(0).max(100).optional(),
    maxSamples: z.number().optional(),
  }).optional(),
  useAI: z.boolean().optional(),
});

function ruleBasedSelection(records: any[], criteria: any): { id: string; reason: string }[] {
  const selected: { id: string; reason: string }[] = [];
  const minBalance = criteria.minBalance || 100000;

  for (const r of records) {
    const reasons: string[] = [];

    if (r.totalBalance >= minBalance) {
      reasons.push(`余额${r.totalBalance.toLocaleString()}元，超过阈值${minBalance.toLocaleString()}元`);
    }

    if (criteria.includeHighRisk !== false && r.riskLevel === "high") {
      reasons.push("高风险客户");
    }

    if (criteria.includeRelatedParties !== false && r.isRelatedParty) {
      reasons.push("关联方交易");
    }

    const longAged = (r.year2to3 || 0) + (r.year3to4 || 0) + (r.year4to5 || 0) + (r.over5Years || 0);
    if (criteria.includeLongAged !== false && longAged > 0 && longAged > r.totalBalance * 0.2) {
      reasons.push(`长账龄占比${((longAged / r.totalBalance) * 100).toFixed(1)}%`);
    }

    if (reasons.length > 0) {
      selected.push({ id: r.id, reason: reasons.join("；") });
    }
  }

  const maxSamples = criteria.maxSamples || Math.max(10, Math.ceil(records.length * (criteria.samplePercentage || 30) / 100));
  return selected.slice(0, maxSamples);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = selectionSchema.parse(body);
    const { projectId, criteria = {}, useAI = false } = parsed;

    const records = await db
      .select()
      .from(arRecords)
      .where(eq(arRecords.projectId, projectId))
      .orderBy(desc(arRecords.totalBalance));

    if (records.length === 0) {
      return NextResponse.json({ error: "没有找到应收账款记录" }, { status: 400 });
    }

    let selections: { id: string; reason: string }[];

    if (useAI && process.env.OPENAI_API_KEY) {
      try {
        const summary = records.map((r) => ({
          id: r.id,
          customer: sanitizeForPrompt(r.customerName, 100),
          balance: r.totalBalance,
          within1Year: r.within1Year,
          year1to2: r.year1to2,
          year2to3: r.year2to3,
          year3to4: r.year3to4,
          year4to5: r.year4to5,
          over5Years: r.over5Years,
          riskLevel: r.riskLevel,
          isRelatedParty: r.isRelatedParty,
        }));

        const systemPrompt = `你是一位专业的审计师助手，专门负责应收账款函证的样本选择。
根据审计准则和风险评估，从提供的应收账款明细中选择需要函证的样本。
选择标准：
1. 大额余额 - 重要性水平以上的余额
2. 高风险客户 - 账龄较长、风险等级高的客户
3. 关联方 - 必须函证
4. 异常项目 - 余额波动大、账龄结构异常
5. 随机抽样 - 对未选中的项目进行适当比例的随机抽样
请返回JSON格式：{"selections": [{"id": "记录ID", "reason": "选择原因"}], "summary": "选择说明"}`;

        const userPrompt = `项目应收账款明细（共${records.length}条，总余额${records.reduce((s, r) => s + r.totalBalance, 0).toLocaleString()}元）：
${JSON.stringify(summary, null, 2)}

选择标准：
- 最低余额阈值：${criteria.minBalance || 100000}元
- 包含高风险：${criteria.includeHighRisk !== false ? "是" : "否"}
- 包含关联方：${criteria.includeRelatedParties !== false ? "是" : "否"}
- 包含长账龄：${criteria.includeLongAged !== false ? "是" : "否"}
- 建议样本量：${criteria.maxSamples || Math.ceil(records.length * 0.3)}条

请选择需要函证的样本。`;

        const { content } = await aiChat({
          projectId,
          action: "sample_selection",
          systemPrompt,
          userPrompt,
          temperature: 0.3,
          jsonMode: true,
        });

        const result = safeJsonParse(content, { selections: [] });
        selections = result.selections || [];

        // Validate IDs exist
        const validIds = new Set(records.map((r) => r.id));
        selections = selections.filter((s) => validIds.has(s.id));
      } catch (error) {
        // Fallback to rule-based
        selections = ruleBasedSelection(records, criteria);
      }
    } else {
      selections = ruleBasedSelection(records, criteria);
    }

    // Update records with selection status
    const selectionMap = new Map(selections.map((s) => [s.id, s.reason]));
    sqlite.transaction(() => {
      for (const record of records) {
        const reason = selectionMap.get(record.id);
        db.update(arRecords)
          .set({
            selectionStatus: reason !== undefined ? "ai_suggested" : "unselected",
            selectionReason: reason ?? null,
          })
          .where(eq(arRecords.id, record.id))
          .run();
      }
    })();

    const totalBalance = records.reduce((s, r) => s + r.totalBalance, 0);
    const selectedBalance = records
      .filter((r) => selections.some((s) => s.id === r.id))
      .reduce((s, r) => s + r.totalBalance, 0);

    return NextResponse.json({
      totalRecords: records.length,
      selectedCount: selections.length,
      totalBalance,
      selectedBalance,
      coverageRate: totalBalance > 0 ? ((selectedBalance / totalBalance) * 100).toFixed(1) : "0",
      selections,
    });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: error.message || "样本选择失败" }, { status: 500 });
  }
}
