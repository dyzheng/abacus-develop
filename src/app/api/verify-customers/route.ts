import { db, sqlite } from "@/db";
import { arRecords } from "@/db/schema";
import { eq, and, inArray } from "drizzle-orm";
import { NextResponse } from "next/server";
import { z } from "zod";
import { handleApiError } from "@/lib/api-error";
import { aiChat } from "@/lib/ai-client";
import { sanitizeForPrompt, safeJsonParse } from "@/lib/utils";

const requestSchema = z.object({
  projectId: z.string(),
  arRecordIds: z.array(z.string()).optional(),
});

interface VerificationResult {
  id: string;
  status: "verified" | "suspicious" | "flagged";
  score: number;
  detail: string;
  inferredBusinessType?: string;
  suggestedRegion?: string;
}

function fallbackVerify(name: string): { status: "unverified" | "suspicious" | "flagged"; score: number; detail: string } {
  const trimmed = name.trim();
  if (trimmed.length < 4) {
    return { status: "flagged", score: 0.2, detail: "名称过短（少于4个字符），可能为虚构客户" };
  }
  const hasSuffix = /(?:有限公司|集团|股份|合伙企业|个体|工作室|事务所)/.test(trimmed);
  if (!hasSuffix) {
    return { status: "suspicious", score: 0.4, detail: "名称缺少常见企业后缀（如有限公司、集团等），建议人工核实" };
  }
  return { status: "unverified", score: 0.5, detail: "规则校验通过，但未经AI深度核验" };
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = requestSchema.parse(body);

    const conditions = [eq(arRecords.projectId, parsed.projectId)];
    if (parsed.arRecordIds && parsed.arRecordIds.length > 0) {
      conditions.push(inArray(arRecords.id, parsed.arRecordIds));
    } else {
      conditions.push(eq(arRecords.verificationStatus, "unverified"));
    }

    const records = await db
      .select({ id: arRecords.id, customerName: arRecords.customerName })
      .from(arRecords)
      .where(and(...conditions));

    if (records.length === 0) {
      return NextResponse.json({ results: [], message: "没有需要核验的记录" });
    }

    const allResults: VerificationResult[] = [];
    const BATCH_SIZE = 20;

    for (let i = 0; i < records.length; i += BATCH_SIZE) {
      const batch = records.slice(i, i + BATCH_SIZE);
      const customerList = batch.map((r, idx) => `${idx + 1}. [${r.id}] ${sanitizeForPrompt(r.customerName, 100)}`).join("\n");

      let batchResults: VerificationResult[];

      try {
        const { content } = await aiChat({
          projectId: parsed.projectId,
          action: "verify-customers",
          systemPrompt: `你是一位中国注册会计师，擅长审计中的客户真实性核验。请分析以下客户名称列表，判断每个客户是否可能是真实存在的企业。

对每个客户，请评估：
1. 名称是否符合中国工商注册命名规范
2. 是否有虚构迹象（如名称过于随意、不符合行业惯例）
3. 推断可能的行业类型
4. 推测可能的注册地区

返回JSON格式：
{
  "results": [
    {
      "id": "记录ID",
      "status": "verified|suspicious|flagged",
      "score": 0.0-1.0的置信度,
      "detail": "分析说明",
      "inferredBusinessType": "推断行业",
      "suggestedRegion": "推测地区"
    }
  ]
}

status说明：
- verified: 名称规范，大概率为真实企业 (score >= 0.7)
- suspicious: 存在疑点，建议进一步核实 (score 0.4-0.7)
- flagged: 高度可疑，可能为虚构 (score < 0.4)`,
          userPrompt: `请核验以下${batch.length}个客户名称：\n${customerList}`,
        });

        const parsed_ai = safeJsonParse<{ results: VerificationResult[] }>(content, { results: [] });
        batchResults = parsed_ai.results;

        // Validate and fill missing
        const resultMap = new Map(batchResults.map((r) => [r.id, r]));
        batchResults = batch.map((rec) => {
          const aiResult = resultMap.get(rec.id);
          if (aiResult && aiResult.status && typeof aiResult.score === "number") {
            return {
              id: rec.id,
              status: (["verified", "suspicious", "flagged"].includes(aiResult.status) ? aiResult.status : "suspicious") as "verified" | "suspicious" | "flagged",
              score: Math.max(0, Math.min(1, aiResult.score)),
              detail: aiResult.detail || "",
              inferredBusinessType: aiResult.inferredBusinessType,
              suggestedRegion: aiResult.suggestedRegion,
            };
          }
          const fb = fallbackVerify(rec.customerName);
          return { id: rec.id, ...fb } as VerificationResult;
        });
      } catch {
        // AI unavailable — use fallback
        batchResults = batch.map((rec) => {
          const fb = fallbackVerify(rec.customerName);
          return { id: rec.id, ...fb } as VerificationResult;
        });
      }

      // Write results to DB
      sqlite.transaction(() => {
        for (const result of batchResults) {
          db.update(arRecords)
            .set({
              verificationStatus: result.status,
              verificationDetail: JSON.stringify({
                detail: result.detail,
                inferredBusinessType: (result as any).inferredBusinessType || null,
                suggestedRegion: (result as any).suggestedRegion || null,
              }),
              verificationScore: result.score,
            })
            .where(eq(arRecords.id, result.id))
            .run();
        }
      })();

      allResults.push(...batchResults);
    }

    return NextResponse.json({ results: allResults, total: allResults.length });
  } catch (error) {
    return handleApiError(error, "客户核验失败");
  }
}
