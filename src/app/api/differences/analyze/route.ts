import { db } from "@/db";
import { differences, responses, confirmations, arRecords } from "@/db/schema";
import { eq } from "drizzle-orm";
import { NextResponse } from "next/server";
import { aiChat } from "@/lib/ai-client";
import { sanitizeForPrompt } from "@/lib/utils";
import { z } from "zod";

const analyzeSchema = z.object({
  differenceId: z.string(),
  projectId: z.string(),
});

export async function POST(request: Request) {
  let differenceId: string | null = null;
  try {
    const body = await request.json();
    const parsed = analyzeSchema.parse(body);
    differenceId = parsed.differenceId;

    const diff = await db.query.differences.findFirst({
      where: eq(differences.id, parsed.differenceId),
    });

    if (!diff) {
      return NextResponse.json({ error: "差异记录不存在" }, { status: 404 });
    }

    // Get related data
    const resp = await db.query.responses.findFirst({
      where: eq(responses.id, diff.responseId),
    });

    const conf = await db.query.confirmations.findFirst({
      where: eq(confirmations.id, diff.confirmationId),
    });

    const arRecord = conf
      ? await db.query.arRecords.findFirst({
          where: eq(arRecords.id, conf.arRecordId),
        })
      : null;

    // Update status to analyzing
    await db
      .update(differences)
      .set({ status: "analyzing", updatedAt: new Date().toISOString() })
      .where(eq(differences.id, parsed.differenceId));

    const systemPrompt = `你是一位经验丰富的审计师，专门负责分析应收账款函证差异。
请根据提供的差异信息，分析可能的差异原因并提供专业建议。
常见差异原因包括：
1. 在途款项（已付未达/已收未达）
2. 未达账项（发票已开未收到）
3. 退货或折让
4. 记账错误（金额错误、重复记账）
5. 期间差异（截止日不同）
6. 争议款项
7. 坏账核销差异

请返回JSON格式：{
  "suggestedCause": "差异原因分类",
  "analysisDetail": "详细分析说明",
  "confidenceScore": 0.0-1.0,
  "recommendations": ["建议1", "建议2"]
}`;

    const userPrompt = `差异分析请求：
客户名称：${sanitizeForPrompt(arRecord?.customerName, 100)}
账面金额：${diff.bookAmount.toLocaleString()}元
对方确认金额：${diff.confirmedAmount.toLocaleString()}元
差异金额：${diff.differenceAmount.toLocaleString()}元
回函类型：${resp?.responseType}
回函备注：${sanitizeForPrompt(resp?.notes, 500)}
客户账龄：1年以内${arRecord?.within1Year || 0}元，1-2年${arRecord?.year1to2 || 0}元，2-3年${arRecord?.year2to3 || 0}元
风险等级：${arRecord?.riskLevel || "未知"}
是否关联方：${arRecord?.isRelatedParty ? "是" : "否"}`;

    let aiResult;
    try {
      const { content } = await aiChat({
        projectId: parsed.projectId,
        action: "difference_analysis",
        systemPrompt,
        userPrompt,
        temperature: 0.2,
        jsonMode: true,
      });
      try {
        aiResult = JSON.parse(content);
      } catch {
        throw new Error("AI 返回的 JSON 格式无效");
      }
    } catch {
      // Fallback rule-based analysis
      const absDiff = Math.abs(diff.differenceAmount);
      const bookAbs = Math.abs(diff.bookAmount);
      let cause = "待分析";
      let detail = "";

      if (bookAbs > 0 && absDiff < bookAbs * 0.01) {
        cause = "尾差/四舍五入";
        detail = "差异金额较小，可能为计算尾差。";
      } else if (diff.differenceAmount > 0) {
        cause = "在途款项（已付未达）";
        detail = "对方确认金额大于账面金额，可能存在已付未达款项。";
      } else {
        cause = "在途款项（已收未达）";
        detail = "对方确认金额小于账面金额，可能存在已收未达款项或未达账项。";
      }

      aiResult = {
        suggestedCause: cause,
        analysisDetail: detail,
        confidenceScore: 0.5,
        recommendations: ["核查银行流水", "确认发票状态"],
      };
    }

    const now = new Date().toISOString();
    await db
      .update(differences)
      .set({
        aiSuggestedCause: aiResult.suggestedCause,
        aiAnalysisDetail: aiResult.analysisDetail + (aiResult.recommendations ? "\n\n建议：\n" + aiResult.recommendations.join("\n") : ""),
        aiConfidenceScore: aiResult.confidenceScore,
        status: "analyzed",
        updatedAt: now,
      })
      .where(eq(differences.id, parsed.differenceId));

    return NextResponse.json(aiResult);
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    // Reset status to pending so the record doesn't get stuck in "analyzing"
    if (differenceId) {
      try {
        await db
          .update(differences)
          .set({ status: "pending", updatedAt: new Date().toISOString() })
          .where(eq(differences.id, differenceId));
      } catch {
        // best-effort rollback
      }
    }
    return NextResponse.json({ error: error.message || "分析失败" }, { status: 500 });
  }
}
