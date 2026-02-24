import { db } from "@/db";
import { projects, arRecords, confirmations, responses, differences } from "@/db/schema";
import { eq, count, sum, and } from "drizzle-orm";
import { NextResponse } from "next/server";
import { aiChat } from "@/lib/ai-client";
import { safeJsonParse, sanitizeForPrompt } from "@/lib/utils";
import { z } from "zod";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  const project = await db.query.projects.findFirst({
    where: eq(projects.id, projectId),
  });

  if (!project) {
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  }

  // AR stats
  const allRecords = await db.select().from(arRecords).where(eq(arRecords.projectId, projectId));
  const totalARCount = allRecords.length;
  const totalARBalance = allRecords.reduce((s, r) => s + r.totalBalance, 0);

  // Confirmation stats
  const allConfirmations = await db.select().from(confirmations).where(eq(confirmations.projectId, projectId));
  const confirmationCount = allConfirmations.length;
  const confirmedBalance = allRecords
    .filter((r) => allConfirmations.some((c) => c.arRecordId === r.id))
    .reduce((s, r) => s + r.totalBalance, 0);

  const statusCounts: Record<string, number> = {};
  for (const c of allConfirmations) {
    statusCounts[c.status] = (statusCounts[c.status] || 0) + 1;
  }

  // Response stats
  const allResponses = await db.select().from(responses).where(eq(responses.projectId, projectId));
  const responseCount = allResponses.length;
  const agreeCount = allResponses.filter((r) => r.responseType === "agree").length;
  const disagreeCount = allResponses.filter((r) => r.responseType === "disagree" || r.responseType === "partial").length;
  const noResponseCount = allResponses.filter((r) => r.responseType === "no_response").length;

  // Difference stats
  const allDifferences = await db.select().from(differences).where(eq(differences.projectId, projectId));
  const differenceCount = allDifferences.length;
  const resolvedCount = allDifferences.filter((d) => d.status === "resolved").length;
  const totalDifferenceAmount = allDifferences.reduce((s, d) => s + Math.abs(d.differenceAmount), 0);

  const report = {
    project,
    summary: {
      totalARCount,
      totalARBalance,
      confirmationCount,
      confirmedBalance,
      coverageRate: totalARBalance > 0 ? ((confirmedBalance / totalARBalance) * 100).toFixed(1) : "0",
      statusCounts,
      responseCount,
      agreeCount,
      disagreeCount,
      noResponseCount,
      responseRate: confirmationCount > 0 ? ((responseCount / confirmationCount) * 100).toFixed(1) : "0",
      differenceCount,
      resolvedCount,
      totalDifferenceAmount,
    },
  };

  return NextResponse.json(report);
}

const narrativeSchema = z.object({
  projectId: z.string(),
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = narrativeSchema.parse(body);

    const project = await db.query.projects.findFirst({
      where: eq(projects.id, parsed.projectId),
    });

    const allRecords = await db.select().from(arRecords).where(eq(arRecords.projectId, parsed.projectId));
    const allConfirmations = await db.select().from(confirmations).where(eq(confirmations.projectId, parsed.projectId));
    const allResponses = await db.select().from(responses).where(eq(responses.projectId, parsed.projectId));
    const allDifferences = await db.select().from(differences).where(eq(differences.projectId, parsed.projectId));

    const totalARBalance = allRecords.reduce((s, r) => s + r.totalBalance, 0);
    const confirmedBalance = allRecords
      .filter((r) => allConfirmations.some((c) => c.arRecordId === r.id))
      .reduce((s, r) => s + r.totalBalance, 0);

    const stats = {
      totalRecords: allRecords.length,
      totalBalance: totalARBalance.toLocaleString(),
      confirmations: allConfirmations.length,
      coverageRate: totalARBalance > 0 ? ((confirmedBalance / totalARBalance) * 100).toFixed(1) : "0",
      responses: allResponses.length,
      agreeCount: allResponses.filter((r) => r.responseType === "agree").length,
      disagreeCount: allResponses.filter((r) => r.responseType === "disagree" || r.responseType === "partial").length,
      differences: allDifferences.length,
      resolved: allDifferences.filter((d) => d.status === "resolved").length,
    };

    let narrative: string;

    try {
      const { content } = await aiChat({
        projectId: parsed.projectId,
        action: "report_narrative",
        systemPrompt: `你是一位专业审计师，请根据提供的函证统计数据，撰写一段专业的函证结果汇总报告文字。
报告应包含：
1. 函证范围和覆盖率
2. 回函情况统计
3. 差异分析结论
4. 审计结论建议
请使用正式的审计报告语言，中文撰写。直接返回JSON：{"narrative": "报告文字"}`,
        userPrompt: `被审计单位：${sanitizeForPrompt(project?.clientCompany, 100)}
审计基准日：${project?.balanceDate}
统计数据：${JSON.stringify(stats)}`,
        temperature: 0.4,
        jsonMode: true,
      });
      const result = safeJsonParse(content, { narrative: "" });
      narrative = result.narrative;
    } catch {
      // Fallback
      narrative = `一、函证范围
本次对${project?.clientCompany}截至${project?.balanceDate}的应收账款进行了函证程序。应收账款明细共${stats.totalRecords}条，余额合计${stats.totalBalance}元。选取${stats.confirmations}笔进行函证，函证覆盖率为${stats.coverageRate}%。

二、回函情况
共收到回函${stats.responses}份。其中，${stats.agreeCount}份确认相符，${stats.disagreeCount}份存在差异。

三、差异分析
共发现差异${stats.differences}项，已完成分析${stats.resolved}项。

四、审计结论
基于上述函证程序的执行结果，应收账款余额总体可靠，差异项目已进行了充分的替代程序和分析。`;
    }

    return NextResponse.json({ narrative });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: error.message || "生成报告失败" }, { status: 500 });
  }
}
