import { db, sqlite } from "@/db";
import { confirmations, arRecords, projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { NextResponse } from "next/server";
import { z } from "zod";
import { handleApiError } from "@/lib/api-error";
import { aiChat } from "@/lib/ai-client";
import { sanitizeForPrompt, amountToChineseUppercase, formatAmount } from "@/lib/utils";

const generateSchema = z.object({
  projectId: z.string(),
  confirmationIds: z.array(z.string()).min(1),
});

const editSchema = z.object({
  confirmationId: z.string(),
  letterContent: z.string().min(1),
});

function buildFallbackLetter(project: any, ar: any, confirmation: any): string {
  const balanceDate = project.balanceDate || "____年__月__日";
  const totalStr = formatAmount(ar.totalBalance);
  const totalChinese = amountToChineseUppercase(ar.totalBalance);

  const addressBlock = [
    ar.customerName,
    ar.address && ar.city ? `${ar.province || ""}${ar.city}${ar.address}` : "",
    ar.contactPerson ? `联系人：${ar.contactPerson}` : "",
    ar.postalCode ? `邮编：${ar.postalCode}` : "",
  ].filter(Boolean).join("\n");

  const agingLines: string[] = [];
  if (ar.within1Year) agingLines.push(`  1年以内：${formatAmount(ar.within1Year)}`);
  if (ar.year1to2) agingLines.push(`  1-2年：${formatAmount(ar.year1to2)}`);
  if (ar.year2to3) agingLines.push(`  2-3年：${formatAmount(ar.year2to3)}`);
  if (ar.year3to4) agingLines.push(`  3-4年：${formatAmount(ar.year3to4)}`);
  if (ar.year4to5) agingLines.push(`  4-5年：${formatAmount(ar.year4to5)}`);
  if (ar.over5Years) agingLines.push(`  5年以上：${formatAmount(ar.over5Years)}`);
  const agingDetail = agingLines.length > 0 ? `\n账龄分解：\n${agingLines.join("\n")}` : "";

  return `${project.auditFirm}
应收账款函证

编号：${confirmation.confirmationNumber}

致：${addressBlock}

${ar.customerName}：

  本函仅为复核贵公司与${project.clientCompany}之间的往来账项而发出，不作为催款之用。

  根据${project.clientCompany}的账簿记录，截至${balanceDate}，贵公司尚欠${project.clientCompany}应收账款余额为：

  人民币（大写）：${totalChinese}
  人民币（小写）：${totalStr}
${agingDetail}

  如上述金额与贵公司记录相符，请在下方"信息证实无误"处签章确认；如有不符，请在"信息不符"处列明不符金额及原因。

  请将回函直接寄至：${project.auditFirm}

  □ 信息证实无误

  □ 信息不符，不符事项如下：
    _______________________________________________
    _______________________________________________

  贵公司盖章：________________
  经办人签名：________________
  日    期：________________`;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = generateSchema.parse(body);
    const now = new Date().toISOString();

    const project = await db.query.projects.findFirst({
      where: eq(projects.id, parsed.projectId),
    });
    if (!project) {
      return NextResponse.json({ error: "项目不存在" }, { status: 404 });
    }

    const results: { id: string; status: string }[] = [];

    for (const cid of parsed.confirmationIds) {
      const row = await db
        .select({ confirmation: confirmations, arRecord: arRecords })
        .from(confirmations)
        .leftJoin(arRecords, eq(confirmations.arRecordId, arRecords.id))
        .where(eq(confirmations.id, cid))
        .get();

      if (!row || !row.arRecord) {
        results.push({ id: cid, status: "skipped" });
        continue;
      }

      if (row.confirmation.status !== "draft") {
        results.push({ id: cid, status: "skipped_not_draft" });
        continue;
      }

      const ar = row.arRecord;
      const c = row.confirmation;
      let letterContent: string;

      try {
        const agingInfo = [
          ar.within1Year ? `1年以内: ${ar.within1Year}` : null,
          ar.year1to2 ? `1-2年: ${ar.year1to2}` : null,
          ar.year2to3 ? `2-3年: ${ar.year2to3}` : null,
          ar.year3to4 ? `3-4年: ${ar.year3to4}` : null,
          ar.year4to5 ? `4-5年: ${ar.year4to5}` : null,
          ar.over5Years ? `5年以上: ${ar.over5Years}` : null,
        ].filter(Boolean).join(", ");

        const { content } = await aiChat({
          projectId: parsed.projectId,
          action: "generate-letter",
          jsonMode: false,
          systemPrompt: `你是一位中国注册会计师，请根据提供的信息生成一份正式的应收账款函证信函。

要求：
1. 包含审计事务所抬头
2. 包含收件人地址块（客户名称、地址、联系人）
3. 正文说明函证目的、基准日、余额明细（含账龄分解）
4. 金额同时显示阿拉伯数字和中文大写
5. 包含回函说明和要求（相符/不符选项）
6. 包含签章区域
7. 语言正式、专业，符合中国审计函证规范`,
          userPrompt: `请生成函证信函，信息如下：
审计事务所：${sanitizeForPrompt(project.auditFirm, 100)}
被审计单位：${sanitizeForPrompt(project.clientCompany, 100)}
基准日：${project.balanceDate}
函证编号：${c.confirmationNumber}
客户名称：${sanitizeForPrompt(ar.customerName, 100)}
客户地址：${sanitizeForPrompt(ar.address, 200)} ${sanitizeForPrompt(ar.city, 50)} ${sanitizeForPrompt(ar.province, 50)}
联系人：${sanitizeForPrompt(ar.contactPerson, 50)}
邮编：${sanitizeForPrompt(ar.postalCode, 20)}
应收账款余额：${ar.totalBalance}元
中文大写：${amountToChineseUppercase(ar.totalBalance)}
账龄分解：${agingInfo || "无明细"}`,
        });
        letterContent = content;
      } catch {
        letterContent = buildFallbackLetter(project, ar, c);
      }

      sqlite.transaction(() => {
        db.update(confirmations)
          .set({
            letterContent,
            letterGeneratedAt: now,
            status: "generated",
            updatedAt: now,
          })
          .where(eq(confirmations.id, cid))
          .run();
      })();

      results.push({ id: cid, status: "generated" });
    }

    return NextResponse.json({ results, generated: results.filter((r) => r.status === "generated").length });
  } catch (error) {
    return handleApiError(error, "生成函证信函失败");
  }
}

export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const parsed = editSchema.parse(body);
    const now = new Date().toISOString();

    db.update(confirmations)
      .set({ letterContent: parsed.letterContent, updatedAt: now })
      .where(eq(confirmations.id, parsed.confirmationId))
      .run();

    return NextResponse.json({ success: true });
  } catch (error) {
    return handleApiError(error, "更新信函失败");
  }
}
