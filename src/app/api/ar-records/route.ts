import { db, sqlite } from "@/db";
import { arRecords } from "@/db/schema";
import { eq, desc, and, gte, lte, like } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { handleApiError } from "@/lib/api-error";

const VALID_RISK_LEVELS = ["low", "medium", "high"] as const;
const VALID_SELECTION_STATUSES = ["unselected", "ai_suggested", "confirmed", "excluded"] as const;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  const conditions = [eq(arRecords.projectId, projectId)];

  const riskLevel = searchParams.get("riskLevel");
  if (riskLevel) {
    if (!VALID_RISK_LEVELS.includes(riskLevel as any)) {
      return NextResponse.json({ error: `无效的 riskLevel: ${riskLevel}` }, { status: 400 });
    }
    conditions.push(eq(arRecords.riskLevel, riskLevel as "low" | "medium" | "high"));
  }

  const selectionStatus = searchParams.get("selectionStatus");
  if (selectionStatus) {
    if (!VALID_SELECTION_STATUSES.includes(selectionStatus as any)) {
      return NextResponse.json({ error: `无效的 selectionStatus: ${selectionStatus}` }, { status: 400 });
    }
    conditions.push(eq(arRecords.selectionStatus, selectionStatus as "unselected" | "ai_suggested" | "confirmed" | "excluded"));
  }

  const search = searchParams.get("search");
  if (search) {
    conditions.push(like(arRecords.customerName, `%${search}%`));
  }

  const minBalance = searchParams.get("minBalance");
  if (minBalance) {
    const val = Number(minBalance);
    if (!isNaN(val)) conditions.push(gte(arRecords.totalBalance, val));
  }

  const maxBalance = searchParams.get("maxBalance");
  if (maxBalance) {
    const val = Number(maxBalance);
    if (!isNaN(val)) conditions.push(lte(arRecords.totalBalance, val));
  }

  const records = await db
    .select()
    .from(arRecords)
    .where(and(...conditions))
    .orderBy(desc(arRecords.totalBalance));

  return NextResponse.json(records);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const records = Array.isArray(body) ? body : [body];
    const now = new Date().toISOString();

    const inserted = [];
    for (const record of records) {
      const id = uuid();
      await db.insert(arRecords).values({
        id,
        projectId: record.projectId,
        importBatchId: record.importBatchId || null,
        customerName: record.customerName,
        customerCode: record.customerCode || null,
        totalBalance: record.totalBalance,
        within1Year: record.within1Year || 0,
        year1to2: record.year1to2 || 0,
        year2to3: record.year2to3 || 0,
        year3to4: record.year3to4 || 0,
        year4to5: record.year4to5 || 0,
        over5Years: record.over5Years || 0,
        riskLevel: record.riskLevel || "low",
        isRelatedParty: record.isRelatedParty || false,
        selectionStatus: "unselected",
        createdAt: now,
      });
      inserted.push(id);
    }

    return NextResponse.json({ inserted: inserted.length, ids: inserted }, { status: 201 });
  } catch (error) {
    return handleApiError(error, "创建记录失败");
  }
}
