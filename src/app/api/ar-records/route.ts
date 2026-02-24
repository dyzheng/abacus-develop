import { db, sqlite } from "@/db";
import { arRecords } from "@/db/schema";
import { eq, desc, and, gte, lte, like, sql, inArray } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { z } from "zod";
import { handleApiError } from "@/lib/api-error";

const VALID_RISK_LEVELS = ["low", "medium", "high"] as const;
const VALID_SELECTION_STATUSES = ["unselected", "ai_suggested", "confirmed", "excluded"] as const;
const VALID_VERIFICATION_STATUSES = ["unverified", "verified", "suspicious", "flagged"] as const;

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

  const verificationStatus = searchParams.get("verificationStatus");
  if (verificationStatus) {
    if (!VALID_VERIFICATION_STATUSES.includes(verificationStatus as any)) {
      return NextResponse.json({ error: `无效的 verificationStatus: ${verificationStatus}` }, { status: 400 });
    }
    conditions.push(eq(arRecords.verificationStatus, verificationStatus as "unverified" | "verified" | "suspicious" | "flagged"));
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

  const whereClause = and(...conditions);

  const page = Number(searchParams.get("page")) || 0;
  const pageSize = Number(searchParams.get("pageSize")) || 0;

  if (page > 0 && pageSize > 0) {
    const [{ total }] = await db
      .select({ total: sql<number>`count(*)` })
      .from(arRecords)
      .where(whereClause);

    const records = await db
      .select()
      .from(arRecords)
      .where(whereClause)
      .orderBy(desc(arRecords.totalBalance))
      .limit(pageSize)
      .offset((page - 1) * pageSize);

    return NextResponse.json({ data: records, total, page, pageSize });
  }

  const records = await db
    .select()
    .from(arRecords)
    .where(whereClause)
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

const updateContactSchema = z.object({
  ids: z.array(z.string()).min(1),
  contactPerson: z.string().nullish(),
  contactPhone: z.string().nullish(),
  contactEmail: z.string().email().nullish().or(z.literal("")),
  address: z.string().nullish(),
  city: z.string().nullish(),
  province: z.string().nullish(),
  postalCode: z.string().nullish(),
});

export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const parsed = updateContactSchema.parse(body);
    const now = new Date().toISOString();

    const { ids, ...fields } = parsed;
    const updateData: Record<string, any> = {};
    for (const [key, value] of Object.entries(fields)) {
      if (value !== undefined) {
        updateData[key] = value || null;
      }
    }

    if (Object.keys(updateData).length === 0) {
      return NextResponse.json({ error: "没有需要更新的字段" }, { status: 400 });
    }

    const updated = sqlite.transaction(() => {
      let count = 0;
      for (const id of ids) {
        db.update(arRecords).set(updateData).where(eq(arRecords.id, id)).run();
        count++;
      }
      return count;
    })();

    return NextResponse.json({ updated });
  } catch (error) {
    return handleApiError(error, "更新记录失败");
  }
}
