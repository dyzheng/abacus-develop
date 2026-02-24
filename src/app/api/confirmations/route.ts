import { db, sqlite } from "@/db";
import { confirmations, arRecords } from "@/db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { z } from "zod";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  const whereClause = eq(confirmations.projectId, projectId);
  const page = Number(searchParams.get("page")) || 0;
  const pageSize = Number(searchParams.get("pageSize")) || 0;

  if (page > 0 && pageSize > 0) {
    const [{ total }] = await db
      .select({ total: sql<number>`count(*)` })
      .from(confirmations)
      .where(whereClause);

    const results = await db
      .select({
        confirmation: confirmations,
        arRecord: arRecords,
      })
      .from(confirmations)
      .leftJoin(arRecords, eq(confirmations.arRecordId, arRecords.id))
      .where(whereClause)
      .orderBy(desc(confirmations.createdAt))
      .limit(pageSize)
      .offset((page - 1) * pageSize);

    return NextResponse.json({ data: results, total, page, pageSize });
  }

  const results = await db
    .select({
      confirmation: confirmations,
      arRecord: arRecords,
    })
    .from(confirmations)
    .leftJoin(arRecords, eq(confirmations.arRecordId, arRecords.id))
    .where(whereClause)
    .orderBy(desc(confirmations.createdAt));

  return NextResponse.json(results);
}

const createSchema = z.object({
  projectId: z.string(),
  arRecordIds: z.array(z.string()),
  type: z.enum(["positive", "blank"]).optional(),
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = createSchema.parse(body);
    const now = new Date().toISOString();
    const year = new Date().getFullYear();

    const created = sqlite.transaction(() => {
      // Get max counter inside transaction to avoid race condition
      const maxRow = db
        .select({ maxNum: sql<string>`max(${confirmations.confirmationNumber})` })
        .from(confirmations)
        .where(eq(confirmations.projectId, parsed.projectId))
        .get();
      let counter = 0;
      if (maxRow?.maxNum) {
        const parts = maxRow.maxNum.split("-");
        counter = parseInt(parts[parts.length - 1], 10) || 0;
      }

      const result: { id: string; confirmationNumber: string }[] = [];
      for (const arRecordId of parsed.arRecordIds) {
        counter++;
        const id = uuid();
        const confirmationNumber = `HC-${year}-${String(counter).padStart(4, "0")}`;

        db.insert(confirmations).values({
          id,
          projectId: parsed.projectId,
          arRecordId,
          confirmationNumber,
          type: parsed.type || "positive",
          status: "draft",
          createdAt: now,
          updatedAt: now,
        }).run();

        db.update(arRecords)
          .set({ selectionStatus: "confirmed" })
          .where(eq(arRecords.id, arRecordId))
          .run();

        result.push({ id, confirmationNumber });
      }
      return result;
    })();

    return NextResponse.json({ created, count: created.length }, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: "创建函证失败" }, { status: 500 });
  }
}

const VALID_TRANSITIONS: Record<string, string[]> = {
  draft: ["generated"],
  generated: ["sent"],
  sent: ["received", "alternative_procedure"],
  received: ["reconciled", "alternative_procedure"],
};

const updateSchema = z.object({
  ids: z.array(z.string()).optional(),
  id: z.string().optional(),
  status: z.enum(["draft", "generated", "sent", "received", "reconciled", "alternative_procedure"]),
  sentDate: z.string().optional(),
  dueDate: z.string().optional(),
  receivedDate: z.string().optional(),
});

export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const parsed = updateSchema.parse(body);
    const now = new Date().toISOString();

    const ids = parsed.ids || (parsed.id ? [parsed.id] : []);

    const updated = sqlite.transaction(() => {
      let count = 0;
      for (const id of ids) {
        const current = db.query.confirmations.findFirst({
          where: eq(confirmations.id, id),
        }).sync();
        if (!current) continue;

        const allowed = VALID_TRANSITIONS[current.status];
        if (allowed && !allowed.includes(parsed.status)) {
          throw new Error(`不允许从 "${current.status}" 转换到 "${parsed.status}"`);
        }

        const updateData: Record<string, string> = { status: parsed.status, updatedAt: now };
        if (parsed.sentDate) updateData.sentDate = parsed.sentDate;
        if (parsed.dueDate) updateData.dueDate = parsed.dueDate;
        if (parsed.receivedDate) updateData.receivedDate = parsed.receivedDate;

        db.update(confirmations).set(updateData).where(eq(confirmations.id, id)).run();
        count++;
      }
      return count;
    })();

    return NextResponse.json({ updated });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    if (error?.message?.startsWith("不允许从")) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    return NextResponse.json({ error: "更新函证失败" }, { status: 500 });
  }
}
