import { db, sqlite } from "@/db";
import { differences, responses, confirmations, arRecords } from "@/db/schema";
import { eq, desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { z } from "zod";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  const results = await db
    .select({
      difference: differences,
      response: responses,
      confirmation: confirmations,
      arRecord: arRecords,
    })
    .from(differences)
    .leftJoin(responses, eq(differences.responseId, responses.id))
    .leftJoin(confirmations, eq(differences.confirmationId, confirmations.id))
    .leftJoin(arRecords, eq(confirmations.arRecordId, arRecords.id))
    .where(eq(differences.projectId, projectId))
    .orderBy(desc(differences.createdAt));

  return NextResponse.json(results);
}

const resolveSchema = z.object({
  id: z.string(),
  auditorResolution: z.string(),
});

export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const parsed = resolveSchema.parse(body);
    const now = new Date().toISOString();

    sqlite.transaction(() => {
      const diff = db.query.differences.findFirst({
        where: eq(differences.id, parsed.id),
      }).sync();

      if (!diff) throw new Error("差异记录不存在");

      db.update(differences)
        .set({
          auditorResolution: parsed.auditorResolution,
          status: "resolved",
          updatedAt: now,
        })
        .where(eq(differences.id, parsed.id))
        .run();

      db.update(confirmations)
        .set({ status: "reconciled", updatedAt: now })
        .where(eq(confirmations.id, diff.confirmationId))
        .run();
    })();

    return NextResponse.json({ success: true });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    if (error?.message === "差异记录不存在") {
      return NextResponse.json({ error: error.message }, { status: 404 });
    }
    return NextResponse.json({ error: "更新失败" }, { status: 500 });
  }
}
