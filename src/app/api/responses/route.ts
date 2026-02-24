import { db, sqlite } from "@/db";
import { responses, confirmations, differences, arRecords } from "@/db/schema";
import { eq, desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { config } from "@/lib/config";
import { z } from "zod";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  const results = await db
    .select({
      response: responses,
      confirmation: confirmations,
      arRecord: arRecords,
    })
    .from(responses)
    .leftJoin(confirmations, eq(responses.confirmationId, confirmations.id))
    .leftJoin(arRecords, eq(confirmations.arRecordId, arRecords.id))
    .where(eq(responses.projectId, projectId))
    .orderBy(desc(responses.createdAt));

  return NextResponse.json(results);
}

const createResponseSchema = z.object({
  confirmationId: z.string(),
  projectId: z.string(),
  responseType: z.enum(["agree", "disagree", "partial", "no_response"]),
  respondedAmount: z.number().optional(),
  respondentName: z.string().optional(),
  respondentTitle: z.string().optional(),
  responseDate: z.string().optional(),
  notes: z.string().optional(),
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = createResponseSchema.parse(body);
    const now = new Date().toISOString();
    const id = uuid();

    // Get the confirmation and AR record to calculate difference
    const confirmation = await db.query.confirmations.findFirst({
      where: eq(confirmations.id, parsed.confirmationId),
    });

    if (!confirmation) {
      return NextResponse.json({ error: "函证不存在" }, { status: 404 });
    }

    const arRecord = await db.query.arRecords.findFirst({
      where: eq(arRecords.id, confirmation.arRecordId),
    });

    const bookAmount = arRecord?.totalBalance || 0;
    const respondedAmount = parsed.respondedAmount ?? bookAmount;
    const differenceAmount = respondedAmount - bookAmount;

    const result = sqlite.transaction(() => {
      db.insert(responses).values({
        id,
        confirmationId: parsed.confirmationId,
        projectId: parsed.projectId,
        responseType: parsed.responseType,
        respondedAmount,
        differenceAmount: Math.abs(differenceAmount) > config.differenceTolerance ? differenceAmount : 0,
        respondentName: parsed.respondentName || null,
        respondentTitle: parsed.respondentTitle || null,
        responseDate: parsed.responseDate || now,
        notes: parsed.notes || null,
        createdAt: now,
      }).run();

      db.update(confirmations)
        .set({
          status: "received",
          receivedDate: parsed.responseDate || now,
          updatedAt: now,
        })
        .where(eq(confirmations.id, parsed.confirmationId))
        .run();

      if ((parsed.responseType === "disagree" || parsed.responseType === "partial") && Math.abs(differenceAmount) > config.differenceTolerance) {
        db.insert(differences).values({
          id: uuid(),
          responseId: id,
          confirmationId: parsed.confirmationId,
          projectId: parsed.projectId,
          bookAmount,
          confirmedAmount: respondedAmount,
          differenceAmount,
          status: "pending",
          createdAt: now,
          updatedAt: now,
        }).run();
      }

      return { id, differenceAmount };
    })();

    return NextResponse.json(result, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: "登记回函失败" }, { status: 500 });
  }
}
