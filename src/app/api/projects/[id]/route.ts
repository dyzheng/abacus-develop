import { db } from "@/db";
import { projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { NextResponse } from "next/server";
import { z } from "zod";

const updateProjectSchema = z.object({
  name: z.string().min(1).optional(),
  clientCompany: z.string().min(1).optional(),
  auditFirm: z.string().min(1).optional(),
  balanceDate: z.string().min(1).optional(),
  status: z.enum(["active", "completed", "archived"]).optional(),
});

export async function GET(_: Request, { params }: { params: { id: string } }) {
  const project = await db.query.projects.findFirst({
    where: eq(projects.id, params.id),
  });
  if (!project) {
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  }
  return NextResponse.json(project);
}

export async function PUT(request: Request, { params }: { params: { id: string } }) {
  try {
    const body = await request.json();
    const parsed = updateProjectSchema.parse(body);

    await db
      .update(projects)
      .set({ ...parsed, updatedAt: new Date().toISOString() })
      .where(eq(projects.id, params.id));

    const project = await db.query.projects.findFirst({
      where: eq(projects.id, params.id),
    });
    return NextResponse.json(project);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: "更新项目失败" }, { status: 500 });
  }
}

export async function DELETE(_: Request, { params }: { params: { id: string } }) {
  await db.delete(projects).where(eq(projects.id, params.id));
  return NextResponse.json({ success: true });
}
