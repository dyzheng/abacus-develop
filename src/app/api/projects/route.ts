import { db } from "@/db";
import { projects } from "@/db/schema";
import { desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { z } from "zod";

const createProjectSchema = z.object({
  name: z.string().min(1),
  clientCompany: z.string().min(1),
  auditFirm: z.string().min(1),
  balanceDate: z.string().min(1),
});

export async function GET() {
  const allProjects = await db.select().from(projects).orderBy(desc(projects.createdAt));
  return NextResponse.json(allProjects);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = createProjectSchema.parse(body);
    const now = new Date().toISOString();
    const id = uuid();

    await db.insert(projects).values({
      id,
      ...parsed,
      status: "active",
      createdAt: now,
      updatedAt: now,
    });

    const project = await db.query.projects.findFirst({
      where: (p, { eq }) => eq(p.id, id),
    });

    return NextResponse.json(project, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: "创建项目失败" }, { status: 500 });
  }
}
