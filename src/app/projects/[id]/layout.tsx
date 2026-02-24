import { Sidebar } from "@/components/sidebar";
import { db } from "@/db";
import { projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { notFound } from "next/navigation";

export default async function ProjectLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { id: string };
}) {
  const project = await db.query.projects.findFirst({
    where: eq(projects.id, params.id),
  });

  if (!project) {
    notFound();
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar projectId={project.id} projectName={project.name} />
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}
