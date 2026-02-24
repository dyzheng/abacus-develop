"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Sidebar } from "@/components/sidebar";
import { Plus, FolderOpen, Building2, Calendar, Briefcase } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { formatDate } from "@/lib/utils";

interface Project {
  id: string;
  name: string;
  clientCompany: string;
  auditFirm: string;
  balanceDate: string;
  status: string;
  createdAt: string;
}

const statusMap: Record<string, { label: string; variant: "default" | "success" | "secondary" }> = {
  active: { label: "进行中", variant: "default" },
  completed: { label: "已完成", variant: "success" },
  archived: { label: "已归档", variant: "secondary" },
};

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const router = useRouter();

  const [form, setForm] = useState({
    name: "",
    clientCompany: "",
    auditFirm: "",
    balanceDate: "",
  });

  useEffect(() => {
    const controller = new AbortController();
    async function fetchProjects() {
      try {
        const res = await fetch("/api/projects", { signal: controller.signal });
        const data = await res.json();
        setProjects(data);
      } catch (e: any) {
        if (e.name !== "AbortError") toast.error("加载项目列表失败");
      } finally {
        setLoading(false);
      }
    }
    fetchProjects();
    return () => controller.abort();
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error();
      const project = await res.json();
      toast.success("项目创建成功");
      setOpen(false);
      setForm({ name: "", clientCompany: "", auditFirm: "", balanceDate: "" });
      router.push(`/projects/${project.id}`);
    } catch {
      toast.error("创建项目失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 page-enter overflow-auto">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">审计项目</h1>
              <p className="text-muted-foreground mt-1">管理所有应收账款函证审计项目</p>
            </div>
            <Button onClick={() => setOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              新建项目
            </Button>
          </div>

          {loading ? (
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className={`card-ink rounded-xl p-6 animate-fade-in animate-delay-${i}`}>
                  <div className="h-5 bg-muted rounded-md w-3/4 animate-pulse" />
                  <div className="h-4 bg-muted rounded-md w-1/2 mt-3 animate-pulse" />
                  <div className="h-4 bg-muted rounded-md w-full mt-6 animate-pulse" />
                  <div className="h-4 bg-muted rounded-md w-2/3 mt-2 animate-pulse" />
                </div>
              ))}
            </div>
          ) : projects.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-5">
                  <FolderOpen className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="text-lg font-medium font-display">暂无审计项目</p>
                <p className="text-muted-foreground mt-1 mb-5 text-sm">点击下方按钮创建第一个审计项目</p>
                <Button onClick={() => setOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  新建项目
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
              {projects.map((project, i) => (
                <div
                  key={project.id}
                  className={`card-ink rounded-xl cursor-pointer group animate-fade-in animate-delay-${Math.min(i + 1, 6)}`}
                  onClick={() => router.push(`/projects/${project.id}`)}
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="font-semibold text-ink leading-tight group-hover:text-seal transition-colors">
                        {project.name}
                      </h3>
                      <Badge variant={statusMap[project.status]?.variant || "default"} className="ml-2 shrink-0">
                        {statusMap[project.status]?.label || project.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">{project.clientCompany}</p>
                    <div className="space-y-2 text-xs text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Briefcase className="h-3.5 w-3.5" />
                        <span>{project.auditFirm}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="h-3.5 w-3.5" />
                        <span>基准日 {formatDate(project.balanceDate)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="px-6 py-3 border-t border-border/50 bg-paper-dark/30 rounded-b-xl">
                    <span className="text-[11px] text-muted-foreground">
                      创建于 {formatDate(project.createdAt)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent onClose={() => setOpen(false)}>
              <DialogHeader>
                <DialogTitle>新建审计项目</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleCreate} className="space-y-4 mt-2">
                <div className="space-y-1.5">
                  <Label htmlFor="name">项目名称</Label>
                  <Input id="name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required placeholder="例: 2024年度应收账款函证" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="clientCompany">被审计单位</Label>
                  <Input id="clientCompany" value={form.clientCompany} onChange={(e) => setForm({ ...form, clientCompany: e.target.value })} required placeholder="例: XX有限公司" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="auditFirm">审计机构</Label>
                  <Input id="auditFirm" value={form.auditFirm} onChange={(e) => setForm({ ...form, auditFirm: e.target.value })} required placeholder="例: XX会计师事务所" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="balanceDate">基准日</Label>
                  <Input id="balanceDate" type="date" value={form.balanceDate} onChange={(e) => setForm({ ...form, balanceDate: e.target.value })} required />
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <Button type="button" variant="outline" onClick={() => setOpen(false)}>取消</Button>
                  <Button type="submit" disabled={submitting}>{submitting ? "创建中..." : "创建项目"}</Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </main>
    </div>
  );
}
