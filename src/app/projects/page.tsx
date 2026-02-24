"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Sidebar } from "@/components/sidebar";
import { Plus, FolderOpen } from "lucide-react";
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
    fetchProjects();
  }, []);

  async function fetchProjects() {
    try {
      const res = await fetch("/api/projects");
      const data = await res.json();
      setProjects(data);
    } catch {
      toast.error("加载项目列表失败");
    } finally {
      setLoading(false);
    }
  }

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

  const statusMap: Record<string, { label: string; variant: "default" | "success" | "secondary" }> = {
    active: { label: "进行中", variant: "default" },
    completed: { label: "已完成", variant: "success" },
    archived: { label: "已归档", variant: "secondary" },
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-6">
        <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">审计项目列表</h1>
          <p className="text-muted-foreground">管理所有应收账款函证审计项目</p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          新建项目
        </Button>
      </div>

      {loading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-5 bg-muted rounded w-3/4" />
                <div className="h-4 bg-muted rounded w-1/2 mt-2" />
              </CardHeader>
              <CardContent>
                <div className="h-4 bg-muted rounded w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : projects.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FolderOpen className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">暂无项目</p>
            <p className="text-muted-foreground mb-4">点击"新建项目"开始创建第一个审计项目</p>
            <Button onClick={() => setOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              新建项目
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card
              key={project.id}
              className="cursor-pointer hover:border-primary transition-colors"
              onClick={() => router.push(`/projects/${project.id}`)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{project.name}</CardTitle>
                  <Badge variant={statusMap[project.status]?.variant || "default"}>
                    {statusMap[project.status]?.label || project.status}
                  </Badge>
                </div>
                <CardDescription>{project.clientCompany}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm space-y-1 text-muted-foreground">
                  <p>审计机构: {project.auditFirm}</p>
                  <p>基准日: {formatDate(project.balanceDate)}</p>
                  <p>创建时间: {formatDate(project.createdAt)}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent onClose={() => setOpen(false)}>
          <DialogHeader>
            <DialogTitle>新建审计项目</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleCreate} className="space-y-4">
            <div>
              <Label htmlFor="name">项目名称</Label>
              <Input id="name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required placeholder="例: 2024年度应收账款函证" />
            </div>
            <div>
              <Label htmlFor="clientCompany">被审计单位</Label>
              <Input id="clientCompany" value={form.clientCompany} onChange={(e) => setForm({ ...form, clientCompany: e.target.value })} required placeholder="例: XX有限公司" />
            </div>
            <div>
              <Label htmlFor="auditFirm">审计机构</Label>
              <Input id="auditFirm" value={form.auditFirm} onChange={(e) => setForm({ ...form, auditFirm: e.target.value })} required placeholder="例: XX会计师事务所" />
            </div>
            <div>
              <Label htmlFor="balanceDate">基准日</Label>
              <Input id="balanceDate" type="date" value={form.balanceDate} onChange={(e) => setForm({ ...form, balanceDate: e.target.value })} required />
            </div>
            <div className="flex justify-end gap-2">
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
