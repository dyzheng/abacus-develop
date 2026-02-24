"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount, formatDate } from "@/lib/utils";
import { Send, FileText } from "lucide-react";
import { toast } from "sonner";

interface ConfirmationRow {
  confirmation: {
    id: string;
    confirmationNumber: string;
    type: string;
    status: string;
    sentDate: string | null;
    dueDate: string | null;
    receivedDate: string | null;
    letterContent: string | null;
    letterGeneratedAt: string | null;
    createdAt: string;
  };
  arRecord: {
    customerName: string;
    totalBalance: number;
  } | null;
}

const STATUS_MAP: Record<string, { label: string; variant: "default" | "secondary" | "info" | "warning" | "success" | "destructive" }> = {
  draft: { label: "草稿", variant: "secondary" },
  generated: { label: "已生成", variant: "info" },
  sent: { label: "已发出", variant: "default" },
  received: { label: "已收回", variant: "success" },
  reconciled: { label: "已核对", variant: "success" },
  alternative_procedure: { label: "替代程序", variant: "warning" },
};

export default function ConfirmationsPage() {
  const params = useParams();
  const projectId = params.id as string;
  const router = useRouter();

  const [rows, setRows] = useState<ConfirmationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [updating, setUpdating] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generatingSingle, setGeneratingSingle] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      try {
        const res = await fetch(`/api/confirmations?projectId=${projectId}`, { signal: controller.signal });
        setRows(await res.json());
      } catch (err) {
        if (!controller.signal.aborted) toast.error("加载失败");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    fetchData();
    return () => controller.abort();
  }, [projectId]);

  const refetchData = useCallback(async () => {
    try {
      const res = await fetch(`/api/confirmations?projectId=${projectId}`);
      setRows(await res.json());
    } catch {
      toast.error("加载失败");
    }
  }, [projectId]);

  async function batchUpdateStatus(status: string) {
    if (selected.size === 0) return;
    setUpdating(true);
    try {
      const body: any = { ids: Array.from(selected), status };
      if (status === "sent") {
        body.sentDate = new Date().toISOString().split("T")[0];
        body.dueDate = new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0];
      }
      const res = await fetch("/api/confirmations", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error();
      toast.success(`已更新 ${selected.size} 份函证状态`);
      setSelected(new Set());
      await refetchData();
    } catch {
      toast.error("更新失败");
    } finally {
      setUpdating(false);
    }
  }

  async function batchGenerateLetters() {
    const draftIds = Array.from(selected).filter((id) => {
      const row = rows.find((r) => r.confirmation.id === id);
      return row?.confirmation.status === "draft";
    });
    if (draftIds.length === 0) {
      toast.error("请选择草稿状态的函证");
      return;
    }
    setGenerating(true);
    try {
      const res = await fetch("/api/confirmations/generate-letter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId, confirmationIds: draftIds }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      toast.success(`已生成 ${data.generated} 份函证信函`);
      setSelected(new Set());
      await refetchData();
    } catch {
      toast.error("生成失败");
    } finally {
      setGenerating(false);
    }
  }

  async function generateSingleLetter(cid: string) {
    setGeneratingSingle(cid);
    try {
      const res = await fetch("/api/confirmations/generate-letter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId, confirmationIds: [cid] }),
      });
      if (!res.ok) throw new Error();
      toast.success("函证信函已生成");
      await refetchData();
    } catch {
      toast.error("生成失败");
    } finally {
      setGeneratingSingle(null);
    }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  if (loading) return <PageLoading />;

  const statusGroups = rows.reduce((acc, r) => {
    const s = r.confirmation.status;
    acc[s] = (acc[s] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const hasDraftSelected = Array.from(selected).some((id) => {
    const row = rows.find((r) => r.confirmation.id === id);
    return row?.confirmation.status === "draft";
  });

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">函证管理</h1>
      <p className="text-muted-foreground mb-6">管理函证状态，执行批量操作</p>

      {/* Status summary */}
      <div className="flex flex-wrap gap-3 mb-6">
        {Object.entries(STATUS_MAP).map(([key, { label }]) => (
          <Card key={key} className="flex-1 min-w-[100px]">
            <CardContent className="pt-3 pb-3 text-center">
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className="text-xl font-bold">{statusGroups[key] || 0}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Batch actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 mb-4 p-3 bg-blue-50 rounded-lg">
          <span className="text-sm font-medium">已选择 {selected.size} 项</span>
          {hasDraftSelected && (
            <Button size="sm" onClick={batchGenerateLetters} disabled={generating || updating}>
              {generating ? <LoadingSpinner className="mr-1" /> : <FileText className="h-3 w-3 mr-1" />}
              一键生成函证
            </Button>
          )}
          <Button size="sm" onClick={() => batchUpdateStatus("sent")} disabled={updating || generating}>
            <Send className="h-3 w-3 mr-1" /> 标记已发出
          </Button>
          <Button size="sm" variant="outline" onClick={() => batchUpdateStatus("alternative_procedure")} disabled={updating || generating}>
            替代程序
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>
            取消选择
          </Button>
        </div>
      )}

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 w-10">
                    <input
                      type="checkbox"
                      checked={selected.size === rows.length && rows.length > 0}
                      onChange={(e) => {
                        if (e.target.checked) setSelected(new Set(rows.map((r) => r.confirmation.id)));
                        else setSelected(new Set());
                      }}
                    />
                  </th>
                  <th className="px-4 py-3 text-left font-medium">函证编号</th>
                  <th className="px-4 py-3 text-left font-medium">客户名称</th>
                  <th className="px-4 py-3 text-right font-medium">函证金额</th>
                  <th className="px-4 py-3 text-center font-medium">类型</th>
                  <th className="px-4 py-3 text-center font-medium">状态</th>
                  <th className="px-4 py-3 text-center font-medium">发出日期</th>
                  <th className="px-4 py-3 text-center font-medium">到期日</th>
                  <th className="px-4 py-3 text-center font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const c = row.confirmation;
                  const isOverdue = c.dueDate && new Date(c.dueDate) < new Date() && c.status === "sent";
                  return (
                    <tr key={c.id} className={`border-b hover:bg-muted/30 ${isOverdue ? "bg-red-50" : ""}`}>
                      <td className="px-4 py-3">
                        <input type="checkbox" checked={selected.has(c.id)} onChange={() => toggleSelect(c.id)} />
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">{c.confirmationNumber}</td>
                      <td className="px-4 py-3 font-medium">{row.arRecord?.customerName || "-"}</td>
                      <td className="px-4 py-3 text-right font-mono">{formatAmount(row.arRecord?.totalBalance || 0)}</td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant="outline">{c.type === "positive" ? "积极式" : "空白式"}</Badge>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant={STATUS_MAP[c.status]?.variant || "secondary"}>
                          {STATUS_MAP[c.status]?.label || c.status}
                        </Badge>
                        {isOverdue && <Badge variant="destructive" className="ml-1 text-[10px]">逾期</Badge>}
                      </td>
                      <td className="px-4 py-3 text-center text-xs">{formatDate(c.sentDate)}</td>
                      <td className="px-4 py-3 text-center text-xs">{formatDate(c.dueDate)}</td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-1">
                          {c.status === "draft" && (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => generateSingleLetter(c.id)}
                              disabled={generatingSingle === c.id}
                            >
                              {generatingSingle === c.id ? <LoadingSpinner /> : <FileText className="h-3 w-3" />}
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => router.push(`/projects/${projectId}/confirmations/${c.id}`)}
                          >
                            详情
                          </Button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={9} className="px-4 py-12 text-center text-muted-foreground">
                      暂无函证记录，请先进行样本选择
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
