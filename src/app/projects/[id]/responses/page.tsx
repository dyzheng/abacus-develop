"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount, formatDate } from "@/lib/utils";
import { ClipboardCheck, Plus } from "lucide-react";
import { toast } from "sonner";

interface ResponseRow {
  response: {
    id: string;
    responseType: string;
    respondedAmount: number | null;
    differenceAmount: number | null;
    respondentName: string | null;
    responseDate: string | null;
    notes: string | null;
    createdAt: string;
  };
  confirmation: {
    id: string;
    confirmationNumber: string;
    status: string;
  } | null;
  arRecord: {
    customerName: string;
    totalBalance: number;
  } | null;
}

const RESPONSE_TYPE_MAP: Record<string, { label: string; variant: "success" | "destructive" | "warning" | "secondary" }> = {
  agree: { label: "确认相符", variant: "success" },
  disagree: { label: "不符", variant: "destructive" },
  partial: { label: "部分相符", variant: "warning" },
  no_response: { label: "未回函", variant: "secondary" },
};

export default function ResponsesPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [rows, setRows] = useState<ResponseRow[]>([]);
  const [confirmations, setConfirmations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [form, setForm] = useState({
    confirmationId: "",
    responseType: "agree",
    respondedAmount: "",
    respondentName: "",
    respondentTitle: "",
    responseDate: new Date().toISOString().split("T")[0],
    notes: "",
  });

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    try {
      const [respRes, confRes] = await Promise.all([
        fetch(`/api/responses?projectId=${projectId}`),
        fetch(`/api/confirmations?projectId=${projectId}`),
      ]);
      setRows(await respRes.json());
      const confData = await confRes.json();
      setConfirmations(confData);
    } catch {
      toast.error("加载失败");
    } finally {
      setLoading(false);
    }
  }

  function openDialog(confirmationId?: string) {
    const conf = confirmations.find((c: any) => c.confirmation.id === confirmationId);
    setForm({
      confirmationId: confirmationId || "",
      responseType: "agree",
      respondedAmount: String(conf?.arRecord?.totalBalance || ""),
      respondentName: "",
      respondentTitle: "",
      responseDate: new Date().toISOString().split("T")[0],
      notes: "",
    });
    setDialogOpen(true);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.confirmationId) {
      toast.error("请选择函证");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmationId: form.confirmationId,
          projectId,
          responseType: form.responseType,
          respondedAmount: Number(form.respondedAmount) || 0,
          respondentName: form.respondentName,
          respondentTitle: form.respondentTitle,
          responseDate: form.responseDate,
          notes: form.notes,
        }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      toast.success("回函登记成功");
      if (data.differenceAmount && Math.abs(data.differenceAmount) > 0.01) {
        toast.info(`检测到差异: ${formatAmount(data.differenceAmount)}，已自动创建差异记录`);
      }
      setDialogOpen(false);
      await fetchData();
    } catch {
      toast.error("登记失败");
    } finally {
      setSubmitting(false);
    }
  }

  // Sent confirmations without responses
  const sentConfs = confirmations.filter(
    (c: any) => c.confirmation.status === "sent" && !rows.some((r) => r.confirmation?.id === c.confirmation.id)
  );

  if (loading) return <PageLoading />;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">回函登记</h1>
          <p className="text-muted-foreground">记录函证回函结果</p>
        </div>
        <Button onClick={() => openDialog()}>
          <Plus className="h-4 w-4 mr-2" /> 登记回函
        </Button>
      </div>

      {/* Pending responses */}
      {sentConfs.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">待回函 ({sentConfs.length})</CardTitle>
            <CardDescription>以下函证已发出但尚未收到回函</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {sentConfs.map((c: any) => (
                <div key={c.confirmation.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-md">
                  <div>
                    <span className="font-medium">{c.arRecord?.customerName}</span>
                    <span className="text-sm text-muted-foreground ml-3">{c.confirmation.confirmationNumber}</span>
                    <span className="text-sm text-muted-foreground ml-3">{formatAmount(c.arRecord?.totalBalance || 0)}</span>
                  </div>
                  <Button size="sm" onClick={() => openDialog(c.confirmation.id)}>
                    登记回函
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Response records */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium">函证编号</th>
                  <th className="px-4 py-3 text-left font-medium">客户名称</th>
                  <th className="px-4 py-3 text-right font-medium">账面金额</th>
                  <th className="px-4 py-3 text-right font-medium">确认金额</th>
                  <th className="px-4 py-3 text-right font-medium">差异金额</th>
                  <th className="px-4 py-3 text-center font-medium">回函类型</th>
                  <th className="px-4 py-3 text-center font-medium">回函日期</th>
                  <th className="px-4 py-3 text-left font-medium">回函人</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const r = row.response;
                  const hasDiff = r.differenceAmount && Math.abs(r.differenceAmount) > 0.01;
                  return (
                    <tr key={r.id} className="border-b hover:bg-muted/30">
                      <td className="px-4 py-3 font-mono text-xs">{row.confirmation?.confirmationNumber || "-"}</td>
                      <td className="px-4 py-3 font-medium">{row.arRecord?.customerName || "-"}</td>
                      <td className="px-4 py-3 text-right font-mono">{formatAmount(row.arRecord?.totalBalance || 0)}</td>
                      <td className="px-4 py-3 text-right font-mono">{formatAmount(r.respondedAmount)}</td>
                      <td className={`px-4 py-3 text-right font-mono ${hasDiff ? "text-red-600 font-semibold" : ""}`}>
                        {hasDiff ? formatAmount(r.differenceAmount) : "-"}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant={RESPONSE_TYPE_MAP[r.responseType]?.variant || "secondary"}>
                          {RESPONSE_TYPE_MAP[r.responseType]?.label || r.responseType}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-center text-xs">{formatDate(r.responseDate)}</td>
                      <td className="px-4 py-3">{r.respondentName || "-"}</td>
                    </tr>
                  );
                })}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-12 text-center text-muted-foreground">
                      暂无回函记录
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent onClose={() => setDialogOpen(false)} className="max-w-md">
          <DialogHeader>
            <DialogTitle>登记回函</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Label>选择函证</Label>
              <select
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                value={form.confirmationId}
                onChange={(e) => {
                  const conf = confirmations.find((c: any) => c.confirmation.id === e.target.value);
                  setForm({
                    ...form,
                    confirmationId: e.target.value,
                    respondedAmount: String(conf?.arRecord?.totalBalance || ""),
                  });
                }}
              >
                <option value="">请选择...</option>
                {confirmations
                  .filter((c: any) => c.confirmation.status === "sent")
                  .map((c: any) => (
                    <option key={c.confirmation.id} value={c.confirmation.id}>
                      {c.confirmation.confirmationNumber} - {c.arRecord?.customerName}
                    </option>
                  ))}
              </select>
            </div>
            <div>
              <Label>回函类型</Label>
              <Select
                value={form.responseType}
                onChange={(e) => setForm({ ...form, responseType: e.target.value })}
                options={[
                  { value: "agree", label: "确认相符" },
                  { value: "disagree", label: "不符" },
                  { value: "partial", label: "部分相符" },
                  { value: "no_response", label: "未回函" },
                ]}
              />
            </div>
            <div>
              <Label>确认金额</Label>
              <Input type="number" step="0.01" value={form.respondedAmount} onChange={(e) => setForm({ ...form, respondedAmount: e.target.value })} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>回函人</Label>
                <Input value={form.respondentName} onChange={(e) => setForm({ ...form, respondentName: e.target.value })} />
              </div>
              <div>
                <Label>回函日期</Label>
                <Input type="date" value={form.responseDate} onChange={(e) => setForm({ ...form, responseDate: e.target.value })} />
              </div>
            </div>
            <div>
              <Label>备注</Label>
              <Textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} placeholder="如有差异请说明" />
            </div>
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>取消</Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? <LoadingSpinner className="mr-2" /> : null}
                提交
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
