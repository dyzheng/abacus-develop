"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
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
import { ArrowLeft, Send, CheckCircle, FileText, Pencil } from "lucide-react";
import { toast } from "sonner";

const STATUS_TIMELINE = ["draft", "generated", "sent", "received", "reconciled"];
const STATUS_LABELS: Record<string, string> = {
  draft: "草稿",
  generated: "已生成",
  sent: "已发出",
  received: "已收回",
  reconciled: "已核对",
  alternative_procedure: "替代程序",
};

export default function ConfirmationDetailPage() {
  const params = useParams();
  const projectId = params.id as string;
  const confirmationId = params.cid as string;
  const router = useRouter();

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [generating, setGenerating] = useState(false);

  // Letter edit dialog
  const [editLetterOpen, setEditLetterOpen] = useState(false);
  const [editLetterContent, setEditLetterContent] = useState("");
  const [savingLetter, setSavingLetter] = useState(false);

  // Response form
  const [responseForm, setResponseForm] = useState({
    responseType: "agree",
    respondedAmount: "",
    respondentName: "",
    respondentTitle: "",
    responseDate: new Date().toISOString().split("T")[0],
    notes: "",
  });

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      try {
        const res = await fetch(`/api/confirmations?projectId=${projectId}`, { signal: controller.signal });
        const rows = await res.json();
        const row = rows.find((r: any) => r.confirmation.id === confirmationId);
        if (row) {
          setData(row);
          setResponseForm((f) => ({
            ...f,
            respondedAmount: String(row.arRecord?.totalBalance || 0),
          }));
        }
      } catch (err) {
        if (!controller.signal.aborted) toast.error("加载失败");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    fetchData();
    return () => controller.abort();
  }, [projectId, confirmationId]);

  const refetchData = useCallback(async () => {
    try {
      const res = await fetch(`/api/confirmations?projectId=${projectId}`);
      const rows = await res.json();
      const row = rows.find((r: any) => r.confirmation.id === confirmationId);
      if (row) {
        setData(row);
        setResponseForm((f) => ({
          ...f,
          respondedAmount: String(row.arRecord?.totalBalance || 0),
        }));
      }
    } catch {
      toast.error("加载失败");
    }
  }, [projectId, confirmationId]);

  async function updateStatus(status: string) {
    setUpdating(true);
    try {
      const body: any = { id: confirmationId, status };
      if (status === "sent") {
        body.sentDate = new Date().toISOString().split("T")[0];
        body.dueDate = new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0];
      }
      await fetch("/api/confirmations", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      toast.success("状态已更新");
      await refetchData();
    } catch {
      toast.error("更新失败");
    } finally {
      setUpdating(false);
    }
  }

  async function generateLetter() {
    setGenerating(true);
    try {
      const res = await fetch("/api/confirmations/generate-letter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId, confirmationIds: [confirmationId] }),
      });
      if (!res.ok) throw new Error();
      toast.success("函证信函已生成");
      await refetchData();
    } catch {
      toast.error("生成失败");
    } finally {
      setGenerating(false);
    }
  }

  function openEditLetter() {
    setEditLetterContent(data?.confirmation?.letterContent || "");
    setEditLetterOpen(true);
  }

  async function saveLetter() {
    setSavingLetter(true);
    try {
      const res = await fetch("/api/confirmations/generate-letter", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmationId, letterContent: editLetterContent }),
      });
      if (!res.ok) throw new Error();
      toast.success("信函已保存");
      setEditLetterOpen(false);
      await refetchData();
    } catch {
      toast.error("保存失败");
    } finally {
      setSavingLetter(false);
    }
  }

  async function submitResponse(e: React.FormEvent) {
    e.preventDefault();
    setUpdating(true);
    try {
      const res = await fetch("/api/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmationId,
          projectId,
          responseType: responseForm.responseType,
          respondedAmount: Number(responseForm.respondedAmount),
          respondentName: responseForm.respondentName,
          respondentTitle: responseForm.respondentTitle,
          responseDate: responseForm.responseDate,
          notes: responseForm.notes,
        }),
      });
      if (!res.ok) throw new Error();
      toast.success("回函登记成功");
      await refetchData();
    } catch {
      toast.error("登记失败");
    } finally {
      setUpdating(false);
    }
  }

  if (loading) return <PageLoading />;
  if (!data) return <div className="p-8 text-center text-muted-foreground">函证不存在</div>;

  const { confirmation: c, arRecord: ar } = data;
  const currentIndex = STATUS_TIMELINE.indexOf(c.status);

  return (
    <div className="max-w-3xl mx-auto">
      <Button variant="ghost" className="mb-4" onClick={() => router.push(`/projects/${projectId}/confirmations`)}>
        <ArrowLeft className="h-4 w-4 mr-2" /> 返回函证列表
      </Button>

      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">{c.confirmationNumber}</h1>
          <p className="text-muted-foreground">{ar?.customerName}</p>
        </div>
        <Badge variant={c.status === "reconciled" ? "success" : c.status === "sent" ? "default" : "secondary"} className="text-base px-3 py-1">
          {STATUS_LABELS[c.status] || c.status}
        </Badge>
      </div>

      {/* Status Timeline */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">状态时间线</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            {STATUS_TIMELINE.map((status, i) => (
              <div key={status} className="flex items-center">
                <div className={`flex flex-col items-center ${i <= currentIndex ? "text-primary" : "text-muted-foreground"}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                    i <= currentIndex ? "bg-primary text-white" : "bg-muted"
                  }`}>
                    {i < currentIndex ? <CheckCircle className="h-4 w-4" /> : i + 1}
                  </div>
                  <span className="text-xs mt-1">{STATUS_LABELS[status]}</span>
                </div>
                {i < STATUS_TIMELINE.length - 1 && (
                  <div className={`w-16 h-0.5 mx-1 ${i < currentIndex ? "bg-primary" : "bg-muted"}`} />
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Info */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">函证信息</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div><span className="text-muted-foreground">函证金额：</span>{formatAmount(ar?.totalBalance || 0)}</div>
            <div><span className="text-muted-foreground">函证类型：</span>{c.type === "positive" ? "积极式" : "空白式"}</div>
            <div><span className="text-muted-foreground">发出日期：</span>{formatDate(c.sentDate)}</div>
            <div><span className="text-muted-foreground">到期日期：</span>{formatDate(c.dueDate)}</div>
            <div><span className="text-muted-foreground">收到日期：</span>{formatDate(c.receivedDate)}</div>
            <div><span className="text-muted-foreground">创建时间：</span>{formatDate(c.createdAt)}</div>
          </div>
        </CardContent>
      </Card>

      {/* Letter Content */}
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4" /> 函证信函
            </CardTitle>
            {c.letterContent && (
              <div className="flex items-center gap-2">
                {c.letterGeneratedAt && (
                  <span className="text-xs text-muted-foreground">生成于 {formatDate(c.letterGeneratedAt)}</span>
                )}
                <Button size="sm" variant="outline" onClick={openEditLetter}>
                  <Pencil className="h-3 w-3 mr-1" /> 编辑信函
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {c.letterContent ? (
            <div className="bg-white border rounded-lg p-6 whitespace-pre-wrap text-sm leading-relaxed font-serif">
              {c.letterContent}
            </div>
          ) : (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
              <p className="text-muted-foreground mb-4">尚未生成函证信函</p>
              {c.status === "draft" && (
                <Button onClick={generateLetter} disabled={generating}>
                  {generating ? <LoadingSpinner className="mr-2" /> : <FileText className="h-4 w-4 mr-2" />}
                  生成函证信函
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Actions based on status */}
      {c.status === "draft" && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex gap-3">
              <Button onClick={() => updateStatus("sent")} disabled={updating}>
                <Send className="h-4 w-4 mr-2" /> 标记已发出
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {c.status === "sent" && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">登记回函</CardTitle>
            <CardDescription>记录收到的回函信息</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={submitResponse} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>回函类型</Label>
                  <Select
                    value={responseForm.responseType}
                    onChange={(e) => setResponseForm({ ...responseForm, responseType: e.target.value })}
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
                  <Input
                    type="number"
                    step="0.01"
                    value={responseForm.respondedAmount}
                    onChange={(e) => setResponseForm({ ...responseForm, respondedAmount: e.target.value })}
                  />
                </div>
                <div>
                  <Label>回函人</Label>
                  <Input
                    value={responseForm.respondentName}
                    onChange={(e) => setResponseForm({ ...responseForm, respondentName: e.target.value })}
                  />
                </div>
                <div>
                  <Label>职务</Label>
                  <Input
                    value={responseForm.respondentTitle}
                    onChange={(e) => setResponseForm({ ...responseForm, respondentTitle: e.target.value })}
                  />
                </div>
                <div>
                  <Label>回函日期</Label>
                  <Input
                    type="date"
                    value={responseForm.responseDate}
                    onChange={(e) => setResponseForm({ ...responseForm, responseDate: e.target.value })}
                  />
                </div>
              </div>
              <div>
                <Label>备注</Label>
                <Textarea
                  value={responseForm.notes}
                  onChange={(e) => setResponseForm({ ...responseForm, notes: e.target.value })}
                  placeholder="如有差异请说明原因"
                />
              </div>
              <Button type="submit" disabled={updating}>
                {updating ? <LoadingSpinner className="mr-2" /> : null}
                登记回函
              </Button>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Edit Letter Dialog */}
      <Dialog open={editLetterOpen} onOpenChange={setEditLetterOpen}>
        <DialogContent onClose={() => setEditLetterOpen(false)} className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>编辑函证信函</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <Textarea
              value={editLetterContent}
              onChange={(e) => setEditLetterContent(e.target.value)}
              className="min-h-[400px] font-mono text-sm"
            />
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setEditLetterOpen(false)}>取消</Button>
              <Button onClick={saveLetter} disabled={savingLetter || !editLetterContent.trim()}>
                {savingLetter ? <LoadingSpinner className="mr-2" /> : null}
                保存
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
