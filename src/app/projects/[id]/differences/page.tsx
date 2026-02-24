"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import { Brain, CheckCircle, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

interface DifferenceRow {
  difference: {
    id: string;
    bookAmount: number;
    confirmedAmount: number;
    differenceAmount: number;
    aiSuggestedCause: string | null;
    aiAnalysisDetail: string | null;
    aiConfidenceScore: number | null;
    auditorResolution: string | null;
    status: string | null;
  };
  response: { responseType: string } | null;
  confirmation: { confirmationNumber: string } | null;
  arRecord: { customerName: string } | null;
}

const STATUS_MAP: Record<string, { label: string; variant: "secondary" | "info" | "warning" | "success" }> = {
  pending: { label: "待分析", variant: "secondary" },
  analyzing: { label: "分析中", variant: "info" },
  analyzed: { label: "已分析", variant: "warning" },
  resolved: { label: "已解决", variant: "success" },
};

export default function DifferencesPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [rows, setRows] = useState<DifferenceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [resolution, setResolution] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      try {
        const res = await fetch(`/api/differences?projectId=${projectId}`, { signal: controller.signal });
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
      const res = await fetch(`/api/differences?projectId=${projectId}`);
      setRows(await res.json());
    } catch {
      toast.error("加载失败");
    }
  }, [projectId]);

  async function runAnalysis(differenceId: string) {
    setAnalyzing(differenceId);
    try {
      const res = await fetch("/api/differences/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ differenceId, projectId }),
      });
      if (!res.ok) throw new Error();
      toast.success("AI分析完成");
      await refetchData();
    } catch {
      toast.error("分析失败");
    } finally {
      setAnalyzing(null);
    }
  }

  async function saveResolution(differenceId: string) {
    if (!resolution[differenceId]) return;
    setSaving(differenceId);
    try {
      const res = await fetch("/api/differences", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: differenceId, auditorResolution: resolution[differenceId] }),
      });
      if (!res.ok) throw new Error();
      toast.success("已保存审计师结论");
      await refetchData();
    } catch {
      toast.error("保存失败");
    } finally {
      setSaving(null);
    }
  }

  if (loading) return <PageLoading />;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">差异分析</h1>
      <p className="text-muted-foreground mb-6">
        共 {rows.length} 项差异，{rows.filter((r) => r.difference.status === "resolved").length} 项已解决
      </p>

      {rows.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            暂无差异记录。当回函登记中存在金额差异时，系统将自动创建差异记录。
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {rows.map((row) => {
            const d = row.difference;
            return (
              <Card key={d.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-base">{row.arRecord?.customerName || "-"}</CardTitle>
                      <CardDescription>{row.confirmation?.confirmationNumber}</CardDescription>
                    </div>
                    <Badge variant={STATUS_MAP[d.status || "pending"]?.variant || "secondary"}>
                      {STATUS_MAP[d.status || "pending"]?.label || d.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Amount comparison */}
                  <div className="grid grid-cols-3 gap-4 p-4 bg-muted/30 rounded-lg">
                    <div>
                      <p className="text-xs text-muted-foreground">账面金额</p>
                      <p className="text-lg font-bold font-mono">{formatAmount(d.bookAmount)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">确认金额</p>
                      <p className="text-lg font-bold font-mono">{formatAmount(d.confirmedAmount)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">差异金额</p>
                      <p className={`text-lg font-bold font-mono ${d.differenceAmount > 0 ? "text-green-600" : "text-red-600"}`}>
                        {formatAmount(d.differenceAmount)}
                      </p>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  {d.aiSuggestedCause ? (
                    <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="h-4 w-4 text-blue-600" />
                        <span className="font-medium text-blue-800">AI分析结果</span>
                        {d.aiConfidenceScore && (
                          <Badge variant="info">置信度 {(d.aiConfidenceScore * 100).toFixed(0)}%</Badge>
                        )}
                      </div>
                      <p className="font-semibold mb-1">{d.aiSuggestedCause}</p>
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">{d.aiAnalysisDetail}</p>
                    </div>
                  ) : (
                    <Button
                      variant="outline"
                      onClick={() => runAnalysis(d.id)}
                      disabled={analyzing === d.id}
                    >
                      {analyzing === d.id ? <LoadingSpinner className="mr-2" /> : <Brain className="h-4 w-4 mr-2" />}
                      {analyzing === d.id ? "AI分析中..." : "运行AI分析"}
                    </Button>
                  )}

                  {/* Auditor Resolution */}
                  {d.status === "resolved" ? (
                    <div className="p-4 bg-green-50 rounded-lg border border-green-100">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                        <span className="font-medium text-green-800">审计师结论</span>
                      </div>
                      <p className="text-sm">{d.auditorResolution}</p>
                    </div>
                  ) : (
                    <div>
                      <Label>审计师结论</Label>
                      <Textarea
                        value={resolution[d.id] || ""}
                        onChange={(e) => setResolution({ ...resolution, [d.id]: e.target.value })}
                        placeholder="请输入审计师对差异的分析结论和处理意见..."
                        className="mt-1"
                      />
                      <Button
                        className="mt-2"
                        size="sm"
                        onClick={() => saveResolution(d.id)}
                        disabled={!resolution[d.id] || saving === d.id}
                      >
                        {saving === d.id ? <LoadingSpinner className="mr-2" /> : null}
                        保存结论
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
