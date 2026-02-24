"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import { FileText, Download, Sparkles } from "lucide-react";
import { toast } from "sonner";

interface ReportData {
  project: {
    name: string;
    clientCompany: string;
    auditFirm: string;
    balanceDate: string;
  };
  summary: {
    totalARCount: number;
    totalARBalance: number;
    confirmationCount: number;
    confirmedBalance: number;
    coverageRate: string;
    statusCounts: Record<string, number>;
    responseCount: number;
    agreeCount: number;
    disagreeCount: number;
    noResponseCount: number;
    responseRate: string;
    differenceCount: number;
    resolvedCount: number;
    totalDifferenceAmount: number;
  };
}

export default function ReportPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [report, setReport] = useState<ReportData | null>(null);
  const [narrative, setNarrative] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchReport() {
      try {
        const res = await fetch(`/api/report?projectId=${projectId}`, { signal: controller.signal });
        const data = await res.json();
        setReport(data);
      } catch (err) {
        if (!controller.signal.aborted) toast.error("加载报告失败");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    fetchReport();
    return () => controller.abort();
  }, [projectId]);

  async function generateNarrative() {
    setGenerating(true);
    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setNarrative(data.narrative);
      toast.success("报告文字生成完成");
    } catch {
      toast.error("生成失败");
    } finally {
      setGenerating(false);
    }
  }

  if (loading) return <PageLoading />;
  if (!report) return <div className="p-8 text-center">无法加载报告数据</div>;

  const { project, summary: s } = report;

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">函证结果汇总</h1>
          <p className="text-muted-foreground">{project.clientCompany} - {project.balanceDate}</p>
        </div>
        <Button variant="outline" onClick={() => window.print()}>
          <Download className="h-4 w-4 mr-2" /> 打印/导出
        </Button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-xs text-muted-foreground">应收账款总额</p>
            <p className="text-lg font-bold">{formatAmount(s.totalARBalance)}</p>
            <p className="text-xs text-muted-foreground">{s.totalARCount} 笔</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-xs text-muted-foreground">函证覆盖率</p>
            <p className="text-2xl font-bold text-primary">{s.coverageRate}%</p>
            <p className="text-xs text-muted-foreground">{s.confirmationCount} 份函证</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-xs text-muted-foreground">回函率</p>
            <p className="text-2xl font-bold text-primary">{s.responseRate}%</p>
            <p className="text-xs text-muted-foreground">{s.responseCount} 份回函</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-xs text-muted-foreground">差异金额</p>
            <p className="text-lg font-bold text-red-600">{formatAmount(s.totalDifferenceAmount)}</p>
            <p className="text-xs text-muted-foreground">{s.differenceCount} 项差异</p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Stats */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">函证状态分布</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(s.statusCounts).map(([status, count]) => {
                const labels: Record<string, string> = {
                  draft: "草稿", generated: "已生成", sent: "已发出",
                  received: "已收回", reconciled: "已核对", alternative_procedure: "替代程序",
                };
                const pct = s.confirmationCount > 0 ? ((count / s.confirmationCount) * 100).toFixed(0) : "0";
                return (
                  <div key={status} className="flex items-center gap-3">
                    <span className="text-sm w-20">{labels[status] || status}</span>
                    <div className="flex-1 bg-muted rounded-full h-2">
                      <div
                        className="bg-primary rounded-full h-2 transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-sm font-mono w-16 text-right">{count} ({pct}%)</span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">回函结果分析</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">确认相符</span>
                <div className="flex items-center gap-2">
                  <Badge variant="success">{s.agreeCount}</Badge>
                  <span className="text-sm text-muted-foreground">
                    {s.responseCount > 0 ? ((s.agreeCount / s.responseCount) * 100).toFixed(0) : 0}%
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">存在差异</span>
                <div className="flex items-center gap-2">
                  <Badge variant="destructive">{s.disagreeCount}</Badge>
                  <span className="text-sm text-muted-foreground">
                    {s.responseCount > 0 ? ((s.disagreeCount / s.responseCount) * 100).toFixed(0) : 0}%
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">未回函</span>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{s.noResponseCount}</Badge>
                </div>
              </div>
              <hr />
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">差异解决进度</span>
                <span className="text-sm font-mono">
                  {s.resolvedCount} / {s.differenceCount}
                  {s.differenceCount > 0 && ` (${((s.resolvedCount / s.differenceCount) * 100).toFixed(0)}%)`}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Narrative */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">报告正文</CardTitle>
              <CardDescription>AI生成的函证结果汇总报告</CardDescription>
            </div>
            <Button onClick={generateNarrative} disabled={generating}>
              {generating ? <LoadingSpinner className="mr-2" /> : <Sparkles className="h-4 w-4 mr-2" />}
              {generating ? "生成中..." : narrative ? "重新生成" : "AI生成报告"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {narrative ? (
            <div className="prose prose-sm max-w-none whitespace-pre-wrap border rounded-lg p-6 bg-white">
              <h2 className="text-center text-lg font-bold mb-4">
                {project.clientCompany}
                <br />
                应收账款函证结果汇总报告
              </h2>
              <p className="text-sm text-muted-foreground text-center mb-6">
                审计机构：{project.auditFirm} | 基准日：{project.balanceDate}
              </p>
              {narrative}
            </div>
          ) : (
            <div className="flex flex-col items-center py-8 text-muted-foreground">
              <FileText className="h-12 w-12 mb-4" />
              <p>点击"AI生成报告"按钮生成函证结果汇总报告</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
