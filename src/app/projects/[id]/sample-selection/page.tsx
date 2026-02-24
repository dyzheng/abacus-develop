"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import { Brain, Check, X, Sparkles } from "lucide-react";
import { toast } from "sonner";

interface ARRecord {
  id: string;
  customerName: string;
  totalBalance: number;
  riskLevel: string | null;
  isRelatedParty: boolean | null;
  selectionStatus: string | null;
  selectionReason: string | null;
  within1Year: number | null;
  year1to2: number | null;
  year2to3: number | null;
}

interface SelectionResult {
  totalRecords: number;
  selectedCount: number;
  totalBalance: number;
  selectedBalance: number;
  coverageRate: string;
  selections: { id: string; reason: string }[];
}

export default function SampleSelectionPage() {
  const params = useParams();
  const projectId = params.id as string;
  const router = useRouter();

  const [records, setRecords] = useState<ARRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SelectionResult | null>(null);
  const [confirmed, setConfirmed] = useState<Set<string>>(new Set());

  const [criteria, setCriteria] = useState({
    minBalance: 100000,
    includeHighRisk: true,
    includeRelatedParties: true,
    includeLongAged: true,
    maxSamples: 20,
  });

  useEffect(() => {
    fetchRecords();
  }, []);

  async function fetchRecords() {
    try {
      const res = await fetch(`/api/ar-records?projectId=${projectId}`);
      const data = await res.json();
      setRecords(data);
      // Pre-select already selected ones
      const alreadySelected = new Set<string>(
        data.filter((r: ARRecord) => r.selectionStatus === "ai_suggested" || r.selectionStatus === "confirmed").map((r: ARRecord) => r.id)
      );
      setConfirmed(alreadySelected);
    } catch {
      toast.error("加载记录失败");
    } finally {
      setLoading(false);
    }
  }

  async function runSelection(useAI: boolean) {
    setRunning(true);
    try {
      const res = await fetch("/api/sample-selection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId, criteria, useAI }),
      });
      if (!res.ok) throw new Error((await res.json()).error);
      const data: SelectionResult = await res.json();
      setResult(data);
      setConfirmed(new Set(data.selections.map((s) => s.id)));
      await fetchRecords();
      toast.success(`选择了 ${data.selectedCount} 条记录，覆盖率 ${data.coverageRate}%`);
    } catch (err: any) {
      toast.error(err.message || "样本选择失败");
    } finally {
      setRunning(false);
    }
  }

  function toggleSelection(id: string) {
    setConfirmed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function confirmSelection() {
    if (confirmed.size === 0) {
      toast.error("请至少选择一条记录");
      return;
    }
    try {
      const res = await fetch("/api/confirmations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId,
          arRecordIds: Array.from(confirmed),
          type: "positive",
        }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      toast.success(`已创建 ${data.count} 份函证`);
      router.push(`/projects/${projectId}/confirmations`);
    } catch {
      toast.error("创建函证失败");
    }
  }

  if (loading) return <PageLoading />;

  const selectedRecords = records.filter((r) => confirmed.has(r.id));
  const selectedBalance = selectedRecords.reduce((s, r) => s + r.totalBalance, 0);
  const totalBalance = records.reduce((s, r) => s + r.totalBalance, 0);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">智能样本选择</h1>
      <p className="text-muted-foreground mb-6">配置选择标准，运行AI或规则引擎进行样本选择</p>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Criteria Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">选择标准</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>最低余额阈值</Label>
              <Input
                type="number"
                value={criteria.minBalance}
                onChange={(e) => setCriteria({ ...criteria, minBalance: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label>最大样本数</Label>
              <Input
                type="number"
                value={criteria.maxSamples}
                onChange={(e) => setCriteria({ ...criteria, maxSamples: Number(e.target.value) })}
              />
            </div>
            <div className="space-y-2">
              {[
                { key: "includeHighRisk", label: "包含高风险客户" },
                { key: "includeRelatedParties", label: "包含关联方" },
                { key: "includeLongAged", label: "包含长账龄" },
              ].map(({ key, label }) => (
                <label key={key} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={(criteria as any)[key]}
                    onChange={(e) => setCriteria({ ...criteria, [key]: e.target.checked })}
                    className="rounded"
                  />
                  {label}
                </label>
              ))}
            </div>

            <div className="space-y-2 pt-2">
              <Button className="w-full" onClick={() => runSelection(true)} disabled={running || records.length === 0}>
                {running ? <LoadingSpinner className="mr-2" /> : <Sparkles className="h-4 w-4 mr-2" />}
                AI智能选择
              </Button>
              <Button className="w-full" variant="outline" onClick={() => runSelection(false)} disabled={running || records.length === 0}>
                {running ? <LoadingSpinner className="mr-2" /> : <Brain className="h-4 w-4 mr-2" />}
                规则选择
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-4 pb-4">
                <p className="text-xs text-muted-foreground">已选择</p>
                <p className="text-2xl font-bold">{confirmed.size} / {records.length}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4 pb-4">
                <p className="text-xs text-muted-foreground">覆盖金额</p>
                <p className="text-lg font-bold">{formatAmount(selectedBalance)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4 pb-4">
                <p className="text-xs text-muted-foreground">覆盖率</p>
                <p className="text-2xl font-bold">
                  {totalBalance > 0 ? ((selectedBalance / totalBalance) * 100).toFixed(1) : "0"}%
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Records Table */}
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b bg-muted/50">
                      <th className="px-4 py-3 text-center w-10">
                        <input
                          type="checkbox"
                          checked={confirmed.size === records.length && records.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) setConfirmed(new Set(records.map((r) => r.id)));
                            else setConfirmed(new Set());
                          }}
                        />
                      </th>
                      <th className="px-4 py-3 text-left font-medium">客户</th>
                      <th className="px-4 py-3 text-right font-medium">余额</th>
                      <th className="px-4 py-3 text-center font-medium">风险</th>
                      <th className="px-4 py-3 text-left font-medium">选择原因</th>
                    </tr>
                  </thead>
                  <tbody>
                    {records.map((r) => (
                      <tr
                        key={r.id}
                        className={`border-b hover:bg-muted/30 cursor-pointer ${confirmed.has(r.id) ? "bg-blue-50" : ""}`}
                        onClick={() => toggleSelection(r.id)}
                      >
                        <td className="px-4 py-3 text-center">
                          <input type="checkbox" checked={confirmed.has(r.id)} readOnly />
                        </td>
                        <td className="px-4 py-3">
                          <span className="font-medium">{r.customerName}</span>
                          {r.isRelatedParty && <Badge variant="warning" className="ml-2 text-[10px]">关联方</Badge>}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">{formatAmount(r.totalBalance)}</td>
                        <td className="px-4 py-3 text-center">
                          <Badge variant={r.riskLevel === "high" ? "destructive" : r.riskLevel === "medium" ? "warning" : "secondary"}>
                            {r.riskLevel === "high" ? "高" : r.riskLevel === "medium" ? "中" : "低"}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-xs text-muted-foreground max-w-[200px] truncate">
                          {r.selectionReason || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button onClick={confirmSelection} disabled={confirmed.size === 0}>
              <Check className="h-4 w-4 mr-2" />
              确认选择并创建函证 ({confirmed.size}份)
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
