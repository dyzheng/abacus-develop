"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { PageLoading } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import { Search, AlertTriangle, ArrowUpDown } from "lucide-react";
import { toast } from "sonner";

interface ARRecord {
  id: string;
  customerName: string;
  customerCode: string | null;
  totalBalance: number;
  within1Year: number | null;
  year1to2: number | null;
  year2to3: number | null;
  year3to4: number | null;
  year4to5: number | null;
  over5Years: number | null;
  riskLevel: string | null;
  isRelatedParty: boolean | null;
  selectionStatus: string | null;
  selectionReason: string | null;
}

type SortField = "totalBalance" | "customerName" | "riskLevel";

export default function ARRecordsPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [records, setRecords] = useState<ARRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [selectionFilter, setSelectionFilter] = useState("");
  const [sortField, setSortField] = useState<SortField>("totalBalance");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    fetchRecords();
  }, [riskFilter, selectionFilter]);

  async function fetchRecords() {
    try {
      const params = new URLSearchParams({ projectId });
      if (riskFilter) params.set("riskLevel", riskFilter);
      if (selectionFilter) params.set("selectionStatus", selectionFilter);
      const res = await fetch(`/api/ar-records?${params}`);
      const data = await res.json();
      setRecords(data);
    } catch {
      toast.error("加载记录失败");
    } finally {
      setLoading(false);
    }
  }

  const filtered = records
    .filter((r) => !search || r.customerName.includes(search) || r.customerCode?.includes(search))
    .sort((a, b) => {
      let cmp = 0;
      if (sortField === "totalBalance") cmp = a.totalBalance - b.totalBalance;
      else if (sortField === "customerName") cmp = a.customerName.localeCompare(b.customerName);
      else if (sortField === "riskLevel") {
        const order = { high: 3, medium: 2, low: 1 };
        cmp = (order[a.riskLevel as keyof typeof order] || 0) - (order[b.riskLevel as keyof typeof order] || 0);
      }
      return sortDir === "desc" ? -cmp : cmp;
    });

  const totalBalance = filtered.reduce((s, r) => s + r.totalBalance, 0);

  const riskVariant = (level: string | null) => {
    if (level === "high") return "destructive" as const;
    if (level === "medium") return "warning" as const;
    return "secondary" as const;
  };

  const riskLabel = (level: string | null) => {
    if (level === "high") return "高风险";
    if (level === "medium") return "中风险";
    return "低风险";
  };

  const selectionLabel = (status: string | null) => {
    const map: Record<string, string> = {
      unselected: "未选择",
      ai_suggested: "AI建议",
      confirmed: "已确认",
      excluded: "已排除",
    };
    return map[status || ""] || status;
  };

  function toggleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  }

  if (loading) return <PageLoading />;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">应收账款明细</h1>
      <p className="text-muted-foreground mb-6">
        共 {filtered.length} 条记录，余额合计 {formatAmount(totalBalance)}
      </p>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="搜索客户名称或编码..."
            className="pl-9"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Select
          options={[
            { value: "high", label: "高风险" },
            { value: "medium", label: "中风险" },
            { value: "low", label: "低风险" },
          ]}
          placeholder="全部风险等级"
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
          className="w-40"
        />
        <Select
          options={[
            { value: "unselected", label: "未选择" },
            { value: "ai_suggested", label: "AI建议" },
            { value: "confirmed", label: "已确认" },
            { value: "excluded", label: "已排除" },
          ]}
          placeholder="全部选择状态"
          value={selectionFilter}
          onChange={(e) => setSelectionFilter(e.target.value)}
          className="w-40"
        />
      </div>

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium cursor-pointer" onClick={() => toggleSort("customerName")}>
                    <div className="flex items-center gap-1">客户名称 <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 text-right font-medium cursor-pointer" onClick={() => toggleSort("totalBalance")}>
                    <div className="flex items-center justify-end gap-1">余额 <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 text-right font-medium">1年以内</th>
                  <th className="px-4 py-3 text-right font-medium">1-2年</th>
                  <th className="px-4 py-3 text-right font-medium">2-3年</th>
                  <th className="px-4 py-3 text-right font-medium">3年以上</th>
                  <th className="px-4 py-3 text-center font-medium cursor-pointer" onClick={() => toggleSort("riskLevel")}>
                    <div className="flex items-center justify-center gap-1">风险 <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 text-center font-medium">选择状态</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id} className="border-b hover:bg-muted/30">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{r.customerName}</span>
                        {r.isRelatedParty && (
                          <Badge variant="warning" className="text-[10px] px-1">关联方</Badge>
                        )}
                      </div>
                      {r.customerCode && (
                        <span className="text-xs text-muted-foreground">{r.customerCode}</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{formatAmount(r.totalBalance)}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">{r.within1Year ? formatAmount(r.within1Year) : "-"}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">{r.year1to2 ? formatAmount(r.year1to2) : "-"}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">{r.year2to3 ? formatAmount(r.year2to3) : "-"}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">
                      {(r.year3to4 || 0) + (r.year4to5 || 0) + (r.over5Years || 0) > 0
                        ? formatAmount((r.year3to4 || 0) + (r.year4to5 || 0) + (r.over5Years || 0))
                        : "-"}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Badge variant={riskVariant(r.riskLevel)}>{riskLabel(r.riskLevel)}</Badge>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Badge variant={r.selectionStatus === "confirmed" ? "success" : r.selectionStatus === "ai_suggested" ? "info" : "secondary"}>
                        {selectionLabel(r.selectionStatus)}
                      </Badge>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-12 text-center text-muted-foreground">
                      暂无记录，请先导入数据
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
