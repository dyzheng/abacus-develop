"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { PageLoading, LoadingSpinner } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import { Search, ArrowUpDown, Brain, Pencil, MapPin, ShieldCheck } from "lucide-react";
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
  contactPerson: string | null;
  contactPhone: string | null;
  contactEmail: string | null;
  address: string | null;
  city: string | null;
  province: string | null;
  postalCode: string | null;
  verificationStatus: string | null;
  verificationDetail: string | null;
  verificationScore: number | null;
}

type SortField = "totalBalance" | "customerName" | "riskLevel";

const VERIFICATION_MAP: Record<string, { label: string; variant: "secondary" | "success" | "warning" | "destructive" }> = {
  unverified: { label: "未核验", variant: "secondary" },
  verified: { label: "已核验", variant: "success" },
  suspicious: { label: "可疑", variant: "warning" },
  flagged: { label: "异常", variant: "destructive" },
};

export default function ARRecordsPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [records, setRecords] = useState<ARRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [selectionFilter, setSelectionFilter] = useState("");
  const [verificationFilter, setVerificationFilter] = useState("");
  const [sortField, setSortField] = useState<SortField>("totalBalance");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // Edit dialog
  const [editRecord, setEditRecord] = useState<ARRecord | null>(null);
  const [editForm, setEditForm] = useState({ contactPerson: "", contactPhone: "", contactEmail: "", address: "", city: "", province: "", postalCode: "" });
  const [saving, setSaving] = useState(false);

  // Verification detail dialog
  const [verifyDetailRecord, setVerifyDetailRecord] = useState<ARRecord | null>(null);

  // AI verification
  const [verifying, setVerifying] = useState(false);

  const fetchRecords = useCallback(async (signal?: AbortSignal) => {
    try {
      const p = new URLSearchParams({ projectId });
      if (riskFilter) p.set("riskLevel", riskFilter);
      if (selectionFilter) p.set("selectionStatus", selectionFilter);
      if (verificationFilter) p.set("verificationStatus", verificationFilter);
      const res = await fetch(`/api/ar-records?${p}`, { signal });
      const data = await res.json();
      setRecords(data);
    } catch (err: any) {
      if (!err?.name?.includes("Abort")) toast.error("加载记录失败");
    } finally {
      setLoading(false);
    }
  }, [projectId, riskFilter, selectionFilter, verificationFilter]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetchRecords(controller.signal);
    return () => controller.abort();
  }, [fetchRecords]);

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

  function openEdit(r: ARRecord) {
    setEditRecord(r);
    setEditForm({
      contactPerson: r.contactPerson || "",
      contactPhone: r.contactPhone || "",
      contactEmail: r.contactEmail || "",
      address: r.address || "",
      city: r.city || "",
      province: r.province || "",
      postalCode: r.postalCode || "",
    });
  }

  async function saveContact() {
    if (!editRecord) return;
    setSaving(true);
    try {
      const res = await fetch("/api/ar-records", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: [editRecord.id], ...editForm }),
      });
      if (!res.ok) throw new Error();
      toast.success("客户信息已更新");
      setEditRecord(null);
      await fetchRecords();
    } catch {
      toast.error("保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function runVerification() {
    setVerifying(true);
    try {
      const res = await fetch("/api/verify-customers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      toast.success(`已核验 ${data.total} 个客户`);
      await fetchRecords();
    } catch {
      toast.error("核验失败");
    } finally {
      setVerifying(false);
    }
  }

  function hasAddress(r: ARRecord) {
    return !!(r.address || r.city || r.contactPerson);
  }

  function parseVerificationDetail(r: ARRecord) {
    if (!r.verificationDetail) return null;
    try {
      return JSON.parse(r.verificationDetail);
    } catch {
      return { detail: r.verificationDetail };
    }
  }

  async function adoptSuggestedAddress(r: ARRecord) {
    const detail = parseVerificationDetail(r);
    if (!detail?.suggestedRegion) return;
    setSaving(true);
    try {
      const res = await fetch("/api/ar-records", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: [r.id], province: detail.suggestedRegion }),
      });
      if (!res.ok) throw new Error();
      toast.success("已采纳建议地址");
      setVerifyDetailRecord(null);
      await fetchRecords();
    } catch {
      toast.error("更新失败");
    } finally {
      setSaving(false);
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
        <Select
          options={[
            { value: "unverified", label: "未核验" },
            { value: "verified", label: "已核验" },
            { value: "suspicious", label: "可疑" },
            { value: "flagged", label: "异常" },
          ]}
          placeholder="全部核验状态"
          value={verificationFilter}
          onChange={(e) => setVerificationFilter(e.target.value)}
          className="w-40"
        />
        <Button onClick={runVerification} disabled={verifying} variant="outline">
          {verifying ? <LoadingSpinner className="mr-2" /> : <Brain className="h-4 w-4 mr-2" />}
          {verifying ? "核验中..." : "AI核验客户"}
        </Button>
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
                  <th className="px-4 py-3 text-center font-medium">核验</th>
                  <th className="px-4 py-3 text-center font-medium">选择状态</th>
                  <th className="px-4 py-3 text-center font-medium">操作</th>
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
                        {hasAddress(r) && (
                          <MapPin className="h-3 w-3 text-green-600" />
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
                      <button onClick={() => setVerifyDetailRecord(r)} className="cursor-pointer">
                        <Badge variant={VERIFICATION_MAP[r.verificationStatus || "unverified"]?.variant || "secondary"}>
                          {VERIFICATION_MAP[r.verificationStatus || "unverified"]?.label || "未核验"}
                        </Badge>
                      </button>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Badge variant={r.selectionStatus === "confirmed" ? "success" : r.selectionStatus === "ai_suggested" ? "info" : "secondary"}>
                        {selectionLabel(r.selectionStatus)}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Button size="sm" variant="ghost" onClick={() => openEdit(r)}>
                        <Pencil className="h-3 w-3" />
                      </Button>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={10} className="px-4 py-12 text-center text-muted-foreground">
                      暂无记录，请先导入数据
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Edit Contact Dialog */}
      <Dialog open={!!editRecord} onOpenChange={(open) => !open && setEditRecord(null)}>
        <DialogContent onClose={() => setEditRecord(null)} className="max-w-md">
          <DialogHeader>
            <DialogTitle>编辑客户信息 - {editRecord?.customerName}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <Label>联系人</Label>
              <Input value={editForm.contactPerson} onChange={(e) => setEditForm({ ...editForm, contactPerson: e.target.value })} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>联系电话</Label>
                <Input value={editForm.contactPhone} onChange={(e) => setEditForm({ ...editForm, contactPhone: e.target.value })} />
              </div>
              <div>
                <Label>邮箱</Label>
                <Input value={editForm.contactEmail} onChange={(e) => setEditForm({ ...editForm, contactEmail: e.target.value })} />
              </div>
            </div>
            <div>
              <Label>地址</Label>
              <Input value={editForm.address} onChange={(e) => setEditForm({ ...editForm, address: e.target.value })} />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <Label>省份</Label>
                <Input value={editForm.province} onChange={(e) => setEditForm({ ...editForm, province: e.target.value })} />
              </div>
              <div>
                <Label>城市</Label>
                <Input value={editForm.city} onChange={(e) => setEditForm({ ...editForm, city: e.target.value })} />
              </div>
              <div>
                <Label>邮编</Label>
                <Input value={editForm.postalCode} onChange={(e) => setEditForm({ ...editForm, postalCode: e.target.value })} />
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setEditRecord(null)}>取消</Button>
              <Button onClick={saveContact} disabled={saving}>
                {saving ? <LoadingSpinner className="mr-2" /> : null}
                保存
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Verification Detail Dialog */}
      <Dialog open={!!verifyDetailRecord} onOpenChange={(open) => !open && setVerifyDetailRecord(null)}>
        <DialogContent onClose={() => setVerifyDetailRecord(null)} className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-5 w-5" />
                核验详情 - {verifyDetailRecord?.customerName}
              </div>
            </DialogTitle>
          </DialogHeader>
          {verifyDetailRecord && (() => {
            const vs = verifyDetailRecord.verificationStatus || "unverified";
            const detail = parseVerificationDetail(verifyDetailRecord);
            const score = verifyDetailRecord.verificationScore;
            return (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Badge variant={VERIFICATION_MAP[vs]?.variant || "secondary"} className="text-sm px-3 py-1">
                    {VERIFICATION_MAP[vs]?.label || "未核验"}
                  </Badge>
                  {score !== null && score !== undefined && (
                    <span className="text-sm text-muted-foreground">置信度: {(score * 100).toFixed(0)}%</span>
                  )}
                </div>
                {detail ? (
                  <div className="space-y-2 text-sm">
                    {detail.detail && <p>{detail.detail}</p>}
                    {detail.inferredBusinessType && (
                      <p><span className="text-muted-foreground">推断行业：</span>{detail.inferredBusinessType}</p>
                    )}
                    {detail.suggestedRegion && (
                      <div className="flex items-center justify-between">
                        <p><span className="text-muted-foreground">推测地区：</span>{detail.suggestedRegion}</p>
                        {!verifyDetailRecord.province && (
                          <Button size="sm" variant="outline" onClick={() => adoptSuggestedAddress(verifyDetailRecord)} disabled={saving}>
                            采纳建议地址
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">暂无核验详情，请先执行AI核验</p>
                )}
              </div>
            );
          })()}
        </DialogContent>
      </Dialog>
    </div>
  );
}
