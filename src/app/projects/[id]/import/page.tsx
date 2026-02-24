"use client";

import { useState, useRef } from "react";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, FileSpreadsheet, ArrowRight, Check, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { LoadingSpinner } from "@/components/ui/loading";

interface PreviewData {
  headers: string[];
  columnMapping: Record<string, string>;
  previewRows: Record<string, any>[];
  totalRows: number;
}

const FIELD_LABELS: Record<string, string> = {
  customerName: "客户名称",
  customerCode: "客户编码",
  totalBalance: "余额",
  within1Year: "1年以内",
  year1to2: "1-2年",
  year2to3: "2-3年",
  year3to4: "3-4年",
  year4to5: "4-5年",
  over5Years: "5年以上",
  isRelatedParty: "关联方",
};

export default function ImportPage() {
  const params = useParams();
  const projectId = params.id as string;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<"upload" | "mapping" | "preview" | "done">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ insertedCount: number } | null>(null);

  async function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", f);
      formData.append("projectId", projectId);
      formData.append("preview", "true");

      const res = await fetch("/api/import", { method: "POST", body: formData });
      if (!res.ok) throw new Error((await res.json()).error);
      const data: PreviewData = await res.json();
      setPreview(data);
      setMapping(data.columnMapping);
      setStep("mapping");
    } catch (err: any) {
      toast.error(err.message || "文件解析失败");
    } finally {
      setLoading(false);
    }
  }

  function handleMappingChange(header: string, field: string) {
    setMapping((prev) => {
      const next = { ...prev };
      if (field === "") {
        delete next[header];
      } else {
        next[header] = field;
      }
      return next;
    });
  }

  async function handleImport() {
    if (!file) return;
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("projectId", projectId);
      formData.append("columnMapping", JSON.stringify(mapping));

      const res = await fetch("/api/import", { method: "POST", body: formData });
      if (!res.ok) throw new Error((await res.json()).error);
      const data = await res.json();
      setResult(data);
      setStep("done");
      toast.success(`成功导入 ${data.insertedCount} 条记录`);
    } catch (err: any) {
      toast.error(err.message || "导入失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">导入数据</h1>
      <p className="text-muted-foreground mb-6">上传Excel或CSV文件，导入应收账款明细数据</p>

      {/* Steps indicator */}
      <div className="flex items-center gap-2 mb-8">
        {[
          { key: "upload", label: "上传文件" },
          { key: "mapping", label: "列映射" },
          { key: "preview", label: "预览确认" },
          { key: "done", label: "完成" },
        ].map((s, i) => (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && <ArrowRight className="h-4 w-4 text-muted-foreground" />}
            <Badge variant={step === s.key ? "default" : s.key === "done" && step === "done" ? "success" : "secondary"}>
              {s.label}
            </Badge>
          </div>
        ))}
      </div>

      {step === "upload" && (
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center">
              <FileSpreadsheet className="h-16 w-16 text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">选择Excel或CSV文件</p>
              <p className="text-sm text-muted-foreground mb-6">支持 .xlsx, .xls, .csv 格式</p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".xlsx,.xls,.csv"
                className="hidden"
                onChange={handleFileSelect}
              />
              <Button onClick={() => fileInputRef.current?.click()} disabled={loading}>
                {loading ? <LoadingSpinner className="mr-2" /> : <Upload className="h-4 w-4 mr-2" />}
                {loading ? "解析中..." : "选择文件"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {step === "mapping" && preview && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>列映射配置</CardTitle>
              <CardDescription>
                系统已自动识别列映射关系，请检查并调整。文件共 {preview.totalRows} 行数据。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {preview.headers.map((header) => (
                  <div key={header} className="flex items-center gap-4">
                    <span className="w-40 text-sm font-medium truncate" title={header}>{header}</span>
                    <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    <select
                      className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                      value={mapping[header] || ""}
                      onChange={(e) => handleMappingChange(header, e.target.value)}
                    >
                      <option value="">-- 不映射 --</option>
                      {Object.entries(FIELD_LABELS).map(([field, label]) => (
                        <option key={field} value={field}>{label}</option>
                      ))}
                    </select>
                    {mapping[header] && <Check className="h-4 w-4 text-green-600 flex-shrink-0" />}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Preview table */}
          <Card>
            <CardHeader>
              <CardTitle>数据预览</CardTitle>
              <CardDescription>前5行数据</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      {preview.headers.map((h) => (
                        <th key={h} className="px-2 py-2 text-left font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.previewRows.map((row, i) => (
                      <tr key={i} className="border-b">
                        {preview.headers.map((h) => (
                          <td key={h} className="px-2 py-2">{row[h]?.toString() || ""}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-between">
            <Button variant="outline" onClick={() => { setStep("upload"); setFile(null); setPreview(null); }}>
              重新上传
            </Button>
            <Button onClick={handleImport} disabled={loading || !mapping["客户名称"] && !Object.values(mapping).includes("customerName")}>
              {loading ? <LoadingSpinner className="mr-2" /> : null}
              {loading ? "导入中..." : `确认导入 ${preview.totalRows} 条记录`}
            </Button>
          </div>
        </div>
      )}

      {step === "done" && result && (
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center">
              <Check className="h-16 w-16 text-green-600 mb-4" />
              <p className="text-lg font-medium mb-2">导入完成</p>
              <p className="text-muted-foreground mb-6">
                成功导入 {result.insertedCount} 条应收账款记录
              </p>
              <div className="flex gap-3">
                <Button variant="outline" onClick={() => { setStep("upload"); setFile(null); setPreview(null); setResult(null); }}>
                  继续导入
                </Button>
                <Button onClick={() => window.location.href = `/projects/${projectId}/ar-records`}>
                  查看明细
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
