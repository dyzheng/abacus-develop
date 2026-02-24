"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageLoading } from "@/components/ui/loading";
import { formatAmount } from "@/lib/utils";
import {
  FileText,
  Send,
  MailCheck,
  AlertTriangle,
  ArrowLeft,
  TrendingUp,
  CheckCircle2,
  Clock,
  ShieldAlert,
  BarChart3,
  PieChart as PieChartIcon,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface ReportData {
  project: {
    name: string;
    clientCompany: string;
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

interface OverdueConfirmation {
  id: string;
  confirmationNumber: string;
  customerName: string;
  dueDate: string;
  status: string;
  daysPastDue: number;
}

const STATUS_LABEL_MAP: Record<string, string> = {
  draft: "草稿",
  generated: "已生成",
  sent: "已发出",
  received: "已收回",
  reconciled: "已核对",
  alternative_procedure: "替代程序",
};

const STATUS_COLOR_MAP: Record<string, string> = {
  draft: "#94a3b8",
  generated: "#60a5fa",
  sent: "#fbbf24",
  received: "#34d399",
  reconciled: "#a78bfa",
  alternative_procedure: "#f87171",
};

const AGING_LABELS = [
  { key: "within1Year", label: "1年以内" },
  { key: "year1to2", label: "1-2年" },
  { key: "year2to3", label: "2-3年" },
  { key: "year3to4", label: "3-4年" },
  { key: "year4to5", label: "4-5年" },
  { key: "over5Years", label: "5年以上" },
];

export default function ProjectDashboard() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [agingData, setAgingData] = useState<{ name: string; amount: number }[]>([]);
  const [overdueList, setOverdueList] = useState<OverdueConfirmation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        // Fetch the main report
        const reportRes = await fetch(`/api/report?projectId=${id}`);
        if (!reportRes.ok) {
          const errBody = await reportRes.json().catch(() => ({}));
          throw new Error(errBody.error || "加载项目报告失败");
        }
        const report: ReportData = await reportRes.json();
        setReportData(report);

        // Fetch AR records for aging analysis
        const arRes = await fetch(`/api/ar-records?projectId=${id}`);
        if (arRes.ok) {
          const arRecords = await arRes.json();
          const agingAgg: Record<string, number> = {};
          AGING_LABELS.forEach((a) => (agingAgg[a.key] = 0));

          if (Array.isArray(arRecords)) {
            for (const rec of arRecords) {
              agingAgg["within1Year"] += rec.within1Year || 0;
              agingAgg["year1to2"] += rec.year1to2 || 0;
              agingAgg["year2to3"] += rec.year2to3 || 0;
              agingAgg["year3to4"] += rec.year3to4 || 0;
              agingAgg["year4to5"] += rec.year4to5 || 0;
              agingAgg["over5Years"] += rec.over5Years || 0;
            }
          }

          const aging = AGING_LABELS.map((a) => ({
            name: a.label,
            amount: agingAgg[a.key],
          }));
          setAgingData(aging);
        }

        // Fetch confirmations for overdue alerts
        const confRes = await fetch(`/api/confirmations?projectId=${id}`);
        if (confRes.ok) {
          const confirmationsList = await confRes.json();
          const today = new Date();
          today.setHours(0, 0, 0, 0);

          const overdue: OverdueConfirmation[] = [];
          const items = Array.isArray(confirmationsList)
            ? confirmationsList
            : confirmationsList.confirmations || [];

          for (const conf of items) {
            if (
              conf.dueDate &&
              conf.status !== "received" &&
              conf.status !== "reconciled"
            ) {
              const due = new Date(conf.dueDate);
              due.setHours(0, 0, 0, 0);
              if (due < today) {
                const diffTime = today.getTime() - due.getTime();
                const daysPastDue = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
                overdue.push({
                  id: conf.id,
                  confirmationNumber: conf.confirmationNumber,
                  customerName: conf.customerName || conf.arRecord?.customerName || "-",
                  dueDate: conf.dueDate,
                  status: conf.status,
                  daysPastDue,
                });
              }
            }
          }

          overdue.sort((a, b) => b.daysPastDue - a.daysPastDue);
          setOverdueList(overdue);
        }
      } catch (err: any) {
        setError(err.message || "加载数据失败");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [id]);

  if (loading) {
    return <PageLoading />;
  }

  if (error || !reportData) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-lg text-muted-foreground">{error || "数据加载失败"}</p>
        <button
          className="text-primary underline text-sm"
          onClick={() => router.push("/projects")}
        >
          返回项目列表
        </button>
      </div>
    );
  }

  const { project, summary } = reportData;

  // Build pie chart data from statusCounts
  const statusPieData = Object.entries(summary.statusCounts)
    .filter(([, count]) => count > 0)
    .map(([status, count]) => ({
      name: STATUS_LABEL_MAP[status] || status,
      value: count,
      color: STATUS_COLOR_MAP[status] || "#8884d8",
    }));

  // Response distribution for a secondary metric view
  const responsePieData = [
    { name: "相符", value: summary.agreeCount, color: "#34d399" },
    { name: "不符", value: summary.disagreeCount, color: "#f87171" },
    { name: "未回函", value: summary.noResponseCount, color: "#94a3b8" },
  ].filter((d) => d.value > 0);

  const unresolvedDifferences = summary.differenceCount - summary.resolvedCount;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => router.push("/projects")}
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          返回
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{project.name}</h1>
          <p className="text-sm text-muted-foreground">
            {project.clientCompany} | 基准日: {project.balanceDate}
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 应收账款总数 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">应收账款总数</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.totalARCount}</div>
            <p className="text-xs text-muted-foreground mt-1">
              余额合计 {formatAmount(summary.totalARBalance)}
            </p>
          </CardContent>
        </Card>

        {/* 函证数量 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">函证数量</CardTitle>
            <Send className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.confirmationCount}</div>
            <p className="text-xs text-muted-foreground mt-1">
              覆盖率{" "}
              <span className="font-semibold text-foreground">
                {summary.coverageRate}%
              </span>{" "}
              | {formatAmount(summary.confirmedBalance)}
            </p>
          </CardContent>
        </Card>

        {/* 回函数量 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">回函数量</CardTitle>
            <MailCheck className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.responseCount}</div>
            <p className="text-xs text-muted-foreground mt-1">
              回函率{" "}
              <span className="font-semibold text-foreground">
                {summary.responseRate}%
              </span>{" "}
              | 相符 {summary.agreeCount} 笔
            </p>
          </CardContent>
        </Card>

        {/* 差异数量 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">差异数量</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.differenceCount}</div>
            <p className="text-xs text-muted-foreground mt-1">
              差异金额 {formatAmount(summary.totalDifferenceAmount)}
              {unresolvedDifferences > 0 && (
                <Badge variant="warning" className="ml-2 text-[10px]">
                  {unresolvedDifferences} 待解决
                </Badge>
              )}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Progress Overview Bar */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            流程进度概览
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ProgressItem
              label="函证覆盖率"
              value={parseFloat(summary.coverageRate)}
              color="bg-blue-500"
            />
            <ProgressItem
              label="回函率"
              value={parseFloat(summary.responseRate)}
              color="bg-green-500"
            />
            <ProgressItem
              label="相符率"
              value={
                summary.responseCount > 0
                  ? (summary.agreeCount / summary.responseCount) * 100
                  : 0
              }
              color="bg-emerald-500"
            />
            <ProgressItem
              label="差异解决率"
              value={
                summary.differenceCount > 0
                  ? (summary.resolvedCount / summary.differenceCount) * 100
                  : 0
              }
              color="bg-purple-500"
            />
          </div>
        </CardContent>
      </Card>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confirmation Status Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <PieChartIcon className="h-4 w-4" />
              函证状态分布
            </CardTitle>
            <CardDescription>各状态函证数量占比</CardDescription>
          </CardHeader>
          <CardContent>
            {statusPieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={statusPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }: any) =>
                      `${name || ''} ${((percent || 0) * 100).toFixed(0)}%`
                    }
                  >
                    {statusPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: any, name: any) => [
                      `${value} 笔`,
                      name,
                    ]}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="暂无函证数据" />
            )}
          </CardContent>
        </Card>

        {/* Aging Analysis Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              账龄分析
            </CardTitle>
            <CardDescription>应收账款账龄结构分布</CardDescription>
          </CardHeader>
          <CardContent>
            {agingData.some((d) => d.amount > 0) ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={agingData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    tickFormatter={(v) =>
                      v >= 10000 ? `${(v / 10000).toFixed(0)}万` : `${v}`
                    }
                  />
                  <Tooltip
                    formatter={(value: any) => [formatAmount(value), "金额"]}
                  />
                  <Bar
                    dataKey="amount"
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={48}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="暂无账龄数据" />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Response Distribution + Difference Summary Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Response Distribution Pie */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4" />
              回函结果分布
            </CardTitle>
            <CardDescription>回函相符与不符占比</CardDescription>
          </CardHeader>
          <CardContent>
            {responsePieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={responsePieData}
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}笔`}
                  >
                    {responsePieData.map((entry, index) => (
                      <Cell key={`resp-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: any, name: any) => [
                      `${value} 笔`,
                      name,
                    ]}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="暂无回函数据" />
            )}
          </CardContent>
        </Card>

        {/* Difference Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <ShieldAlert className="h-4 w-4" />
              差异汇总
            </CardTitle>
            <CardDescription>差异调查与解决情况</CardDescription>
          </CardHeader>
          <CardContent>
            {summary.differenceCount > 0 ? (
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-3xl font-bold text-orange-500">
                      {summary.differenceCount}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">差异总数</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-green-500">
                      {summary.resolvedCount}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">已解决</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-red-500">
                      {unresolvedDifferences}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">待解决</div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-muted-foreground">解决进度</span>
                    <span className="font-medium">
                      {summary.differenceCount > 0
                        ? (
                            (summary.resolvedCount / summary.differenceCount) *
                            100
                          ).toFixed(1)
                        : 0}
                      %
                    </span>
                  </div>
                  <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-green-500 rounded-full transition-all duration-500"
                      style={{
                        width: `${
                          summary.differenceCount > 0
                            ? (summary.resolvedCount / summary.differenceCount) * 100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>

                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground">差异金额合计</div>
                  <div className="text-lg font-semibold mt-1">
                    {formatAmount(summary.totalDifferenceAmount)}
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState message="暂无差异记录" />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Overdue Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="h-4 w-4" />
            逾期函证提醒
            {overdueList.length > 0 && (
              <Badge variant="destructive" className="ml-1">
                {overdueList.length}
              </Badge>
            )}
          </CardTitle>
          <CardDescription>已超过回函截止日期但尚未收到回函的函证</CardDescription>
        </CardHeader>
        <CardContent>
          {overdueList.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left">
                    <th className="pb-2 pr-4 font-medium text-muted-foreground">
                      函证编号
                    </th>
                    <th className="pb-2 pr-4 font-medium text-muted-foreground">
                      客户名称
                    </th>
                    <th className="pb-2 pr-4 font-medium text-muted-foreground">
                      截止日期
                    </th>
                    <th className="pb-2 pr-4 font-medium text-muted-foreground">
                      逾期天数
                    </th>
                    <th className="pb-2 font-medium text-muted-foreground">当前状态</th>
                  </tr>
                </thead>
                <tbody>
                  {overdueList.map((item) => (
                    <tr
                      key={item.id}
                      className="border-b last:border-0 hover:bg-muted/50 transition-colors"
                    >
                      <td className="py-3 pr-4 font-mono text-xs">
                        {item.confirmationNumber}
                      </td>
                      <td className="py-3 pr-4">{item.customerName}</td>
                      <td className="py-3 pr-4">{item.dueDate}</td>
                      <td className="py-3 pr-4">
                        <Badge
                          variant={item.daysPastDue > 14 ? "destructive" : "warning"}
                        >
                          {item.daysPastDue} 天
                        </Badge>
                      </td>
                      <td className="py-3">
                        <Badge variant="outline">
                          {STATUS_LABEL_MAP[item.status] || item.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle2 className="h-8 w-8 mx-auto mb-2 text-green-500" />
              <p className="text-sm">所有函证均在截止日期内，暂无逾期提醒</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/* ---------- Helper Components ---------- */

function ProgressItem({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const clamped = Math.min(Math.max(value, 0), 100);
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold">{clamped.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-[200px] text-muted-foreground text-sm">
      {message}
    </div>
  );
}
