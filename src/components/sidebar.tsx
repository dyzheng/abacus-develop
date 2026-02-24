"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Upload,
  FileText,
  Brain,
  Mail,
  ClipboardCheck,
  GitCompare,
  BarChart3,
  FolderOpen,
  ChevronLeft,
} from "lucide-react";

const projectNavItems = [
  { href: "", label: "项目概览", icon: LayoutDashboard },
  { href: "/import", label: "导入数据", icon: Upload },
  { href: "/ar-records", label: "应收账款明细", icon: FileText },
  { href: "/sample-selection", label: "智能样本选择", icon: Brain },
  { href: "/confirmations", label: "函证管理", icon: Mail },
  { href: "/responses", label: "回函登记", icon: ClipboardCheck },
  { href: "/differences", label: "差异分析", icon: GitCompare },
  { href: "/report", label: "函证结果汇总", icon: BarChart3 },
];

export function Sidebar({ projectId, projectName }: { projectId?: string; projectName?: string }) {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-card min-h-screen flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-bold text-primary">函证智能体</h1>
        <p className="text-xs text-muted-foreground">AR Confirmation Agent</p>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {!projectId ? (
          <Link
            href="/projects"
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
              pathname === "/projects"
                ? "bg-primary text-primary-foreground"
                : "text-foreground hover:bg-accent"
            )}
          >
            <FolderOpen className="h-4 w-4" />
            审计项目列表
          </Link>
        ) : (
          <>
            <Link
              href="/projects"
              className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ChevronLeft className="h-4 w-4" />
              返回项目列表
            </Link>
            {projectName && (
              <div className="px-3 py-2 mb-2">
                <p className="text-xs text-muted-foreground">当前项目</p>
                <p className="text-sm font-semibold truncate">{projectName}</p>
              </div>
            )}
            {projectNavItems.map((item) => {
              const fullHref = `/projects/${projectId}${item.href}`;
              const isActive = item.href === ""
                ? pathname === fullHref
                : pathname.startsWith(fullHref);
              return (
                <Link
                  key={item.href || "overview"}
                  href={fullHref}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-foreground hover:bg-accent"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </>
        )}
      </nav>
    </aside>
  );
}
