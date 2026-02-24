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
  { href: "", label: "项目概览", icon: LayoutDashboard, step: 1 },
  { href: "/import", label: "导入数据", icon: Upload, step: 2 },
  { href: "/ar-records", label: "应收账款明细", icon: FileText, step: 3 },
  { href: "/sample-selection", label: "智能样本选择", icon: Brain, step: 4 },
  { href: "/confirmations", label: "函证管理", icon: Mail, step: 5 },
  { href: "/responses", label: "回函登记", icon: ClipboardCheck, step: 6 },
  { href: "/differences", label: "差异分析", icon: GitCompare, step: 7 },
  { href: "/report", label: "函证结果汇总", icon: BarChart3, step: 8 },
];

export function Sidebar({ projectId, projectName }: { projectId?: string; projectName?: string }) {
  const pathname = usePathname();

  return (
    <aside className="w-64 min-h-screen flex flex-col sidebar-ink no-print">
      {/* Brand */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-md bg-seal flex items-center justify-center">
            <span className="text-white text-xs font-bold font-display">函</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-wide font-display">函证智能体</h1>
            <p className="text-[10px] text-sidebar-muted tracking-wider">AUDIT CONFIRMATION</p>
          </div>
        </div>
      </div>

      <div className="mx-4 border-t border-white/10" />

      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {!projectId ? (
          <Link
            href="/projects"
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
              pathname === "/projects"
                ? "bg-seal text-white shadow-lg shadow-seal/20"
                : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-white"
            )}
          >
            <FolderOpen className="h-4 w-4" />
            审计项目列表
          </Link>
        ) : (
          <>
            <Link
              href="/projects"
              className="flex items-center gap-2 px-3 py-2 text-xs text-sidebar-muted hover:text-white transition-colors duration-200 mb-1"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              返回项目列表
            </Link>

            {projectName && (
              <div className="px-3 py-3 mb-2 rounded-lg bg-sidebar-accent">
                <p className="text-[10px] uppercase tracking-widest text-sidebar-muted mb-1">当前项目</p>
                <p className="text-sm font-semibold text-white truncate">{projectName}</p>
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
                    "group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200",
                    isActive
                      ? "bg-seal text-white font-medium shadow-lg shadow-seal/20"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-white"
                  )}
                >
                  <item.icon className={cn(
                    "h-4 w-4 transition-transform duration-200",
                    !isActive && "group-hover:scale-110"
                  )} />
                  <span className="flex-1">{item.label}</span>
                  {isActive && (
                    <div className="w-1.5 h-1.5 rounded-full bg-white/80" />
                  )}
                </Link>
              );
            })}
          </>
        )}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/10">
        <p className="text-[10px] text-sidebar-muted">审计智能体 v1.0</p>
      </div>
    </aside>
  );
}
