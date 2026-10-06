'use client';

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { useAuth } from "../components/auth/auth-provider";
import { LoginForm } from "../components/auth/login-form";
import { AuthProvider } from "../components/auth/auth-provider";
import { DashboardView } from "../components/views/dashboard-view";
import { ScansView } from "../components/views/scans-view";
import { FindingsView } from "../components/views/findings-view";
import { SettingsView } from "../components/views/settings-view";
import { ToolsView } from "../components/views/tools-view";
import { ReportHistoryView } from "../components/views/report-history-view";
import { FindingDetailView } from "../components/views/finding-detail-view";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import {
  LayoutDashboard,
  Radar,
  Bug,
  Wrench,
  Settings as SettingsIcon,
  LogOut,
  ShieldCheck,
  Loader2,
  AlertOctagon,
  History,
  Sun,
  Moon,
  ArrowLeft,
} from "lucide-react";
import { getExecutions, abortScan as cancelScan, type ToolExecution } from "../lib/api";
import { useToast } from "../hooks/use-toast";

type ViewName = "dashboard" | "scans" | "findings" | "tools" | "settings" | "reports";

interface SelectedFinding {
  id: string;
  // Store the full finding object so FindingDetailView v1 can render
  // without needing GET /api/findings/{id} (not yet implemented).
  // We pass via props from report-history-view.
  data: any;
}

const NAV_ITEMS: { name: ViewName; label: string; icon: React.ReactNode }[] = [
  { name: "dashboard", label: "Dashboard", icon: <LayoutDashboard className="w-4 h-4" /> },
  { name: "scans", label: "Scans", icon: <Radar className="w-4 h-4" /> },
  { name: "reports", label: "Reports", icon: <History className="w-4 h-4" /> },
  { name: "findings", label: "Findings", icon: <Bug className="w-4 h-4" /> },
  { name: "tools", label: "Tools", icon: <Wrench className="w-4 h-4" /> },
  { name: "settings", label: "Settings", icon: <SettingsIcon className="w-4 h-4" /> },
];

export default function Home() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  const isDark = theme === "dark";
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      className="text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
    </Button>
  );
}

function AppShell() {
  const { user, loading, isAuthenticated, logout } = useAuth();
  const [view, setView] = useState<ViewName>("dashboard");
  const [panicOpen, setPanicOpen] = useState(false);
  const [runningExecutions, setRunningExecutions] = useState<ToolExecution[]>([]);
  // W19-FIX4 Phase L v1: when user clicks a finding in report-history-view,
  // store the finding object + switch to "finding detail" pseudo-view.
  const [selectedFinding, setSelectedFinding] = useState<SelectedFinding | null>(null);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
        <Loader2 className="w-8 h-8 animate-spin text-zinc-600 dark:text-zinc-400" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginForm />;
  }

  // If a finding is selected, show finding detail (overrides current view)
  if (selectedFinding) {
    return (
      <div className="min-h-screen flex flex-col bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
        <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
          <div className="flex items-center justify-between px-4 h-14">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedFinding(null)}
                className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                Back to Reports
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <Button variant="ghost" size="sm" onClick={logout} className="text-zinc-400 hover:text-zinc-100">
                <LogOut className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </header>
        {/* Note: v1 FindingDetailView needs the finding object in scope.
            We pass selectedFinding.id + a back callback. The detail view
            will be enhanced in Phase L v2 to fetch by id when backend
            gets GET /api/findings/{id} endpoint. For v1, the report-history-view
            passes the finding object via window.history state. */}
        <FindingDetailView
          findingId={selectedFinding.id}
          findingData={selectedFinding.data}
          onBack={() => setSelectedFinding(null)}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      {/* Top bar */}
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-zinc-200 dark:bg-zinc-800 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <span className="font-semibold text-zinc-900 dark:text-zinc-100">VAPT-AI</span>
            <Badge variant="outline" className="text-[10px] text-zinc-500 border-zinc-300 dark:border-zinc-700 ml-1">
              v3.2
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPanicOpen(true)}
              className="border-red-300 dark:border-red-900 text-red-600 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-950 hover:text-red-500 dark:hover:text-red-200"
            >
              <AlertOctagon className="w-4 h-4 mr-2" />
              Panic
            </Button>

            <div className="text-xs text-zinc-600 dark:text-zinc-400">
              {user?.email}
              {user?.role && <span className="ml-1 text-zinc-500">({user.role})</span>}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={logout}
              className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
            >
              <LogOut className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Body: sidebar + content */}
      <div className="flex flex-1">
        {/* Sidebar */}
        <aside className="w-56 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-3">
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.name}
                onClick={() => setView(item.name)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                  view === item.name
                    ? "bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-800"
                    : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-zinc-100"
                }`}
              >
                {item.icon}
                {item.label}
              </button>
            ))}
          </nav>

          <div className="mt-8 p-3 bg-zinc-50 dark:bg-zinc-950 rounded border border-zinc-200 dark:border-zinc-800">
            <div className="text-xs text-zinc-500 dark:text-zinc-500 uppercase tracking-wider mb-2">Active scan</div>
            <ActiveScanWidget />
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-6 overflow-y-auto">
          {view === "dashboard" && <DashboardView onNavigate={(v) => setView(v as ViewName)} />}
          {view === "scans" && <ScansView />}
          {view === "reports" && (
            <ReportHistoryView
              onFindingClick={(finding) => setSelectedFinding({ id: finding.id, data: finding })}
            />
          )}
          {view === "findings" && <FindingsView />}
          {view === "tools" && <ToolsView />}
          {view === "settings" && <SettingsView />}
        </main>
      </div>

      {/* Panic modal */}
      {panicOpen && (
        <PanicModal
          onClose={() => setPanicOpen(false)}
          runningExecutions={runningExecutions}
          onRefresh={setRunningExecutions}
        />
      )}
    </div>
  );
}

/** Sidebar widget: shows active scan + tool execution count.
 *
 * Polling strategy (Phase D fix):
 *   - Always poll `/api/scans/active` every 15s — it's a cheap index lookup.
 *   - Only poll `/api/mcp/executions` when there IS an active scan (saves
 *     backend traffic + log spam when idle).
 *   - When an active scan is present, tighten the interval to 5s so the
 *     sidebar updates live during scans.
 *   - When the scan completes (no more active scans), back off to 15s.
 */
function ActiveScanWidget() {
  const [execs, setExecs] = useState<ToolExecution[]>([]);
  const [scans, setScans] = useState<any[]>([]);

  useEffect(() => {
    let active = true;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      if (!active) return;
      try {
        // Always fetch active scans — cheap lookup, tells us whether to
        // also fetch executions
        const scansResp = await import("../lib/api")
          .then((m) => m.getActiveScans())
          .catch(() => ({ active_scans: [] }));
        if (!active) return;
        const activeScans = scansResp.active_scans || [];
        setScans(activeScans);

        if (activeScans.length > 0) {
          // There IS an active scan — also fetch recent tool executions
          const execsResp = await getExecutions(undefined, 5).catch(() => ({ executions: [] }));
          if (!active) return;
          setExecs(execsResp.executions || []);
          // Tighten interval to 5s during active scans for live updates
          if (intervalId === null) {
            intervalId = setInterval(poll, 5000);
          }
        } else {
          // No active scan — clear stale executions + back off to 15s
          setExecs([]);
          if (intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
          }
        }
      } catch {
        // swallow — network hiccup, will retry on next tick
      }
    };

    poll();
    // Idle interval (15s) — gets tightened to 5s when an active scan appears
    const idleInterval = setInterval(poll, 15000);
    return () => {
      active = false;
      clearInterval(idleInterval);
      if (intervalId !== null) clearInterval(intervalId);
    };
  }, []);

  if (scans.length === 0) {
    return (
      <div className="text-xs text-zinc-600 italic">No active scans</div>
    );
  }

  return (
    <div className="space-y-2">
      {scans.slice(0, 1).map((s) => (
        <div key={s.scan_id}>
          <div className="text-xs font-mono text-zinc-300 truncate">{s.scan_id.slice(0, 12)}...</div>
          <div className="text-xs text-zinc-500 truncate">{s.target}</div>
          <div className="text-xs text-emerald-400">{s.progress}% done</div>
          <div className="text-xs text-zinc-600 mt-1">{execs.length} recent tool executions</div>
        </div>
      ))}
    </div>
  );
}

/** Panic modal: lists running tool executions + bulk cancel button per scan. */
function PanicModal({
  onClose,
  runningExecutions,
  onRefresh,
}: {
  onClose: () => void;
  runningExecutions: ToolExecution[];
  onRefresh: (execs: ToolExecution[]) => void;
}) {
  const { toast } = useToast();
  const [cancelling, setCancelling] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const resp = await getExecutions(undefined, 30);
        onRefresh(resp.executions || []);
      } catch {}
    };
    load();
    const i = setInterval(load, 3000);
    return () => clearInterval(i);
  }, []);

  const running = runningExecutions.filter((e) =>
    e.status === "running" || e.status === "queued"
  );

  // Group running executions by scan so we can offer one bulk-cancel
  // button per scan. The explicit type argument is required: without it
  // TS picks the non-generic `reduce` overload and `Object.entries()`
  // would infer the values as `unknown`.
  const byScan = running.reduce<Record<string, ToolExecution[]>>((acc, e) => {
    const key = e.scan_id || "(no scan)";
    (acc[key] = acc[key] || []).push(e);
    return acc;
  }, {});

  const handleCancelScan = async (scanId: string) => {
    setCancelling(scanId);
    try {
      const result = await cancelScan(scanId);
      toast({
        title: "Cancelled",
        description: result.message,
      });
    } catch (err: any) {
      toast({
        title: "Cancel failed",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setCancelling(null);
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 border border-zinc-800 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-zinc-800 p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertOctagon className="w-5 h-5 text-red-400" />
            <h2 className="text-lg font-semibold text-zinc-50">Panic button</h2>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-200 text-xl">×</button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {running.length === 0 ? (
            <div className="text-center text-zinc-500 py-8">
              No running tool executions. All clear.
            </div>
          ) : (
            <>
              <p className="text-sm text-zinc-400">
                {running.length} tool execution(s) currently running. Cancel by scan to stop all tools for that scan.
              </p>
              {Object.entries(byScan).map(([scanId, execs]) => (
                <div key={scanId} className="border border-zinc-800 rounded p-3 bg-zinc-950">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <div className="text-xs text-zinc-500 uppercase tracking-wider">Scan</div>
                      <div className="text-sm font-mono text-zinc-300">{scanId}</div>
                    </div>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleCancelScan(scanId)}
                      disabled={cancelling === scanId}
                      className="bg-red-700 hover:bg-red-600"
                    >
                      {cancelling === scanId ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : null}
                      Cancel all ({execs.length})
                    </Button>
                  </div>
                  <div className="space-y-1 text-xs text-zinc-500 font-mono">
                    {execs.slice(0, 5).map((e) => (
                      <div key={e.id} className="flex justify-between">
                        <span className="truncate">{e.tool_name}</span>
                        <span className="text-zinc-600 ml-2">{e.status}</span>
                      </div>
                    ))}
                    {execs.length > 5 && (
                      <div className="text-zinc-600 italic">+ {execs.length - 5} more...</div>
                    )}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        <div className="border-t border-zinc-800 p-4 text-right">
          <Button variant="outline" onClick={onClose} className="border-zinc-700 text-zinc-200">
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}