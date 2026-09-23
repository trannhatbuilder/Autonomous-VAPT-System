'use client';

import { useState, useEffect } from "react";
import { useAuth } from "../components/auth/auth-provider";
import { LoginForm } from "../components/auth/login-form";
import { AuthProvider } from "../components/auth/auth-provider";
import { DashboardView } from "../components/views/dashboard-view";
import { ScansView } from "../components/views/scans-view";
import { FindingsView } from "../components/views/findings-view";
import { SettingsView } from "../components/views/settings-view";
import { ToolsView } from "../components/views/tools-view";
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
} from "lucide-react";
import { getExecutions, abortScan as cancelScan } from "../lib/api";
import { useToast } from "../hooks/use-toast";

type ViewName = "dashboard" | "scans" | "findings" | "tools" | "settings";

const NAV_ITEMS: { name: ViewName; label: string; icon: React.ReactNode }[] = [
  { name: "dashboard", label: "Dashboard", icon: <LayoutDashboard className="w-4 h-4" /> },
  { name: "scans", label: "Scans", icon: <Radar className="w-4 h-4" /> },
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

function AppShell() {
  const { user, loading, isAuthenticated, logout } = useAuth();
  const [view, setView] = useState<ViewName>("dashboard");
  const [panicOpen, setPanicOpen] = useState(false);
  const [runningExecutions, setRunningExecutions] = useState<any[]>([]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-950">
        <Loader2 className="w-8 h-8 animate-spin text-zinc-600" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginForm />;
  }

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950 text-zinc-100">
      {/* Top bar */}
      <header className="border-b border-zinc-800 bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-zinc-800 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
            </div>
            <span className="font-semibold">VAPT-AI</span>
            <Badge variant="outline" className="text-[10px] text-zinc-500 border-zinc-700 ml-1">
              v3.2
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPanicOpen(true)}
              className="border-red-900 text-red-300 hover:bg-red-950 hover:text-red-200"
            >
              <AlertOctagon className="w-4 h-4 mr-2" />
              Panic
            </Button>

            <div className="text-xs text-zinc-400">
              {user?.email}
              {user?.role && <span className="ml-1 text-zinc-600">({user.role})</span>}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={logout}
              className="text-zinc-400 hover:text-zinc-100"
            >
              <LogOut className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Body: sidebar + content */}
      <div className="flex flex-1">
        {/* Sidebar */}
        <aside className="w-56 border-r border-zinc-800 bg-zinc-900/40 p-3">
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.name}
                onClick={() => setView(item.name)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                  view === item.name
                    ? "bg-emerald-950 text-emerald-200 border border-emerald-800"
                    : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
                }`}
              >
                {item.icon}
                {item.label}
              </button>
            ))}
          </nav>

          <div className="mt-8 p-3 bg-zinc-950 rounded border border-zinc-800">
            <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Active scan</div>
            <ActiveScanWidget />
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-6 overflow-y-auto">
          {view === "dashboard" && <DashboardView onNavigate={(v) => setView(v as ViewName)} />}
          {view === "scans" && <ScansView />}
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

/** Sidebar widget: shows active scan + tool execution count. */
function ActiveScanWidget() {
  const [execs, setExecs] = useState<any[]>([]);
  const [scans, setScans] = useState<any[]>([]);

  useEffect(() => {
    const poll = async () => {
      try {
        const [execsResp, scansResp] = await Promise.all([
          getExecutions(undefined, 5).catch(() => ({ executions: [] })),
          import("../lib/api").then((m) => m.getActiveScans()).catch(() => ({ active_scans: [] })),
        ]);
        setExecs(execsResp.executions || []);
        setScans(scansResp.active_scans || []);
      } catch {}
    };
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
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
  runningExecutions: any[];
  onRefresh: (execs: any[]) => void;
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
              {Object.entries(running.reduce((acc, e) => {
                const key = e.scan_id || "(no scan)";
                (acc[key] = acc[key] || []).push(e);
                return acc;
              }, {} as Record<string, any[]>)).map(([scanId, execs]) => (
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
