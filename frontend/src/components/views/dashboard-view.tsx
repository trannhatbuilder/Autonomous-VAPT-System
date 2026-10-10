'use client';

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Loader2, Bug, Zap, RefreshCw, Radar } from "lucide-react";
import { getFindings, getExecutions, getScanHistory } from "../../lib/api";

export function DashboardView({ onNavigate }: { onNavigate: (view: string) => void }) {
  const [loading, setLoading] = useState(true);
  const [findingsCount, setFindingsCount] = useState(0);
  const [findingsBySeverity, setFindingsBySeverity] = useState<Record<string, number>>({});
  const [recentExecutions, setRecentExecutions] = useState<any[]>([]);
  const [scansCount, setScansCount] = useState(0);

  const loadDashboard = async () => {
    setLoading(true);
    try {
      const [findingsResp, execsResp, scansResp] = await Promise.all([
        getFindings({ limit: 500 }).catch(() => ({ findings: [], total: 0 })),
        getExecutions(undefined, 10).catch(() => ({ executions: [] })),
        getScanHistory({ limit: 1 }).catch(() => ({ scans: [], total: 0 })),
      ]);

      const allFindings = (findingsResp.findings || []) as any[];
      setFindingsCount(findingsResp.total || allFindings.length);
      const bySev: Record<string, number> = {};
      allFindings.forEach((f) => {
        const sev = (f.severity || "info").toLowerCase();
        bySev[sev] = (bySev[sev] || 0) + 1;
      });
      setFindingsBySeverity(bySev);

      setRecentExecutions(execsResp.executions || []);
      setScansCount(scansResp.total || 0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadDashboard(); }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Dashboard</h2>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-1">Overview of scans, findings, and tool executions.</p>
        </div>
        <Button
          onClick={loadDashboard}
          variant="outline"
          size="sm"
          className="border-zinc-300 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
        </div>
      ) : (
        <>
          {/* Stat cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatCard
              label="Total scans"
              value={scansCount}
              icon={<Radar className="w-5 h-5 text-emerald-500 dark:text-emerald-400" />}
              onClick={() => onNavigate("scans")}
            />
            <StatCard
              label="Total findings"
              value={findingsCount}
              icon={<Bug className="w-5 h-5 text-amber-500 dark:text-amber-400" />}
              onClick={() => onNavigate("findings")}
            />
            <StatCard
              label="Critical findings"
              value={findingsBySeverity.critical || 0}
              icon={<Bug className="w-5 h-5 text-red-500 dark:text-red-400" />}
              onClick={() => onNavigate("findings")}
            />
            <StatCard
              label="High findings"
              value={findingsBySeverity.high || 0}
              icon={<Bug className="w-5 h-5 text-orange-500 dark:text-orange-400" />}
              onClick={() => onNavigate("findings")}
            />
          </div>

          {/* Severity breakdown */}
          <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base">Severity breakdown</CardTitle>
              <CardDescription className="text-zinc-600 dark:text-zinc-400">Distribution of findings by severity.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-5 gap-3">
                {["critical", "high", "medium", "low", "info"].map((sev) => {
                  const count = findingsBySeverity[sev] || 0;
                  const colors: Record<string, string> = {
                    critical: "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 border-red-200 dark:border-red-900",
                    high: "bg-orange-50 dark:bg-orange-950/40 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-900",
                    medium: "bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-900",
                    low: "bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-900",
                    info: "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border-zinc-300 dark:border-zinc-700",
                  };
                  return (
                    <div key={sev} className={`rounded-lg border p-3 ${colors[sev]}`}>
                      <div className="text-xs uppercase tracking-wider opacity-80">{sev}</div>
                      <div className="text-2xl font-semibold mt-1 tabular-nums">{count}</div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Recent executions */}
          <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-500 dark:text-amber-400" />
                Recent tool executions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 max-h-80 overflow-y-auto">
              {recentExecutions.length === 0 ? (
                <div className="text-sm text-zinc-500 italic">No executions yet. Run a scan to see tool activity.</div>
              ) : (
                recentExecutions.map((e) => (
                  <div key={e.id} className="flex items-center justify-between gap-4 p-2 bg-zinc-50 dark:bg-zinc-950 rounded border border-zinc-200 dark:border-zinc-800 text-xs">
                    <div className="font-mono text-zinc-700 dark:text-zinc-300 truncate">{e.tool_name}</div>
                    <Badge variant="outline" className={
                      e.status === "completed" ? "text-emerald-500 dark:text-emerald-400 border-emerald-800" :
                      e.status === "failed" ? "text-red-500 dark:text-red-400 border-red-800" :
                      e.status === "cancelled" ? "text-zinc-600 dark:text-zinc-400 border-zinc-300 dark:border-zinc-700" :
                      e.status === "hard_timeout" ? "text-orange-400 border-orange-800" :
                      "text-zinc-500 border-zinc-300 dark:border-zinc-700"
                    }>
                      {e.status}
                    </Badge>
                    <div className="text-zinc-600 tabular-nums">{e.duration_seconds?.toFixed(2)}s</div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, icon, onClick }: { label: string; value: number; icon: React.ReactNode; onClick?: () => void }) {
  return (
    <Card
      className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer"
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs uppercase tracking-wider text-zinc-500">{label}</div>
            <div className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100 mt-1 tabular-nums">{value}</div>
          </div>
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}