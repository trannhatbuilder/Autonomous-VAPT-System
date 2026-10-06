"use client";

import { useEffect, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import {
  History,
  Search,
  ChevronDown,
  ChevronRight,
  Loader2,
  Bug,
  Calendar,
  ExternalLink,
  Globe,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { getScanHistory, getFindings, type ScanSummary, type ProcessDetailRow } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

/**
 * Report History View (Phase K) — W19-FIX4.
 *
 * Lists past scans GROUPED BY TARGET (one section per unique target).
 * User clicks a target → expands to show all scans for that target.
 * User clicks a scan → shows findings of that scan (in-line list).
 * User clicks a finding → onFindingClick(finding) → parent navigates to
 * finding-detail-view.
 *
 * Light/dark theme aware via Tailwind `dark:` variants.
 */

interface Finding {
  id: string;
  scan_id: string;
  name: string;
  vuln_type: string;
  severity: string;
  cvss_vector: string | null;
  cvss_score: number;
  location: string;
  description: string;
  remediation: string;
  poc_status: string;
  verified: boolean;
  internet_verified?: boolean;
  internet_verification?: {
    confirmed: boolean;
    confidence: number;
    references: string[];
    summary: string;
  } | null;
  created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-700 text-red-50 dark:bg-red-900 dark:text-red-100",
  high: "bg-orange-700 text-orange-50 dark:bg-orange-900 dark:text-orange-100",
  medium: "bg-amber-700 text-amber-50 dark:bg-amber-900 dark:text-amber-100",
  low: "bg-blue-700 text-blue-50 dark:bg-blue-900 dark:text-blue-100",
  info: "bg-zinc-700 text-zinc-100 dark:bg-zinc-800 dark:text-zinc-100",
};

const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-500",
  low: "bg-blue-500",
  info: "bg-zinc-500",
};

interface ReportHistoryViewProps {
  onFindingClick?: (finding: Finding) => void;
}

export function ReportHistoryView({ onFindingClick }: ReportHistoryViewProps) {
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTarget, setSearchTarget] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [expandedTargets, setExpandedTargets] = useState<Set<string>>(new Set());
  const [expandedScans, setExpandedScans] = useState<Set<string>>(new Set());
  const [scanFindings, setScanFindings] = useState<Record<string, Finding[]>>({});
  const [loadingFindingsFor, setLoadingFindingsFor] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const resp = await getScanHistory({
          limit: 100,
          offset: 0,
          sort_by: "started_at",
          sort_dir: "desc",
        });
        setScans(resp.scans || []);
      } catch (err: any) {
        toast({
          title: "Failed to load scan history",
          description: err.message,
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    })();
  }, [toast]);

  // Group scans by target (normalize URL → host:port)
  const groupedByTarget = useMemo<Record<string, ScanSummary[]>>(() => {
    const groups: Record<string, ScanSummary[]> = {};
    for (const scan of scans) {
      try {
        const url = new URL(scan.target);
        const key = `${url.hostname}${url.port ? `:${url.port}` : ""}`;
        (groups[key] = groups[key] || []).push(scan);
      } catch {
        // Not a URL — use raw target
        const key = scan.target || "unknown";
        (groups[key] = groups[key] || []).push(scan);
      }
    }
    // Sort each group by started_at desc (handle null started_at)
    Object.values(groups).forEach((group) => {
      group.sort((a, b) => {
        const aTime = a.started_at ? new Date(a.started_at).getTime() : 0;
        const bTime = b.started_at ? new Date(b.started_at).getTime() : 0;
        return bTime - aTime;
      });
    });
    return groups;
  }, [scans]);

  // Filter by search query
  const filteredGroups = useMemo<Record<string, ScanSummary[]>>(() => {
    if (!searchTarget) return groupedByTarget;
    const q = searchTarget.toLowerCase();
    const filtered: Record<string, ScanSummary[]> = {};
    for (const [target, groupScansArr] of Object.entries(groupedByTarget)) {
      if (target.toLowerCase().includes(q)) {
        filtered[target] = groupScansArr as ScanSummary[];
      }
    }
    return filtered;
  }, [groupedByTarget, searchTarget]);

  const toggleTarget = (target: string) => {
    setExpandedTargets((prev) => {
      const next = new Set(prev);
      if (next.has(target)) next.delete(target);
      else next.add(target);
      return next;
    });
  };

  const toggleScan = async (scanId: string) => {
    setExpandedScans((prev) => {
      const next = new Set(prev);
      if (next.has(scanId)) {
        next.delete(scanId);
      } else {
        next.add(scanId);
        // Lazy-load findings for this scan if not already loaded
        if (!scanFindings[scanId]) {
          setLoadingFindingsFor(scanId);
          getFindings({ scan_id: scanId, limit: 100 })
            .then((resp) => {
              setScanFindings((prev) => ({ ...prev, [scanId]: resp.findings || [] }));
            })
            .catch((err) => {
              toast({
                title: "Failed to load findings",
                description: err.message,
                variant: "destructive",
              });
            })
            .finally(() => setLoadingFindingsFor(null));
        }
      }
      return next;
    });
  };

  // Apply severity filter to findings
  const filterFindings = (findings: Finding[]) => {
    if (severityFilter === "all") return findings;
    return findings.filter((f) => f.severity.toLowerCase() === severityFilter);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-4 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50 flex items-center gap-2">
            <History className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
            Report History
          </h2>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-1">
            Past scans grouped by target. Click target to expand scans, click scan to see findings.
          </p>
        </div>
      </div>

      {/* Filters */}
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
            <Search className="w-4 h-4 text-zinc-500" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div className="space-y-2">
            <Label htmlFor="search_target" className="text-zinc-700 dark:text-zinc-200">
              Search by target
            </Label>
            <Input
              id="search_target"
              placeholder="example.com or 10.0.0.1"
              value={searchTarget}
              onChange={(e) => setSearchTarget(e.target.value)}
              className="bg-white dark:bg-zinc-950 border-zinc-300 dark:border-zinc-800 text-zinc-900 dark:text-zinc-50 placeholder-zinc-500 dark:placeholder-zinc-600 font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-zinc-700 dark:text-zinc-200">Severity filter</Label>
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="bg-white dark:bg-zinc-950 border-zinc-300 dark:border-zinc-800 text-zinc-900 dark:text-zinc-50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
                <SelectItem value="all">All severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            {Object.keys(filteredGroups).length} unique targets · {scans.length} total scans
          </div>
        </CardContent>
      </Card>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
        </div>
      )}

      {/* Empty state */}
      {!loading && scans.length === 0 && (
        <Card className="bg-white dark:bg-zinc-900/40 border-zinc-200 dark:border-zinc-800">
          <CardContent className="pt-6 text-center text-zinc-500 dark:text-zinc-400">
            No scans yet. Go to Scans tab to start one.
          </CardContent>
        </Card>
      )}

      {/* Grouped by target */}
      {!loading && Object.keys(filteredGroups).length > 0 && (
        <div className="space-y-3">
          {Object.keys(filteredGroups).map((target) => {
            const groupScans: ScanSummary[] = filteredGroups[target] || [];
            const isExpanded = expandedTargets.has(target);
            const totalFindings = groupScans.reduce((sum, s) => sum + (s.findings_count || 0), 0);
            return (
              <Card
                key={target}
                className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 overflow-hidden"
              >
                {/* Target header — click to expand */}
                <button
                  onClick={() => toggleTarget(target)}
                  className="w-full text-left p-4 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-zinc-500" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-zinc-500" />
                    )}
                    <Globe className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                    <div>
                      <div className="font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                        {target}
                      </div>
                      <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                        {groupScans.length} scan(s) · {totalFindings} finding(s) total
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {/* Severity mini-summary dots — uses findings_count as proxy
                        since ScanSummary doesn't have findings_by_severity in
                        the API type. Will be enhanced when backend exposes
                        per-scan severity breakdown. */}
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {totalFindings} total
                    </span>
                  </div>
                </button>

                {/* Expanded: list of scans for this target */}
                {isExpanded && (
                  <div className="border-t border-zinc-200 dark:border-zinc-800 divide-y divide-zinc-200 dark:divide-zinc-800">
                    {groupScans.map((scan) => {
                      const isScanExpanded = expandedScans.has(scan.id);
                      const findings = scanFindings[scan.id] || [];
                      const filteredFindings = filterFindings(findings);
                      return (
                        <div key={scan.id} className="p-4">
                          {/* Scan row — click to expand findings */}
                          <button
                            onClick={() => toggleScan(scan.id)}
                            className="w-full text-left flex items-center justify-between gap-2"
                          >
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                              {isScanExpanded ? (
                                <ChevronDown className="w-4 h-4 text-zinc-500 shrink-0" />
                              ) : (
                                <ChevronRight className="w-4 h-4 text-zinc-500 shrink-0" />
                              )}
                              <Calendar className="w-3 h-3 text-zinc-500 shrink-0" />
                              <span className="text-xs text-zinc-500 dark:text-zinc-400 shrink-0">
                                {scan.started_at
                                  ? new Date(scan.started_at).toLocaleString()
                                  : "—"}
                              </span>
                              <span className="text-xs text-zinc-500 dark:text-zinc-400 font-mono shrink-0">
                                {scan.id.slice(0, 16)}...
                              </span>
                              <Badge
                                className={`shrink-0 ${
                                  scan.status === "completed"
                                    ? "bg-emerald-700 text-emerald-50 dark:bg-emerald-900 dark:text-emerald-100"
                                    : scan.status === "running"
                                    ? "bg-blue-700 text-blue-50 dark:bg-blue-900 dark:text-blue-100"
                                    : scan.status === "aborted"
                                    ? "bg-zinc-700 text-zinc-100 dark:bg-zinc-800 dark:text-zinc-100"
                                    : "bg-red-700 text-red-50 dark:bg-red-900 dark:text-red-100"
                                }`}
                              >
                                {scan.status}
                              </Badge>
                              <span className="text-xs text-zinc-500 dark:text-zinc-400 shrink-0">
                                {scan.findings_count || 0} findings
                              </span>
                            </div>
                          </button>

                          {/* Expanded: findings for this scan */}
                          {isScanExpanded && (
                            <div className="mt-3 ml-7 space-y-2">
                              {loadingFindingsFor === scan.id && (
                                <div className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  Loading findings...
                                </div>
                              )}
                              {!loadingFindingsFor && filteredFindings.length === 0 && (
                                <div className="text-xs text-zinc-500 dark:text-zinc-400 italic pl-4">
                                  No findings for this scan (or all filtered out).
                                </div>
                              )}
                              {!loadingFindingsFor &&
                                filteredFindings.length > 0 &&
                                filteredFindings.map((finding) => (
                                  <FindingRow
                                    key={finding.id}
                                    finding={finding}
                                    onClick={() => onFindingClick?.(finding)}
                                  />
                                ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── FindingRow — single finding in the scan-expanded list ──────────

function FindingRow({ finding, onClick }: { finding: Finding; onClick: () => void }) {
  const sev = (finding.severity || "info").toLowerCase();
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-3 bg-zinc-50 dark:bg-zinc-950/60 border border-zinc-200 dark:border-zinc-800 rounded hover:border-emerald-500 dark:hover:border-emerald-700 transition-colors flex items-start gap-3"
    >
      <span className={`inline-block w-2 h-2 rounded-full mt-2 shrink-0 ${SEVERITY_DOT[sev]}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge
            className={`shrink-0 ${SEVERITY_COLORS[sev]}`}
          >
            {finding.severity}
          </Badge>
          <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
            {finding.name}
          </span>
          {finding.verified && (
            <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0" />
          )}
          {finding.internet_verified && (
            <Badge variant="outline" className="text-xs text-emerald-700 dark:text-emerald-300 border-emerald-700 dark:border-emerald-700 shrink-0">
              <Globe className="w-3 h-3 mr-1" />
              internet-checked
            </Badge>
          )}
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 font-mono truncate">
          {finding.vuln_type} @ {finding.location}
        </div>
        <div className="text-xs text-zinc-600 dark:text-zinc-500 mt-1 flex items-center gap-3 flex-wrap">
          <span>CVSS {finding.cvss_score}</span>
          <span>·</span>
          <span>PoC: {finding.poc_status}</span>
          {finding.cvss_vector && (
            <>
              <span>·</span>
              <span className="font-mono text-[10px]">{finding.cvss_vector}</span>
            </>
          )}
        </div>
      </div>
      <ExternalLink className="w-3 h-3 text-zinc-500 mt-2 shrink-0" />
    </button>
  );
}