'use client';

import { useEffect, useState, useRef, useCallback } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { Alert, AlertDescription } from "../ui/alert";
import {
  Loader2, Play, Square, RefreshCw, Activity, Terminal,
  ChevronDown, ChevronRight, Search, History, Trash2, Download,
} from "lucide-react";
import {
  startScan, getScanEventsUrl, abortScan,
  getScanHistory, getScanDetail, getProcessDetails, getProcessDetail,
  deleteScan, exportVulnerabilities, downloadTextFile,
  type ScanSummary, type ScanDetail, type ProcessDetailRow, type ProcessDetailFull,
} from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

// ── Scan event types (for live SSE) ────────────────────────────────────
type EventType =
  | "scan_started" | "scan_progress" | "phase_change" | "iteration"
  | "tool_call_started" | "tool_call_completed" | "tool_call_progress"
  | "assistant_message" | "thinking" | "finding_detected"
  | "hitl_approval_required" | "hitl_decision_made"
  | "scan_complete" | "scan_error" | "heartbeat" | "unknown";

interface ScanEvent {
  event?: string;
  type: EventType;
  scan_id?: string;
  turn?: number;
  progress?: number;
  iteration?: number;
  scope?: string;
  phase?: string;
  message?: string;
  thought?: string;
  tool_name?: string;
  observation?: string;
  agent_name?: string;
  tool_call_id?: string;
  arguments?: Record<string, unknown>;
  index?: number;
  total?: number;
  success?: boolean;
  result_preview?: string;
  execution_id?: string;
  error?: string;
  status?: string;
  content?: string;
  reasoning?: string;
  text?: string;
  finding_id?: string;
  vuln_type?: string;
  severity?: string;
  location?: string;
  findings_count?: number;
  duration_seconds?: number;
  /**
   * Unix timestamp in SECONDS (float) — set by the backend event bus via
   * Python's `time.time()` (see app/pentest/events.py) and sent as a JSON
   * number. Typed loosely because non-SSE sources may serialize it as an
   * ISO-8601 string.
   */
  timestamp?: number | string;
}

/**
 * Format an event timestamp for display.
 *
 * The SSE bus emits Unix seconds (`time.time()`), so numeric values must be
 * multiplied by 1000 before being passed to `new Date(...)`. ISO strings are
 * used as-is. Returns "" for missing/invalid input so callers can fall back
 * to a dash.
 */
function formatEventTime(ts: number | string | null | undefined): string {
  if (ts === null || ts === undefined || ts === "") return "";
  const date = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString();
}

// ── Main ScansView (2-pane layout) ─────────────────────────────────────
// Mirrors CyberStrikeAI chat UI:
//   Left sidebar: list past scans + start new scan form
//   Right main: when scan selected → show live timeline (SSE if running,
//                DB replay if finished)

export function ScansView() {
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [scansTotal, setScansTotal] = useState(0);
  const [scansPage, setScansPage] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [loadingScans, setLoadingScans] = useState(true);
  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [liveScanId, setLiveScanId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<string>("idle");

  // Start new scan form state
  const [target, setTarget] = useState("");
  const [userPrompt, setUserPrompt] = useState("");
  const [starting, setStarting] = useState(false);

  const { toast } = useToast();
  const PAGE_SIZE = 20;

  const loadScans = useCallback(async (page = 0, search = "") => {
    setLoadingScans(true);
    try {
      const resp = await getScanHistory({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        search: search || undefined,
        sort_by: "started_at",
        sort_dir: "desc",
      });
      setScans(resp.scans);
      setScansTotal(resp.total);
      setScansPage(page);
    } catch (err: any) {
      toast({
        title: "Failed to load scan history",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoadingScans(false);
    }
  }, [toast]);

  useEffect(() => { loadScans(0, ""); }, [loadScans]);

  // ── Start a new scan ─────────────────────────────────────────────────
  const handleStartScan = async () => {
    if (!target) {
      toast({ title: "Target required", description: "Enter a target URL or IP to start a scan.", variant: "destructive" });
      return;
    }
    setStarting(true);
    setLiveStatus("starting");
    try {
      const result = await startScan(target, userPrompt);
      if (result.status === "preflight_failed" || !result.scan_id) {
        setLiveStatus("error");
        toast({ title: "Cannot start scan", description: result.message || "Tools missing.", variant: "destructive" });
        return;
      }
      setLiveScanId(result.scan_id);
      setSelectedScanId(result.scan_id);
      setLiveStatus("running");
      toast({ title: "Scan started", description: `Scan ID: ${result.scan_id.slice(0, 16)}...` });
    } catch (err: any) {
      setLiveStatus("error");
      toast({ title: "Failed to start scan", description: err.message, variant: "destructive" });
    } finally {
      setStarting(false);
    }
  };

  const handleAbort = async () => {
    if (!liveScanId) return;
    try {
      await abortScan(liveScanId);
      toast({ title: "Scan aborted", description: `Cancelled scan ${liveScanId.slice(0, 16)}...` });
      setLiveStatus("aborted");
      setLiveScanId(null);
    } catch (err: any) {
      toast({ title: "Abort failed", description: err.message, variant: "destructive" });
    }
  };

  const handleSelectScan = (scanId: string) => {
    setSelectedScanId(scanId);
    // If the selected scan is still running (from getScanHistory we have status),
    // use live SSE; otherwise use DB replay
    const scan = scans.find(s => s.id === scanId);
    if (scan && (scan.status === "running" || scan.status === "pending")) {
      setLiveScanId(scanId);
      setLiveStatus("running");
    } else {
      setLiveScanId(null);
      setLiveStatus("idle");
    }
  };

  const handleDeleteScan = async (scanId: string) => {
    if (!confirm(`Delete scan ${scanId.slice(0, 16)}...? Findings will be preserved (with scan_tag snapshot).`)) return;
    try {
      const result = await deleteScan(scanId);
      toast({
        title: "Scan deleted",
        description: `Findings preserved: ${result.findings_preserved}, process_details deleted: ${result.process_details_deleted}`,
      });
      // Reload the list
      loadScans(scansPage, searchQuery);
      if (selectedScanId === scanId) setSelectedScanId(null);
    } catch (err: any) {
      toast({ title: "Delete failed", description: err.message, variant: "destructive" });
    }
  };

  const isLive = liveScanId === selectedScanId && (liveStatus === "running" || liveStatus === "starting");

  return (
    <div className="flex h-[calc(100vh-4rem)] -mx-6 -my-6">
      {/* Left sidebar: Start form + scan history list */}
      <aside className="w-96 border-r border-zinc-800 bg-zinc-900/40 flex flex-col overflow-hidden">
        {/* Start new scan form */}
        <div className="p-4 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-zinc-200 mb-3">Start new scan</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="target" className="text-xs text-zinc-400">Target</Label>
              <Input
                id="target"
                type="text"
                placeholder="https://example.com or 10.0.0.1"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={starting}
                className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono text-sm h-9"
              />
            </div>
            <div>
              <Label htmlFor="prompt" className="text-xs text-zinc-400">Prompt (optional)</Label>
              <Input
                id="prompt"
                type="text"
                placeholder="Scan for SQLi + XSS"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                disabled={starting}
                className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 text-sm h-9"
              />
            </div>
            {!isLive ? (
              <Button onClick={handleStartScan} disabled={!target || starting} className="w-full bg-emerald-600 hover:bg-emerald-500 text-zinc-50 h-9" size="sm">
                {starting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                Start scan
              </Button>
            ) : (
              <Button onClick={handleAbort} variant="destructive" className="w-full bg-red-700 hover:bg-red-600 h-9" size="sm">
                <Square className="w-4 h-4 mr-2" /> Abort scan
              </Button>
            )}
          </div>
        </div>

        {/* Search + scan history list */}
        <div className="p-3 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-zinc-500 shrink-0" />
            <Input
              type="text"
              placeholder="Search scans..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") loadScans(0, searchQuery); }}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 text-sm h-8"
            />
            <Button size="sm" variant="ghost" onClick={() => loadScans(0, searchQuery)} className="text-zinc-400 hover:text-zinc-100 h-8 px-2">
              <Search className="w-3 h-3" />
            </Button>
          </div>
        </div>

        {/* Scan list */}
        <div className="flex-1 overflow-y-auto">
          {loadingScans ? (
            <div className="p-4 text-center text-zinc-600 text-sm">
              <Loader2 className="w-4 h-4 animate-spin mx-auto mb-2" />
              Loading scans...
            </div>
          ) : scans.length === 0 ? (
            <div className="p-4 text-center text-zinc-600 text-sm">No scans found.</div>
          ) : (
            scans.map((scan) => (
              <ScanListItem
                key={scan.id}
                scan={scan}
                isSelected={selectedScanId === scan.id}
                onSelect={() => handleSelectScan(scan.id)}
                onDelete={() => handleDeleteScan(scan.id)}
              />
            ))
          )}
          {/* Pagination */}
          {scansTotal > PAGE_SIZE && (
            <div className="p-2 flex items-center justify-between text-xs text-zinc-500">
              <span>Page {scansPage + 1} of {Math.ceil(scansTotal / PAGE_SIZE)}</span>
              <div className="flex gap-1">
                <Button size="sm" variant="ghost" disabled={scansPage === 0} onClick={() => loadScans(scansPage - 1, searchQuery)} className="h-7 px-2 text-xs">
                  Prev
                </Button>
                <Button size="sm" variant="ghost" disabled={(scansPage + 1) * PAGE_SIZE >= scansTotal} onClick={() => loadScans(scansPage + 1, searchQuery)} className="h-7 px-2 text-xs">
                  Next
                </Button>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Right main: scan detail / live timeline */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {!selectedScanId ? (
          <div className="flex-1 flex items-center justify-center text-zinc-600">
            <div className="text-center">
              <History className="w-12 h-12 mx-auto mb-3 text-zinc-700" />
              <p className="text-sm">Select a scan from the sidebar or start a new scan.</p>
            </div>
          </div>
        ) : (
          <ScanTimeline
            scanId={selectedScanId}
            isLive={isLive}
            onLiveComplete={() => { setLiveScanId(null); setLiveStatus("completed"); loadScans(0, searchQuery); }}
          />
        )}
      </main>
    </div>
  );
}

// ── ScanListItem (sidebar item) ───────────────────────────────────────

function ScanListItem({ scan, isSelected, onSelect, onDelete }: {
  scan: ScanSummary;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const statusColor =
    scan.status === "completed" ? "text-emerald-400" :
    scan.status === "running" ? "text-blue-400" :
    scan.status === "failed" ? "text-red-400" :
    scan.status === "aborted" ? "text-zinc-400" :
    "text-zinc-500";
  const bgClass = isSelected ? "bg-zinc-800/60 border-l-2 border-emerald-600" : "hover:bg-zinc-800/30 border-l-2 border-transparent";
  return (
    <div onClick={onSelect} className={`cursor-pointer px-3 py-3 border-b border-zinc-800 ${bgClass} transition-colors`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className={`text-xs font-mono truncate ${statusColor}`}>{scan.target}</div>
          <div className="text-[10px] text-zinc-600 mt-1 truncate">
            {scan.started_at ? new Date(scan.started_at).toLocaleString() : "—"}
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="text-zinc-700 hover:text-red-400 shrink-0"
          title="Delete scan (findings preserved)"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
      <div className="flex items-center gap-2 mt-1.5 text-[10px]">
        <span className={statusColor}>{scan.status}</span>
        {scan.findings_count !== null && scan.findings_count > 0 && (
          <span className="text-amber-400">{scan.findings_count} findings</span>
        )}
        {scan.progress !== null && scan.progress < 100 && scan.status === "running" && (
          <span className="text-zinc-500">{scan.progress}%</span>
        )}
      </div>
    </div>
  );
}

// ── ScanTimeline (right pane) ─────────────────────────────────────────
// If scan is live (running): connect SSE, append events as they arrive.
// If scan is finished: fetch process_details from DB (paginated).

function ScanTimeline({ scanId, isLive, onLiveComplete }: {
  scanId: string;
  isLive: boolean;
  onLiveComplete: () => void;
}) {
  const [scan, setScan] = useState<ScanDetail | null>(null);
  const [events, setEvents] = useState<ScanEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [dbEvents, setDbEvents] = useState<ProcessDetailRow[]>([]);
  const [dbTotal, setDbTotal] = useState(0);
  const [dbOffset, setDbOffset] = useState(0);
  const [loadingDb, setLoadingDb] = useState(false);
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const eventSourceRef = useRef<EventSource | null>(null);
  const timelineEndRef = useRef<HTMLDivElement | null>(null);

  // ── Load scan detail (always) ─────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const detail = await getScanDetail(scanId, { include_process_details: 0 });
        if (!cancelled) {
          setScan(detail);
          setProgress(detail.progress);
        }
      } catch (err) {
        console.error("Failed to load scan detail", err);
      }
    })();
    return () => { cancelled = true; };
  }, [scanId]);

  // ── Live mode: SSE stream ────────────────────────────────────────
  useEffect(() => {
    if (!isLive) {
      // Cleanup SSE if switching from live to non-live
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }
    const url = getScanEventsUrl(scanId);
    console.log("[VAPT-SSE] Connecting to:", url);
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => console.log("[VAPT-SSE] Connection opened");
    es.onmessage = (e) => {
      try {
        const raw: ScanEvent = JSON.parse(e.data);
        const evtName = (raw as any).event || (raw as any).type || "unknown";
        if (evtName === "heartbeat") return;
        const data: ScanEvent = { ...raw, type: evtName as EventType };
        setEvents((prev) => [...prev, data]);
        if (data.progress !== undefined && data.progress !== null) setProgress(data.progress);
        if (data.type === "scan_complete") { es.close(); onLiveComplete(); }
        else if (data.type === "scan_error") { es.close(); onLiveComplete(); }
      } catch (err) { console.warn("[VAPT-SSE] Failed to parse SSE event:", e.data); }
    };
    es.onerror = () => {
      if (es.readyState === 2) console.warn("[VAPT-SSE] Connection closed (state=2)");
    };
    return () => { es.close(); eventSourceRef.current = null; };
  }, [scanId, isLive, onLiveComplete]);

  // ── DB replay mode: fetch process_details ────────────────────────
  useEffect(() => {
    if (isLive) { setDbEvents([]); setDbTotal(0); setDbOffset(0); return; }
    setLoadingDb(true);
    (async () => {
      try {
        const resp = await getProcessDetails(scanId, { limit: 100, offset: 0 });
        setDbEvents(resp.process_details);
        setDbTotal(resp.total);
        setDbOffset(resp.process_details.length);
      } catch (err) {
        console.error("Failed to load process_details", err);
      } finally { setLoadingDb(false); }
    })();
  }, [scanId, isLive]);

  // ── Auto-scroll to bottom on new events ─────────────────────────
  useEffect(() => {
    if (timelineEndRef.current) {
      timelineEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [events, dbEvents]);

  const loadMoreDb = async () => {
    if (dbOffset >= dbTotal) return;
    setLoadingDb(true);
    try {
      const resp = await getProcessDetails(scanId, { limit: 100, offset: dbOffset });
      setDbEvents((prev) => [...prev, ...resp.process_details]);
      setDbOffset((prev) => prev + resp.process_details.length);
    } finally { setLoadingDb(false); }
  };

  const toggleTool = (id: string) => setExpandedTools((prev) => ({ ...prev, [id]: !prev[id] }));

  // ── Build unified timeline ──────────────────────────────────────
  // Live events come from SSE (ScanEvent[]), DB events come from
  // process_details (ProcessDetailRow[]). We render both in the same
  // timeline — live takes precedence (newer) when both are present.
  type TimelineItem =
    | { kind: "live"; event: ScanEvent; key: string }
    | { kind: "db"; row: ProcessDetailRow; key: string };

  const timeline: TimelineItem[] = [];
  if (isLive && events.length > 0) {
    events.forEach((ev, idx) => timeline.push({ kind: "live", event: ev, key: `live-${idx}` }));
  } else if (!isLive && dbEvents.length > 0) {
    dbEvents.forEach((row) => timeline.push({ kind: "db", row, key: `db-${row.id}` }));
  }

  return (
    <div className="flex-1 overflow-hidden flex flex-col">
      {/* Scan header */}
      <div className="px-6 py-4 border-b border-zinc-800 bg-zinc-900/40">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-semibold text-zinc-100 truncate">
              {scan?.target || "Loading..."}
            </h2>
            <p className="text-xs text-zinc-500 font-mono mt-1">
              {scanId}
              {scan?.started_at && <span className="ml-2">· {new Date(scan.started_at).toLocaleString()}</span>}
              {scan?.completed_at && <span className="ml-2">→ {new Date(scan.completed_at).toLocaleString()}</span>}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className={
              scan?.status === "completed" ? "bg-emerald-700 text-zinc-50" :
              scan?.status === "running" ? "bg-blue-700 text-zinc-50" :
              scan?.status === "failed" ? "bg-red-700 text-zinc-50" :
              "bg-zinc-700 text-zinc-200"
            }>
              {isLive ? "live" : scan?.status || "—"}
            </Badge>
            {scan?.findings_count !== null && scan?.findings_count !== undefined && scan.findings_count > 0 && (
              <Badge className="bg-amber-700 text-zinc-50">
                {scan.findings_count} findings
              </Badge>
            )}
          </div>
        </div>
        {/* Progress bar */}
        <div className="mt-3 flex items-center gap-3">
          <Progress value={progress} className="h-1.5 bg-zinc-800 flex-1" />
          <span className="text-xs text-zinc-500 tabular-nums w-10 text-right">{progress}%</span>
        </div>
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto bg-zinc-950 p-4">
        <div className="max-w-4xl mx-auto space-y-1 font-mono text-xs">
          {timeline.length === 0 ? (
            <div className="text-zinc-600 italic text-center py-12">
              {isLive ? "Waiting for events..." : (loadingDb ? "Loading timeline..." : "No events recorded.")}
            </div>
          ) : (
            timeline.map((item) => {
              if (item.kind === "live") {
                return <LiveEventLine key={item.key} event={item.event} />;
              } else {
                return <DbEventLine key={item.key} row={item.row} scanId={scanId} expanded={!!expandedTools[item.row.id]} onToggle={() => toggleTool(item.row.id)} />;
              }
            })
          )}
          {timeline.length > 0 && <div ref={timelineEndRef} />}
        </div>
        {/* Load more (DB mode only) */}
        {!isLive && dbOffset < dbTotal && (
          <div className="text-center py-4">
            <Button size="sm" variant="ghost" onClick={loadMoreDb} disabled={loadingDb} className="text-zinc-400">
              {loadingDb ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <ChevronDown className="w-4 h-4 mr-2" />}
              Load more ({dbTotal - dbOffset} remaining)
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── LiveEventLine (rendered for SSE events during live scan) ──────────

function LiveEventLine({ event }: { event: ScanEvent }) {
  const time = formatEventTime(event.timestamp);
  const type = event.type;

  if (type === "phase_change") {
    return (
      <div className="flex items-center gap-2 py-1 mt-2 border-t border-zinc-800">
        <span className="text-zinc-700">{time}</span>
        <span className="text-purple-400">━━━</span>
        <span className="text-purple-300 font-semibold">{event.phase}</span>
        {event.message && <span className="text-zinc-500">— {event.message}</span>}
        {event.progress !== undefined && <span className="text-zinc-600 ml-auto">[{event.progress}%]</span>}
      </div>
    );
  }
  if (type === "iteration") {
    return (
      <div className="flex gap-2 py-1 mt-1 text-zinc-500">
        <span className="text-zinc-700">{time}</span>
        <span>──</span>
        <span>{event.scope === "main" ? "Main agent" : `Sub-agent ${event.agent_name || "?"}`} · round {event.iteration}</span>
      </div>
    );
  }
  if (type === "thinking") {
    return (
      <div className="flex gap-2 leading-relaxed text-zinc-500 italic">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0">think</span>
        <span className="break-all">{event.text}</span>
      </div>
    );
  }
  if (type === "assistant_message") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-blue-950/20 border border-blue-900/50 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-blue-400">{event.agent_name || "assistant"}</span>
        <span className="text-zinc-200 break-all whitespace-pre-wrap">{event.content}</span>
      </div>
    );
  }
  if (type === "finding_detected") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-amber-950/30 border border-amber-900 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-amber-400">FINDING [{event.severity}]</span>
        <span className="text-amber-200 break-all">{event.vuln_type} @ {event.location}</span>
      </div>
    );
  }
  if (type === "scan_started") {
    return (
      <div className="flex gap-2 leading-relaxed py-1 border-b border-zinc-800 mb-2">
        <span className="text-zinc-700">{time || "—"}</span>
        <span className="text-blue-400">SCAN STARTED</span>
        <span className="text-zinc-300 break-all">target: {event.scan_id || ""}</span>
      </div>
    );
  }
  if (type === "scan_complete") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-emerald-950/30 border border-emerald-900 rounded">
        <span className="text-zinc-700">{time || "—"}</span>
        <span className="text-emerald-400">SCAN COMPLETE</span>
        <span className="text-emerald-200">{event.findings_count} findings · {event.duration_seconds?.toFixed(1)}s</span>
      </div>
    );
  }
  if (type === "scan_error") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-red-950/30 border border-red-900 rounded">
        <span className="text-zinc-700">{time || "—"}</span>
        <span className="text-red-400">SCAN ERROR</span>
        <span className="text-red-200 break-all">{event.error}</span>
      </div>
    );
  }
  if (type === "tool_call_started") {
    return (
      <div className="flex gap-2 leading-relaxed my-0.5 p-2 bg-amber-950/20 border border-amber-900/50 rounded">
        <span className="text-zinc-700 shrink-0">{time}</span>
        <span className="shrink-0 text-amber-400">TOOL</span>
        <span className="text-zinc-200 font-semibold shrink-0">{event.tool_name}</span>
        {event.agent_name && <span className="text-zinc-500 text-[10px]">[{event.agent_name}]</span>}
        {event.arguments && <span className="text-zinc-500 truncate flex-1">({JSON.stringify(event.arguments).slice(0, 80)})</span>}
        <span className="text-amber-400 animate-pulse shrink-0">running</span>
      </div>
    );
  }
  if (type === "tool_call_completed") {
    const success = event.success;
    return (
      <div className={`flex gap-2 leading-relaxed my-0.5 p-2 border rounded ${success ? "bg-emerald-950/20 border-emerald-900/50" : "bg-red-950/20 border-red-900/50"}`}>
        <span className="text-zinc-700 shrink-0">{time}</span>
        <span className={`shrink-0 ${success ? "text-emerald-400" : "text-red-400"}`}>
          {success ? "OK" : "FAIL"} {event.tool_name}
        </span>
        {event.result_preview && (
          <span className={`truncate flex-1 ${success ? "text-zinc-400" : "text-red-300"}`}>
            {event.result_preview}
          </span>
        )}
      </div>
    );
  }
  // scan_progress fallback
  if (type === "scan_progress") {
    const prefix = event.agent_name ? `[${event.agent_name}]` : "[orchestrator]";
    let content = event.thought || "";
    if (event.tool_name) content += ` | tool=${event.tool_name}`;
    return (
      <div className="flex gap-2 leading-relaxed">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-zinc-500">{prefix}</span>
        <span className="text-zinc-300 break-all">{content}</span>
      </div>
    );
  }
  // Unknown — skip rendering
  return null;
}

// ── DbEventLine (rendered for DB-replayed events from process_details) ─

function DbEventLine({ row, scanId, expanded, onToggle }: {
  row: ProcessDetailRow;
  scanId: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const time = row.created_at ? new Date(row.created_at).toLocaleTimeString() : "";
  const type = row.event_type;
  const message = row.message || "";
  // data JSONB from DB — parse tool_name, agent_name, etc.
  const data = row.data as Record<string, any> | null;
  const toolName = data?.tool_name;
  const agentName = data?.agent_name;
  const success = data?.success;
  const argsPreview = data?.arguments ? JSON.stringify(data.arguments).slice(0, 100) : "";
  const resultPreview = data?.result_preview || data?.result || "";

  // Tool call events — render as expandable card
  if (type === "tool_call_started" || type === "tool_call_completed") {
    const isCompleted = type === "tool_call_completed";
    const statusIcon = isCompleted ? (success ? "OK" : "FAIL") : "...";
    const statusColor = isCompleted ? (success ? "text-emerald-400" : "text-red-400") : "text-amber-400 animate-pulse";
    const bgClass = isCompleted ? (success ? "bg-emerald-950/20 border-emerald-900/50" : "bg-red-950/20 border-red-900/50") : "bg-amber-950/20 border-amber-900/50";
    return (
      <div className={`my-0.5 p-2 border rounded ${bgClass}`}>
        <div className="flex items-center gap-2 cursor-pointer" onClick={onToggle}>
          <span className="text-zinc-700 shrink-0">{time}</span>
          <span className={`shrink-0 font-semibold ${statusColor}`}>{statusIcon}</span>
          <span className="text-zinc-200 shrink-0">{toolName}</span>
          {agentName && <span className="text-zinc-500 text-[10px]">[{agentName}]</span>}
          {argsPreview && <span className="text-zinc-500 truncate flex-1">({argsPreview})</span>}
          {expanded ? <ChevronDown className="w-3 h-3 ml-auto shrink-0" /> : <ChevronRight className="w-3 h-3 ml-auto shrink-0" />}
        </div>
        {expanded && (
          <DbToolPayload row={row} scanId={scanId} />
        )}
      </div>
    );
  }
  // Other event types — render as simple line
  return (
    <div className="flex gap-2 leading-relaxed">
      <span className="text-zinc-700 shrink-0">{time || "—"}</span>
      <span className="shrink-0 text-zinc-500">{type}</span>
      <span className="text-zinc-300 break-all">{message}</span>
      {resultPreview && <span className="text-zinc-500 break-all">— {String(resultPreview).slice(0, 100)}</span>}
    </div>
  );
}

// ── DbToolPayload (lazy-loaded full payload for expanded tool card) ───

function DbToolPayload({ row, scanId }: { row: ProcessDetailRow; scanId: string }) {
  const [full, setFull] = useState<ProcessDetailFull | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!row.data) {
      // If row.data is null, fetch full payload from /process-details/{id}
      setLoading(true);
      (async () => {
        try {
          const f = await getProcessDetail(scanId, row.id);
          setFull(f);
        } catch (err) {
          console.error("Failed to load full payload", err);
        } finally { setLoading(false); }
      })();
    }
  }, [row, scanId]);

  const data = full?.data || row.data;
  if (loading) return <div className="mt-2 pl-6 text-zinc-500 text-[11px]"><Loader2 className="w-3 h-3 animate-spin inline mr-1" /> Loading payload...</div>;
  if (!data) return <div className="mt-2 pl-6 text-zinc-600 text-[11px]">No payload.</div>;

  const argsStr = (data as any).arguments ? JSON.stringify((data as any).arguments, null, 2) : "";
  const resultStr = (data as any).result_preview || (data as any).result || "";
  const errorStr = (data as any).error || "";

  return (
    <div className="mt-2 pl-6 text-[11px] space-y-1">
      {argsStr && (
        <div>
          <span className="text-zinc-600">args:</span>
          <pre className="text-zinc-400 whitespace-pre-wrap break-all mt-1">{argsStr}</pre>
        </div>
      )}
      {resultStr && (
        <div>
          <span className="text-zinc-600">result:</span>
          <pre className={`whitespace-pre-wrap break-all mt-1 ${(data as any).success === false ? "text-red-300" : "text-zinc-300"}`}>
            {String(resultStr).slice(0, 500)}
          </pre>
        </div>
      )}
      {errorStr && (
        <div>
          <span className="text-zinc-600">error:</span>
          <pre className="text-red-300 whitespace-pre-wrap break-all mt-1">{String(errorStr).slice(0, 300)}</pre>
        </div>
      )}
    </div>
  );
}