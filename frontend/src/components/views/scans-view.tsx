'use client';

import { useEffect, useState, useRef } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { Alert, AlertDescription } from "../ui/alert";
import { Loader2, Play, Square, Activity, Terminal, ChevronDown, ChevronRight } from "lucide-react";
import { startScan, getScanEventsUrl, abortScan } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

// ── Event types ────────────────────────────────────────────────────────────
// Mirrors the Phase D event catalog in app/pentest/events.py.
type EventType =
  | "scan_started"
  | "scan_progress"
  | "phase_change"
  | "iteration"
  | "tool_call_started"
  | "tool_call_progress"
  | "tool_call_completed"
  | "assistant_message"
  | "thinking"
  | "finding_detected"
  | "hitl_approval_required"
  | "hitl_decision_made"
  | "scan_complete"
  | "scan_error"
  | "heartbeat"
  | "unknown";

interface ScanEvent {
  event?: string;
  type: EventType;
  scan_id?: string;
  // progress / iteration
  turn?: number;
  progress?: number;
  iteration?: number;
  scope?: string;
  phase?: string;
  message?: string;
  // scan_progress fields
  thought?: string;
  tool_name?: string;
  observation?: string;
  agent_name?: string;
  // tool_call_started / _completed fields
  tool_call_id?: string;
  arguments?: Record<string, unknown>;
  index?: number;
  total?: number;
  success?: boolean;
  result_preview?: string;
  execution_id?: string;
  elapsed_seconds?: number;
  error?: string;
  status?: string;
  // assistant_message
  content?: string;
  reasoning?: string;
  // thinking
  text?: string;
  // finding_detected
  finding_id?: string;
  vuln_type?: string;
  severity?: string;
  location?: string;
  // terminal
  findings_count?: number;
  duration_seconds?: number;
  // hitl
  hitl_id?: string;
  target?: string;
  predicted_impact?: string;
  decision?: string;
  decided_by?: string;
  comment?: string;
  // housekeeping
  // Unix epoch seconds (backend sends `time.time()`), NOT an ISO string.
  timestamp?: number;
}

// ── Per-tool-call status tracker ───────────────────────────────────────────
// Mirrors CyberStrikeAI's toolCallStatusMap: each `tool_call_started` event
// registers a "running" entry keyed by `tool_call_id`; the matching
// `tool_call_completed` event flips it to completed/failed. The EventLine
// for the started event re-renders with the new status when the completed
// event arrives (we re-scan the events list).
interface ToolCallState {
  tool_call_id: string;
  tool_name: string;
  status: "running" | "completed" | "failed";
  args_preview: string;
  result_preview?: string;
  error?: string;
  started_at: string;
  completed_at?: string;
  agent_name?: string;
  iteration?: number;
  // Latest elapsed time reported by a tool_call_progress heartbeat (seconds).
  elapsed_seconds?: number;
}

/** Convert an event's unix-epoch (seconds) timestamp to an ISO string. */
function toIso(timestamp?: number): string {
  return timestamp ? new Date(timestamp * 1000).toISOString() : new Date().toISOString();
}

function computeToolCallStates(events: ScanEvent[]): Record<string, ToolCallState> {
  const map: Record<string, ToolCallState> = {};
  for (const ev of events) {
    if (ev.type === "tool_call_started" && ev.tool_call_id) {
      map[ev.tool_call_id] = {
        tool_call_id: ev.tool_call_id,
        tool_name: ev.tool_name || "?",
        status: "running",
        args_preview: ev.arguments ? JSON.stringify(ev.arguments).slice(0, 120) : "",
        started_at: toIso(ev.timestamp),
        agent_name: ev.agent_name,
        iteration: ev.iteration,
      };
    } else if (ev.type === "tool_call_progress") {
      // Heartbeat while a tool runs. Progress events carry tool_name (not
      // tool_call_id), so attach to the most recent still-running card for
      // that tool. Keeps the tool card's elapsed timer ticking instead of
      // leaving the timeline frozen on `tool_call_started`.
      const running = Object.values(map).filter(
        (s) => s.status === "running" && s.tool_name === ev.tool_name,
      );
      const target = running[running.length - 1];
      if (target) {
        target.elapsed_seconds = ev.elapsed_seconds;
      }
    } else if (ev.type === "tool_call_completed" && ev.tool_call_id) {
      const existing = map[ev.tool_call_id];
      if (existing) {
        existing.status = ev.success ? "completed" : "failed";
        existing.result_preview = ev.result_preview;
        existing.error = ev.error;
        existing.completed_at = toIso(ev.timestamp);
      } else {
        // tool_call_completed without matching started (e.g. replay) — synthesize
        map[ev.tool_call_id] = {
          tool_call_id: ev.tool_call_id,
          tool_name: ev.tool_name || "?",
          status: ev.success ? "completed" : "failed",
          args_preview: "",
          result_preview: ev.result_preview,
          error: ev.error,
          started_at: toIso(ev.timestamp),
          completed_at: toIso(ev.timestamp),
          agent_name: ev.agent_name,
          iteration: ev.iteration,
        };
      }
    }
  }
  return map;
}

export function ScansView() {
  const { toast } = useToast();
  const [target, setTarget] = useState("");
  const [userPrompt, setUserPrompt] = useState("");
  const [starting, setStarting] = useState(false);
  const [activeScanId, setActiveScanId] = useState<string | null>(null);
  const [events, setEvents] = useState<ScanEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<string>("");
  const [status, setStatus] = useState<string>("idle");
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const eventSourceRef = useRef<EventSource | null>(null);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleStartScan = async () => {
    if (!target) {
      toast({
        title: "Target required",
        description: "Enter a target URL or IP to start a scan.",
        variant: "destructive",
      });
      return;
    }
    setStarting(true);
    setEvents([]);
    setProgress(0);
    setCurrentPhase("");
    setStatus("starting");
    try {
      const result = await startScan(target, userPrompt);

      // Phase D preflight check — backend may refuse to start scan
      // if critical tools are missing. Surface the message to the user.
      if (result.status === "preflight_failed" || !result.scan_id) {
        setStatus("error");
        toast({
          title: "Cannot start scan — tools missing",
          description: result.message || "Critical tools not installed.",
          variant: "destructive",
        });
        return;
      }

      setActiveScanId(result.scan_id);
      setStatus("running");
      toast({
        title: "Scan started",
        description: `Scan ID: ${result.scan_id.slice(0, 16)}...`,
      });

      // Connect SSE
      const url = getScanEventsUrl(result.scan_id);
      console.log("[VAPT-SSE] Connecting to:", url);
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onopen = () => {
        console.log("[VAPT-SSE] Connection opened");
      };

      es.onmessage = (e) => {
        try {
          const raw: ScanEvent = JSON.parse(e.data);
          const evtName = (raw as any).event || (raw as any).type || "unknown";
          if (evtName === "heartbeat") return;
          const data: ScanEvent = { ...raw, type: evtName as EventType };

          setEvents((prev) => [...prev, data]);

          if (data.progress !== undefined && data.progress !== null) {
            setProgress(data.progress);
          }
          if (data.type === "phase_change" && data.phase) {
            setCurrentPhase(data.phase);
          }
          if (data.type === "scan_complete") {
            console.log("[VAPT-SSE] Scan complete:", data);
            setStatus("completed");
            setProgress(100);
            es.close();
          } else if (data.type === "scan_error") {
            console.error("[VAPT-SSE] Scan error:", data);
            setStatus("error");
            es.close();
          }
        } catch (err) {
          console.warn("[VAPT-SSE] Failed to parse SSE event:", e.data);
        }
      };

      es.onerror = (e: any) => {
        const state = es.readyState;
        const stateName = state === 0 ? "CONNECTING" : state === 1 ? "OPEN" : "CLOSED";
        console.warn(`[VAPT-SSE] Error (readyState=${stateName}). Will auto-reconnect if not CLOSED.`);
        if (state === 2) {
          setStatus("error");
          toast({
            title: "SSE connection lost",
            description: "Scan status stream disconnected. Check backend log.",
            variant: "destructive",
          });
        }
      };
    } catch (err: any) {
      setStatus("error");
      toast({
        title: "Failed to start scan",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setStarting(false);
    }
  };

  const handleAbort = async () => {
    if (!activeScanId) return;
    try {
      await abortScan(activeScanId);
      toast({
        title: "Scan aborted",
        description: `Cancelled scan ${activeScanId.slice(0, 16)}...`,
      });
      setStatus("aborted");
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    } catch (err: any) {
      toast({
        title: "Abort failed",
        description: err.message,
        variant: "destructive",
      });
    }
  };

  const toggleTool = (id: string) => {
    setExpandedTools((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const isRunning = status === "running" || status === "starting";

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-zinc-50">Start a scan</h2>
        <p className="text-sm text-zinc-400 mt-1">
          Enter a single target (URL, IP, or hostname). VAPT-AI will run the supervisor
          multi-agent flow: recon → vuln-triage → penetration (HITL) → reporting.
        </p>
      </div>

      {/* Start form */}
      <Card className="bg-zinc-900/60 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-50">New scan</CardTitle>
          <CardDescription className="text-zinc-400">
            Single-target web pentest. For network ranges, use the CLI.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="target" className="text-zinc-200">Target</Label>
            <Input
              id="target"
              type="text"
              placeholder="https://example.com  |  192.168.1.10  |  example.com"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={isRunning}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono"
            />
            <p className="text-xs text-zinc-500">
              Must include scheme for URLs (https://). Subdomains of the target are auto-included in scope.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="prompt" className="text-zinc-200">Prompt <span className="text-zinc-500">(optional)</span></Label>
            <Textarea
              id="prompt"
              placeholder="Scan for OWASP Top 10 vulnerabilities. Focus on SQLi + XSS."
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              disabled={isRunning}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 min-h-[80px] resize-y"
            />
            <p className="text-xs text-zinc-500">
              Natural-language scan goal. Empty = full default pentest.
            </p>
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          {!isRunning ? (
            <Button
              onClick={handleStartScan}
              disabled={!target || starting}
              className="bg-emerald-600 hover:bg-emerald-500 text-zinc-50"
            >
              {starting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
              Start scan
            </Button>
          ) : (
            <Button
              onClick={handleAbort}
              variant="destructive"
              className="bg-red-700 hover:bg-red-600"
            >
              <Square className="w-4 h-4 mr-2" />
              Abort scan
            </Button>
          )}
        </CardFooter>
      </Card>

      {/* Live progress */}
      {activeScanId && (
        <Card className="bg-zinc-900/60 border-zinc-800">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-zinc-50 flex items-center gap-2">
                  <Activity className={`w-4 h-4 ${isRunning ? "text-emerald-400 animate-pulse" : "text-zinc-500"}`} />
                  Live scan progress
                </CardTitle>
                <CardDescription className="text-zinc-400 mt-1 font-mono text-xs">
                  scan_id: {activeScanId}
                  {currentPhase && <span className="ml-2 text-zinc-500">· phase: {currentPhase}</span>}
                </CardDescription>
              </div>
              <Badge
                variant={status === "completed" ? "default" : status === "error" ? "destructive" : "secondary"}
                className={
                  status === "completed" ? "bg-emerald-700 text-zinc-50" :
                  status === "error" ? "bg-red-700 text-zinc-50" :
                  status === "aborted" ? "bg-zinc-700 text-zinc-200" :
                  "bg-zinc-800 text-zinc-100"
                }
              >
                {status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Progress</span>
                <span className="tabular-nums">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2 bg-zinc-800" />
            </div>

            {/* Timeline */}
            <div className="space-y-1 max-h-[600px] overflow-y-auto bg-zinc-950 rounded border border-zinc-800 p-3 font-mono text-xs">
              {events.length === 0 ? (
                <div className="text-zinc-600 italic">Waiting for events...</div>
              ) : (
                events.map((ev, idx) => (
                  <EventLine
                    key={idx}
                    event={ev}
                    toolCallStates={computeToolCallStates(events.slice(0, idx + 1))}
                    expanded={expandedTools}
                    onToggle={toggleTool}
                  />
                ))
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── EventLine ──────────────────────────────────────────────────────────────
// Renders a single event in the timeline. Different event types get
// different visual treatments so the user can immediately see what's
// happening (e.g. a tool card with status dot, a phase divider, a
// chat bubble for assistant messages, etc.).

function EventLine({
  event,
  toolCallStates,
  expanded,
  onToggle,
}: {
  event: ScanEvent;
  toolCallStates: Record<string, ToolCallState>;
  expanded: Record<string, boolean>;
  onToggle: (id: string) => void;
}) {
  const time = event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : "";

  // ── Tool call events: render as a tool card with status dot ────────
  if (event.type === "tool_call_started" && event.tool_call_id) {
    const state = toolCallStates[event.tool_call_id];
    if (!state) return null; // shouldn't happen
    return (
      <ToolCard
        state={state}
        time={time}
        expanded={!!expanded[event.tool_call_id]}
        onToggle={() => onToggle(event.tool_call_id!)}
      />
    );
  }
  // tool_call_completed events don't render their own line — they update
  // the matching tool card's status. The card is rendered when its
  // tool_call_started event was emitted earlier in the timeline.
  if (event.type === "tool_call_completed") {
    return null;
  }
  // tool_call_progress heartbeats update the running tool card's elapsed
  // timer (see computeToolCallStates) — no separate timeline line.
  if (event.type === "tool_call_progress") {
    return null;
  }

  // ── Phase change: render as a divider ──────────────────────────────
  if (event.type === "phase_change") {
    return (
      <div className="flex items-center gap-2 py-1 mt-2 border-t border-zinc-800">
        <span className="text-zinc-600">{time}</span>
        <span className="text-purple-400 shrink-0">━━━ phase:</span>
        <span className="text-purple-300 font-semibold">{event.phase}</span>
        {event.message && <span className="text-zinc-500">— {event.message}</span>}
        {event.progress !== undefined && (
          <span className="text-zinc-600 ml-auto">[{event.progress}%]</span>
        )}
      </div>
    );
  }

  // ── Iteration: render as a sub-divider ──────────────────────────────
  if (event.type === "iteration") {
    const scope = event.scope === "main" ? "Main agent" : `Sub-agent ${event.agent_name || "?"}`;
    return (
      <div className="flex items-center gap-2 py-1 mt-1 text-zinc-500">
        <span className="text-zinc-700">{time}</span>
        <span className="text-zinc-500">──</span>
        <span>{scope} · round {event.iteration}</span>
      </div>
    );
  }

  // ── Thinking: brief status line ─────────────────────────────────────
  if (event.type === "thinking") {
    return (
      <div className="flex gap-2 leading-relaxed text-zinc-500 italic">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0">🤔</span>
        <span className="break-all">{event.text}</span>
      </div>
    );
  }

  // ── Assistant message: render as a chat bubble ─────────────────────
  if (event.type === "assistant_message") {
    return (
      <div className="flex gap-2 leading-relaxed my-1">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-blue-400">💬 {event.agent_name || "assistant"}</span>
        <span className="text-zinc-200 break-all whitespace-pre-wrap">{event.content}</span>
      </div>
    );
  }

  // ── Finding detected: amber highlight ──────────────────────────────
  if (event.type === "finding_detected") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-amber-950/30 border border-amber-900 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-amber-400">⚠ FINDING [{event.severity}]</span>
        <span className="text-amber-200 break-all">
          {event.vuln_type} @ {event.location}
        </span>
      </div>
    );
  }

  // ── Scan terminal events ────────────────────────────────────────────
  if (event.type === "scan_complete") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-emerald-950/30 border border-emerald-900 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-emerald-400">✓ SCAN COMPLETE</span>
        <span className="text-emerald-200 break-all">
          {event.findings_count} findings · {event.duration_seconds?.toFixed(1)}s
        </span>
      </div>
    );
  }
  if (event.type === "scan_error") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-red-950/30 border border-red-900 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-red-400">✗ SCAN ERROR</span>
        <span className="text-red-200 break-all">{event.error}</span>
      </div>
    );
  }

  // ── scan_started ────────────────────────────────────────────────────
  if (event.type === "scan_started") {
    return (
      <div className="flex gap-2 leading-relaxed py-1 border-b border-zinc-800 mb-2">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-blue-400">▶ SCAN STARTED</span>
        <span className="text-zinc-300 break-all">target: {event.target || ""}</span>
      </div>
    );
  }

  // ── Generic scan_progress fallback ─────────────────────────────────
  // Renders with agent_name prefix + thought text. Used by the older
  // emit_scan_progress call sites that haven't been migrated to granular
  // events yet.
  if (event.type === "scan_progress") {
    const prefix = event.agent_name ? `[${event.agent_name}]` : "[orchestrator]";
    let content = event.thought || "";
    if (event.tool_name) content += ` | tool=${event.tool_name}`;
    if (event.observation) content += ` | obs=${event.observation.slice(0, 100)}`;
    return (
      <div className="flex gap-2 leading-relaxed">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-zinc-500">{prefix}</span>
        <span className="text-zinc-300 break-all">{content}</span>
      </div>
    );
  }

  // ── HITL ────────────────────────────────────────────────────────────
  if (event.type === "hitl_approval_required") {
    return (
      <div className="flex gap-2 leading-relaxed my-1 p-2 bg-purple-950/30 border border-purple-900 rounded">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className="shrink-0 text-purple-400">🛡 HITL required</span>
        <span className="text-purple-200 break-all">
          {event.tool_name} @ {event.target} ({event.predicted_impact})
        </span>
      </div>
    );
  }
  if (event.type === "hitl_decision_made") {
    const isApprove = event.decision === "approve";
    return (
      <div className="flex gap-2 leading-relaxed my-1">
        <span className="text-zinc-700 shrink-0">{time || "—"}</span>
        <span className={`shrink-0 ${isApprove ? "text-emerald-400" : "text-red-400"}`}>
          {isApprove ? "✓" : "✗"} HITL {event.decision}
        </span>
        <span className="text-zinc-300 break-all">
          by {event.decided_by} — {event.tool_name} @ {event.target}
        </span>
      </div>
    );
  }

  // Unknown event — render raw
  return (
    <div className="flex gap-2 leading-relaxed text-zinc-600">
      <span className="text-zinc-700 shrink-0">{time || "—"}</span>
      <span className="shrink-0">{event.type}</span>
      <span className="break-all">{JSON.stringify(event).slice(0, 200)}</span>
    </div>
  );
}

// ── ToolCard ────────────────────────────────────────────────────────────────
// Renders a tool_call_started event with status dot. Updates its status
// when the matching tool_call_completed event arrives (the EventLine
// parent passes fresh toolCallStates on each render).

function ToolCard({
  state,
  time,
  expanded,
  onToggle,
}: {
  state: ToolCallState;
  time: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const statusColor =
    state.status === "running" ? "text-amber-400 animate-pulse" :
    state.status === "completed" ? "text-emerald-400" :
    "text-red-400";
  const statusIcon =
    state.status === "running" ? "⏳" :
    state.status === "completed" ? "✓" :
    "✗";
  const bgClass =
    state.status === "running" ? "bg-amber-950/20 border-amber-900/50" :
    state.status === "completed" ? "bg-emerald-950/20 border-emerald-900/50" :
    "bg-red-950/20 border-red-900/50";

  return (
    <div className={`my-1 p-2 border rounded ${bgClass}`}>
      <div
        className="flex items-center gap-2 cursor-pointer"
        onClick={onToggle}
      >
        <span className="text-zinc-700 shrink-0">{time}</span>
        <span className={`shrink-0 ${statusColor}`}>🔧 {statusIcon}</span>
        <span className="text-zinc-200 font-semibold shrink-0">{state.tool_name}</span>
        {state.agent_name && (
          <span className="text-zinc-500 text-[10px] shrink-0">[{state.agent_name}]</span>
        )}
        {state.iteration && (
          <span className="text-zinc-600 text-[10px] shrink-0">iter {state.iteration}</span>
        )}
        {state.status === "running" && state.elapsed_seconds !== undefined && (
          <span className="text-amber-400/80 text-[10px] shrink-0">
            running {state.elapsed_seconds}s
          </span>
        )}
        {state.args_preview && (
          <span className="text-zinc-500 truncate flex-1">({state.args_preview})</span>
        )}
        {expanded ? <ChevronDown className="w-3 h-3 ml-auto shrink-0" /> : <ChevronRight className="w-3 h-3 ml-auto shrink-0" />}
      </div>
      {expanded && (
        <div className="mt-2 pl-6 text-[11px] space-y-1">
          {state.args_preview && (
            <div>
              <span className="text-zinc-600">args:</span>
              <pre className="text-zinc-400 whitespace-pre-wrap break-all mt-1">{state.args_preview}</pre>
            </div>
          )}
          {state.result_preview && (
            <div>
              <span className="text-zinc-600">result:</span>
              <pre className={`whitespace-pre-wrap break-all mt-1 ${state.status === "completed" ? "text-zinc-300" : "text-red-300"}`}>
                {state.result_preview}
              </pre>
            </div>
          )}
          {state.error && (
            <div>
              <span className="text-zinc-600">error:</span>
              <pre className="text-red-300 whitespace-pre-wrap break-all mt-1">{state.error}</pre>
            </div>
          )}
          {state.completed_at && (
            <div className="text-zinc-600">
              completed at {new Date(state.completed_at).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}