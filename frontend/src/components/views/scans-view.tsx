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
import { Loader2, Play, Square, RefreshCw, Activity, Terminal } from "lucide-react";
import { startScan, getScanEventsUrl, abortScan } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

interface ScanEvent {
  // Backend emits the event name under "event" (e.g. {"event": "scan_progress"}).
  // Older/other emitters may use "type"; onmessage normalizes it into `type`.
  event?: string;
  type: string;  // scan_started | scan_progress | scan_complete | scan_error | finding_detected | hitl_approval_required
  turn?: number;
  progress?: number;
  thought?: string;
  tool_name?: string;
  observation?: string;
  agent_name?: string;
  finding_id?: string;
  vuln_type?: string;
  severity?: string;
  location?: string;
  error?: string;
  timestamp?: string;
}

export function ScansView() {
  const { toast } = useToast();
  const [target, setTarget] = useState("");
  const [userPrompt, setUserPrompt] = useState("");
  const [starting, setStarting] = useState(false);
  const [activeScanId, setActiveScanId] = useState<string | null>(null);
  const [events, setEvents] = useState<ScanEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>("idle");
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
    setStatus("starting");
    try {
      const result = await startScan(target, userPrompt);
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
          // Backend sends the event name as "event" (e.g. {"event": "scan_progress"}),
          // while the UI (colors, EventLine, completion checks) keys off "type".
          // Normalize once here so every downstream consumer sees `type`.
          const evtName = (raw as any).event || (raw as any).type || "unknown";
          // Filter out heartbeat events (defensive — backend now sends heartbeats
          // as SSE comment frames ": ping\n\n" which don't trigger onmessage,
          // but in case they leak through, don't pollute the events list)
          if (evtName === "heartbeat") return;
          const data: ScanEvent = { ...raw, type: evtName };

          setEvents((prev) => [...prev, data]);
          if (data.progress !== undefined) {
            setProgress(data.progress);
          }
          if (evtName === "scan_complete") {
            console.log("[VAPT-SSE] Scan complete:", data);
            setStatus("completed");
            setProgress(100);
            es.close();
          } else if (evtName === "scan_error") {
            console.error("[VAPT-SSE] Scan error:", data);
            setStatus("error");
            es.close();
          }
        } catch (err) {
          console.warn("[VAPT-SSE] Failed to parse SSE event:", e.data);
        }
      };

      es.onerror = (e: any) => {
        // EventSource auto-reconnects on transient errors.
        // Log details for debugging — readyState tells us what's happening:
        //   0=CONNECTING, 1=OPEN, 2=CLOSED
        const state = es.readyState;
        const stateName = state === 0 ? "CONNECTING" : state === 1 ? "OPEN" : "CLOSED";
        console.warn(`[VAPT-SSE] Error (readyState=${stateName}). Will auto-reconnect if not CLOSED.`);
        // If CLOSED, it means the connection gave up — surface the error to UI
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

            {/* Events stream */}
            <div className="space-y-2 max-h-[500px] overflow-y-auto bg-zinc-950 rounded border border-zinc-800 p-3 font-mono text-xs">
              {events.length === 0 ? (
                <div className="text-zinc-600 italic">Waiting for events...</div>
              ) : (
                events.map((ev, idx) => (
                  <EventLine key={idx} event={ev} />
                ))
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function EventLine({ event }: { event: ScanEvent }) {
  const typeColors: Record<string, string> = {
    scan_started: "text-blue-400",
    scan_progress: "text-zinc-400",
    scan_complete: "text-emerald-400",
    scan_error: "text-red-400",
    finding_detected: "text-amber-400",
    hitl_approval_required: "text-purple-400",
    hitl_decision_made: "text-purple-400",
    heartbeat: "text-zinc-700",
  };
  const typeColor = typeColors[event.type] || "text-zinc-400";
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "";

  let content: string;
  if (event.type === "scan_progress") {
    content = `[${event.agent_name || "?"}] turn=${event.turn || "?"} | ${event.thought || ""}`;
    if (event.tool_name) content += ` | tool=${event.tool_name}`;
    if (event.observation) content += ` | obs=${event.observation.slice(0, 100)}`;
  } else if (event.type === "finding_detected") {
    content = `FINDING [${event.severity}] ${event.vuln_type} @ ${event.location}`;
  } else if (event.type === "scan_complete") {
    content = `Scan complete — ${event.observation || ""}`;
  } else if (event.type === "scan_error") {
    content = `Error: ${event.error || "unknown"}`;
  } else if (event.type === "hitl_approval_required") {
    content = `HITL approval required — waiting for review`;
  } else {
    content = event.observation || event.thought || JSON.stringify(event).slice(0, 200);
  }

  return (
    <div className="flex gap-2 leading-relaxed">
      <span className="text-zinc-700 shrink-0">{time || "—"}</span>
      <span className={`shrink-0 ${typeColor}`}>{event.type}</span>
      <span className="text-zinc-300 break-all">{content}</span>
    </div>
  );
}