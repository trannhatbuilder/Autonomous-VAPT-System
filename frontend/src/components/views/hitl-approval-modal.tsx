"use client";

import * as React from "react";
import { Shield, AlertTriangle, Clock, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { approveHITL, rejectHITL } from "../../lib/api";

/**
 * HITL Approval Modal — W19-FIX3 Phase E1.
 *
 * Triggered when backend emits SSE event `hitl_approval_required` (per master
 * plan §W7 + D25). The system has intercepted a destructive operation
 * (sqlmap --os-shell, metasploit exploit, mimikatz, etc.) and is waiting
 * for user decision.
 *
 * Modes:
 *  - audit_agent (default) → LLM critic decides 2-3s, modal shows result
 *  - human_block            → user must click Approve/Abort (5-min timeout)
 *  - auto_approve (debug)   → no modal (auto-approved server-side)
 *
 * In audit_agent mode, the modal usually appears briefly to show the audit
 * decision + allow override if user disagrees. In human_block mode, the
 * modal blocks until user action OR 5-min timeout (auto-abort).
 */

export interface HITLApproval {
  id: string;
  scan_id: string;
  tool_name: string;
  target: string;
  args: Record<string, unknown> | null;
  predicted_impact: string | null;
  agent_reasoning: string | null;
  kg_confidence: number | null;
  status: "pending" | "approved" | "aborted" | "timeout";
  user_decision: string | null;
  decided_at: string | null;
  expires_at: string | null;
  created_at: string;
}

const HITL_TIMEOUT_SECONDS = 300; // 5 minutes per master plan §W7

export function HITLApprovalModal({
  approval,
  onClose,
}: {
  approval: HITLApproval | null;
  onClose: () => void;
}) {
  const [acting, setActing] = React.useState(false);
  const [remaining, setRemaining] = React.useState(HITL_TIMEOUT_SECONDS);
  const [error, setError] = React.useState<string | null>(null);

  // Countdown timer
  React.useEffect(() => {
    if (!approval || approval.status !== "pending") return;
    setRemaining(HITL_TIMEOUT_SECONDS);
    const interval = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          // Auto-close after timeout — backend handles the auto-abort
          setTimeout(onClose, 500);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [approval, onClose]);

  if (!approval) return null;

  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;
  const timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;

  const argsStr = approval.args
    ? JSON.stringify(approval.args, null, 2)
    : "(no args)";

  const handleApprove = async () => {
    if (acting) return;
    setActing(true);
    setError(null);
    try {
      await approveHITL(approval.id);
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to approve");
    } finally {
      setActing(false);
    }
  };

  const handleAbort = async () => {
    if (acting) return;
    setActing(true);
    setError(null);
    try {
      await rejectHITL(approval.id);
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to abort");
    } finally {
      setActing(false);
    }
  };

  // Detect destructive tool type for icon/color
  const isExploit = ["metasploit", "sqlmap", "mimikatz"].some((t) =>
    approval.tool_name.toLowerCase().includes(t)
  );
  const accentColor = isExploit ? "text-red-400" : "text-amber-400";
  const borderColor = isExploit ? "border-red-900" : "border-amber-900";

  return (
    <Dialog open={!!approval} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className={`bg-zinc-950 ${borderColor} border-2 max-w-2xl`}>
        <DialogHeader>
          <DialogTitle className={`flex items-center gap-2 ${accentColor}`}>
            <Shield className="w-5 h-5" />
            HITL Approval Required — Destructive Operation
          </DialogTitle>
          <DialogDescription className="text-zinc-400">
            The agent is attempting a destructive operation. The system has
            intercepted it for your approval. Timeout will auto-abort in{" "}
            <span className="font-mono text-amber-300 font-bold">{timeStr}</span>.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 max-h-[60vh] overflow-y-auto">
          {/* Tool + target summary */}
          <div className="bg-zinc-900/50 border border-zinc-800 rounded p-3">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className={`w-4 h-4 ${accentColor}`} />
              <span className="font-mono text-sm font-semibold text-zinc-100">
                {approval.tool_name}
              </span>
              <Badge variant="outline" className="text-zinc-400 border-zinc-700">
                destructive
              </Badge>
            </div>
            <div className="text-xs text-zinc-400 font-mono">
              <span className="text-zinc-500">target:</span>{" "}
              <span className="text-zinc-200">{approval.target}</span>
            </div>
          </div>

          {/* Predicted impact */}
          {approval.predicted_impact && (
            <div>
              <div className="text-xs text-zinc-500 mb-1">Predicted Impact</div>
              <div className="text-sm text-amber-200 bg-amber-950/20 border border-amber-900/50 rounded p-2">
                {approval.predicted_impact}
              </div>
            </div>
          )}

          {/* Agent reasoning */}
          {approval.agent_reasoning && (
            <div>
              <div className="text-xs text-zinc-500 mb-1">Agent Reasoning</div>
              <div className="text-xs text-zinc-300 bg-zinc-900/50 border border-zinc-800 rounded p-2 font-mono whitespace-pre-wrap">
                {approval.agent_reasoning}
              </div>
            </div>
          )}

          {/* KG confidence */}
          {approval.kg_confidence !== null && approval.kg_confidence !== undefined && (
            <div className="flex items-center gap-2 text-xs">
              <span className="text-zinc-500">KG Confidence:</span>
              <Badge
                className={
                  approval.kg_confidence >= 0.7
                    ? "bg-emerald-700 text-zinc-50"
                    : approval.kg_confidence >= 0.4
                    ? "bg-amber-700 text-zinc-50"
                    : "bg-red-700 text-zinc-50"
                }
              >
                {(approval.kg_confidence * 100).toFixed(0)}%
              </Badge>
            </div>
          )}

          {/* Full args (pretty-printed) */}
          <div>
            <div className="text-xs text-zinc-500 mb-1">Tool Args</div>
            <pre className="text-xs text-zinc-300 bg-zinc-900/70 border border-zinc-800 rounded p-2 font-mono overflow-x-auto">
              {argsStr}
            </pre>
          </div>

          {/* Countdown */}
          <div className="flex items-center gap-2 text-xs text-zinc-400">
            <Clock className="w-3 h-3" />
            <span>Auto-abort in</span>
            <span
              className={`font-mono font-bold ${
                remaining < 60 ? "text-red-400 animate-pulse" : "text-amber-300"
              }`}
            >
              {timeStr}
            </span>
          </div>

          {error && (
            <div className="text-xs text-red-400 bg-red-950/30 border border-red-900 rounded p-2">
              {error}
            </div>
          )}
        </div>

        <DialogFooter className="gap-2">
          <Button
            variant="destructive"
            onClick={handleAbort}
            disabled={acting}
            className="bg-red-700 hover:bg-red-600"
          >
            {acting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Abort
          </Button>
          <Button
            onClick={handleApprove}
            disabled={acting}
            className="bg-emerald-600 hover:bg-emerald-500 text-zinc-50"
          >
            {acting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Approve
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}