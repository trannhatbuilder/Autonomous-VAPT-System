"use client";

import { useEffect, useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import {
  ArrowLeft,
  Bug,
  Shield,
  Globe,
  CheckCircle2,
  XCircle,
  Terminal,
  FileText,
  Lightbulb,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { getFinding, type ProcessDetailRow } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

/**
 * Finding Detail View (Phase L v1+v2) — W19-FIX4 + W19-FIX5.
 *
 * Displays full detail for a single finding, following the EVVO report
 * template structure:
 *   1. Header (title, severity, CVSS, URL)
 *   2. Description (LLM-generated + "Independent verification" line)
 *   3. Proof of Concept (command + key output; falls back to evidence
 *      layer=detection, the only layer the agent actually writes)
 *   4. Evidence (command + raw_output from evidence layer=detection)
 *   5. Remediation (numbered steps from finding.remediation)
 *   6. AI Confidence (overall % + 4-dim breakdown bars + internet cross-check)
 *
 * W19-FIX5 Phase L v2 (EVVO source arrived — full implementation):
 *   - 4-dim Confidence breakdown (Evidence 35% / Reasoning 25% /
 *     Verification 30% / Historical 10%) — bars rendered from
 *     finding.explanation.confidence_breakdown
 *   - Why we're confident (markdown from finding.explanation.why_confident)
 *   - How to Disprove (per vuln_type from finding.explanation.how_to_disprove)
 *   - Recommended Next Steps (list from finding.explanation.recommended_next_steps)
 */

interface FindingDetailProps {
  findingId: string;
  findingData?: any;  // Phase L v1: passed from report-history-view (no API fetch)
  onBack: () => void;
}

// W19-FIX5 Phase L v2: explanation data from backend
// (app.harness.explainability.FindingExplanation.to_dict())
interface ConfidenceBreakdown {
  component: string;
  score: number;
  weight: number;
  description: string;
}

interface FindingExplanation {
  confidence_breakdown: ConfidenceBreakdown[];
  overall_confidence: number;
  why_confident: string;
  how_to_disprove: string;
  recommended_next_steps: string[];
}

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
  // Phase 2 redesign: standards mapping badges (rendered in sidebar).
  // The backend returns these fields on GET /api/findings/{id}.
  cwe_id?: string | null;
  cve_id?: string | null;
  wstg_test_id?: string | null;
  mitre_attack_technique?: string | null;
  mitre_attack_tactic?: string | null;
  auditor_verdict?: string | null;
  confidence_score?: number;
  exploit_method?: string | null;
  poc_command?: string | null;
  // Set by the backend when tool_bridge substituted a {{URL}}/<target>
  // placeholder in the command/evidence with the finding's real target.
  placeholder_repaired?: boolean;
  metadata_json?: {
    poc?: {
      status?: string;
      command?: string | null;
      placeholder_repaired?: boolean;
    };
  } | null;
  internet_verified?: boolean;
  internet_verification?: {
    confirmed: boolean;
    confidence: number;
    references: string[];
    summary: string;
  } | null;
  // W19-FIX5 Phase L v2: explanation data (4-dim + Why/How/Next Steps)
  explanation?: FindingExplanation | null;
  evidence?: Array<{
    layer: string;
    raw_output: string;
    tool_used: string;
    custody_seal: string;
    evidence_hash: string;
    captured_at: string;
    spill_path: string | null;
  }>;
  created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-700 text-white! dark:bg-red-900",
  high: "bg-orange-700 text-white! dark:bg-orange-900",
  medium: "bg-amber-700 text-white! dark:bg-amber-900",
  low: "bg-blue-700 text-white! dark:bg-blue-900",
  info: "bg-zinc-700 text-white! dark:bg-zinc-800",
};

export function FindingDetailView({ findingId, findingData, onBack }: FindingDetailProps) {
  // Seed state from the `findingData` prop via a lazy initializer so we never
  // call setState synchronously inside the effect body (react-hooks/
  // set-state-in-effect). The prop payload is only used when there is no
  // findingId to fetch from; it carries NO evidence rows.
  const [finding, setFinding] = useState<Finding | null>(() =>
    !findingId && findingData ? (findingData as Finding) : null
  );
  const [loading, setLoading] = useState(() => Boolean(findingId) || !findingData);
  const { toast } = useToast();

  useEffect(() => {
    // Phase L v2: fetch finding by ID from GET /api/findings/{id} endpoint.
    // This returns the finding WITH its evidence rows joined so the UI can
    // display the PoC (tool + command + raw output).
    //
    // CRITICAL: must use the `getFinding` api helper (which goes through
    // `apiFetch` → adds `Authorization: Bearer <jwt>` + the correct base URL).
    // A previous raw `fetch(...)` sent no auth header (the JWT lives in
    // localStorage, not a cookie) → 401 → silent fallback to the list payload,
    // which carries NO evidence → the PoC section rendered empty.
    // Falls back to the findingData prop if the API call fails.
    if (!findingId) return;

    let cancelled = false;

    (async () => {
      try {
        const data = await getFinding(findingId);
        if (!cancelled && data) {
          setFinding(data as unknown as Finding);
        }
      } catch (err) {
        console.error(`Failed to fetch finding ${findingId}`, err);
        // Fallback to findingData prop (passed from report-history-view).
        // NOTE: that payload has no `evidence` rows — the PoC section will
        // show "No evidence recorded" if this fallback is ever used.
        if (!cancelled && findingData) {
          setFinding(findingData as Finding);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [findingData, findingId, toast]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
      </div>
    );
  }

  if (!finding) {
    return (
      <Card className="bg-white dark:bg-zinc-900/40 border-zinc-200 dark:border-zinc-800">
        <CardContent className="pt-6 text-center text-zinc-500 dark:text-zinc-400">
          Finding not loaded. This v1 view expects the finding object passed via parent state.
          <br />
          <Button variant="outline" onClick={onBack} className="mt-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to report history
          </Button>
        </CardContent>
      </Card>
    );
  }

  const sev = (finding.severity || "info").toLowerCase();
  const detectionEvidence = finding.evidence?.filter((e) => e.layer === "detection") || [];
  const exploitationEvidence = finding.evidence?.filter((e) => e.layer === "exploitation") || [];
  const auditEvidence = finding.evidence?.filter((e) => e.layer === "audit") || [];
  // FIX: the agent records ALL evidence at layer="detection"
  // (tool_bridge._record_vulnerability is the only writer), so the
  // exploitation layer never exists. The PoC IS the detection evidence:
  // command line + tool output. Fall back to it when exploitation is empty.
  const pocEvidence = exploitationEvidence.length ? exploitationEvidence : detectionEvidence;

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-4">
      {/* Top bar: back button */}
      <div className="flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onBack} className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to report history
        </Button>
        {/* Quick-jump anchors for long pages — clickable section navigation */}
        <nav className="hidden md:flex items-center gap-1 text-xs text-zinc-500">
          {[
            { id: "description", label: "Description", icon: FileText },
            { id: "poc", label: "PoC", icon: Terminal },
            { id: "evidence", label: "Evidence", icon: FileText },
            { id: "remediation", label: "Remediation", icon: Lightbulb },
            { id: "confidence", label: "Confidence", icon: Shield },
          ].map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="inline-flex items-center gap-1 px-2 py-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-zinc-100"
            >
              <s.icon className="w-3 h-3" />
              {s.label}
            </a>
          ))}
        </nav>
      </div>

      {/* ── 2-column layout: sticky sidebar + scrollable main ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
        {/* ── LEFT: Sticky sidebar with key metadata ── */}
        <aside className="lg:sticky lg:top-4 space-y-3">
          {/* Header card — severity, name, location */}
          <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <Bug className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />
                <Badge className={SEVERITY_COLORS[sev]}>{finding.severity}</Badge>
                {finding.verified && (
                  <Badge variant="outline" className="text-emerald-700 dark:text-emerald-300 border-emerald-700 dark:border-emerald-700">
                    <CheckCircle2 className="w-3 h-3 mr-1" /> Verified
                  </Badge>
                )}
                {finding.auditor_verdict && (
                  <Badge variant="outline" className="text-zinc-600 dark:text-zinc-300 border-zinc-500 dark:border-zinc-700">
                    {finding.auditor_verdict}
                  </Badge>
                )}
              </div>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-lg leading-tight">
                {finding.name}
              </CardTitle>
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 font-mono break-all">
                {finding.location}
              </div>
            </CardHeader>
          </Card>

          {/* CVSS + metadata quick-grid */}
          <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
            <CardContent className="p-4 space-y-3">
              {/* Big CVSS score */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[10px] text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                    CVSS Score
                  </div>
                  <div className={`text-3xl font-bold ${
                    (finding.cvss_score || 0) >= 9 ? "text-red-600 dark:text-red-400" :
                    (finding.cvss_score || 0) >= 7 ? "text-orange-600 dark:text-orange-400" :
                    (finding.cvss_score || 0) >= 4 ? "text-amber-600 dark:text-amber-400" :
                    "text-blue-600 dark:text-blue-400"
                  }`}>
                    {finding.cvss_score?.toFixed(1)}
                  </div>
                </div>
                {/* Compact CVSS bar */}
                <div className="w-24 h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      (finding.cvss_score || 0) >= 9 ? "bg-red-500" :
                      (finding.cvss_score || 0) >= 7 ? "bg-orange-500" :
                      (finding.cvss_score || 0) >= 4 ? "bg-amber-500" :
                      "bg-blue-500"
                    }`}
                    style={{ width: `${Math.min((finding.cvss_score || 0) * 10, 100)}%` }}
                  />
                </div>
              </div>
              {finding.cvss_vector && (
                <div className="text-[10px] text-zinc-500 dark:text-zinc-400 font-mono break-all border-t border-zinc-100 dark:border-zinc-800 pt-2">
                  {finding.cvss_vector}
                </div>
              )}

              {/* Metadata grid */}
              <div className="grid grid-cols-2 gap-2 text-xs border-t border-zinc-100 dark:border-zinc-800 pt-3">
                <SidebarMetaItem label="Type" value={finding.vuln_type} mono />
                <SidebarMetaItem label="PoC" value={finding.poc_status} />
                <SidebarMetaItem label="Method" value={finding.exploit_method || "—"} mono />
                <SidebarMetaItem label="Detected" value={finding.created_at ? new Date(finding.created_at).toLocaleDateString() : "—"} small />
              </div>

              {/* Standards mapping badges */}
              {(finding.cwe_id || finding.cve_id || finding.wstg_test_id || finding.mitre_attack_technique) && (
                <div className="flex flex-wrap gap-1 border-t border-zinc-100 dark:border-zinc-800 pt-3">
                  {finding.cve_id && (
                    <Badge variant="outline" className="text-[10px] text-red-700 dark:text-red-300 border-red-700 dark:border-red-900 font-mono">
                      {finding.cve_id}
                    </Badge>
                  )}
                  {finding.cwe_id && (
                    <Badge variant="outline" className="text-[10px] text-orange-700 dark:text-orange-300 border-orange-700 dark:border-orange-900 font-mono">
                      {finding.cwe_id}
                    </Badge>
                  )}
                  {finding.wstg_test_id && (
                    <Badge variant="outline" className="text-[10px] text-blue-700 dark:text-blue-300 border-blue-700 dark:border-blue-900 font-mono">
                      {finding.wstg_test_id}
                    </Badge>
                  )}
                  {finding.mitre_attack_technique && (
                    <Badge variant="outline" className="text-[10px] text-zinc-700 dark:text-zinc-300 border-zinc-500 dark:border-zinc-700 font-mono">
                      {finding.mitre_attack_technique}
                    </Badge>
                  )}
                </div>
              )}

              {/* Custody chain quick-link */}
              <a
                href={`/api/findings/${finding.id}/custody-chain`}
                target="_blank"
                rel="noopener noreferrer"
                className="block text-[10px] text-emerald-700 dark:text-emerald-400 hover:underline border-t border-zinc-100 dark:border-zinc-800 pt-2"
              >
                → View custody chain (HMAC-verified)
              </a>
            </CardContent>
          </Card>
        </aside>

        {/* ── RIGHT: scrollable main content with sections ── */}
        <div className="space-y-4 min-w-0">
          {/* ── 2. Description ── */}
          <Card id="description" className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 scroll-mt-4">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                <FileText className="w-4 h-4 text-zinc-500" />
                Description
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
                {finding.description || "(no description)"}
              </p>
              {finding.vuln_type && (
                <div className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
                  <span className="uppercase tracking-wider">Vulnerability type:</span>{" "}
                  <span className="font-mono">{finding.vuln_type}</span>
                </div>
              )}
              {finding.exploit_method && (
                <div className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  <span className="uppercase tracking-wider">Exploit method:</span>{" "}
                  <span className="font-mono">{finding.exploit_method}</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* ── 3. Proof of Concept ── */}
          <Card id="poc" className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 scroll-mt-4">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                <Terminal className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                Proof of Concept
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {pocEvidence.length === 0 ? (
                <div className="text-sm text-zinc-500 dark:text-zinc-400 italic">
                  No evidence recorded for this finding — treat as unconfirmed.
                  {finding.poc_status === "successful"
                    ? " PoC status: successful."
                    : finding.poc_status === "attempted"
                    ? " PoC status: attempted (inconclusive)."
                    : " PoC status: not_attempted."}
                </div>
              ) : (
                <>
                  {/* PoC verified / recorded badge — derived from real flags */}
                  {finding.verified && finding.poc_status === "successful" ? (
                    <div className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                      ✓ PoC VERIFIED
                      <span className="font-normal text-zinc-500 dark:text-zinc-400">
                        {" "}— tool output below confirms the vulnerability
                      </span>
                    </div>
                  ) : finding.poc_status === "successful" ? (
                    <div className="text-sm font-semibold text-orange-600 dark:text-orange-400">
                      ◐ PoC RECORDED (pending verification)
                    </div>
                  ) : null}

                  {/* Placeholder-repair disclosure — the stored evidence had a
                      {{URL}}/<target> placeholder that the backend replaced with
                      the finding's target. Warn the reader to verify manually. */}
                  {(finding.placeholder_repaired ||
                    finding.metadata_json?.poc?.placeholder_repaired) && (
                    <div className="text-xs text-amber-600 dark:text-amber-400">
                      ⚠ Placeholder in the original evidence was auto-replaced with the
                      finding&apos;s URL. Verify the command manually before re-running it.
                    </div>
                  )}

                  {pocEvidence.map((ev, i) => {
                    // Command line — priority: "$ ..." in raw evidence → the exact
                    // command from metadata_json.poc.command → exploit_method.
                    let commandLine: string | null = null;
                    const raw = ev.raw_output || "";
                    const [firstLine, ...restLines] = raw.split("\n");
                    let resultBody = raw;
                    if (firstLine?.trimStart().startsWith("$ ")) {
                      commandLine = firstLine.trimStart().slice(2);
                      resultBody = restLines.join("\n").replace(/^\n+/, "");
                    } else if (finding.poc_command) {
                      commandLine = finding.poc_command;
                    } else if (
                      finding.exploit_method &&
                      finding.exploit_method !== "agent" &&
                      finding.exploit_method !== "pipeline-injected"
                    ) {
                      commandLine = finding.exploit_method;
                    }
                    return (
                      <div key={i} className="space-y-2">
                        <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                          Tool used: {ev.tool_used}
                        </div>
                        {commandLine && (
                          <div>
                            <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                              Command
                            </div>
                            <pre className="text-xs text-zinc-300 bg-zinc-950 border border-zinc-800 rounded p-3 font-mono overflow-x-auto whitespace-pre-wrap break-all">
                              $ {commandLine}
                            </pre>
                          </div>
                        )}
                        <div>
                          <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                            Result
                          </div>
                          <pre className="text-xs text-zinc-300 bg-zinc-950 border border-zinc-800 rounded p-3 font-mono overflow-x-auto max-h-80 overflow-y-auto">
                            {resultBody.slice(0, 5000)}
                            {resultBody.length > 5000 && "\n... [truncated — full output in evidence chain]"}
                          </pre>
                        </div>
                        <div className="text-[10px] text-zinc-500 dark:text-zinc-500 font-mono">
                          evidence_hash: {(ev.evidence_hash || "").slice(0, 32)}... · captured: {ev.captured_at}
                        </div>
                      </div>
                    );
                  })}
                  {pocEvidence.length > 1 && (
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 italic">
                      + {pocEvidence.length - 1} additional evidence entry(ies) in the
                      evidence chain for this finding.
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* ── 4. Evidence (detection layer) ── */}
          {/* Hidden when the PoC card above already rendered the detection rows
              (i.e. no exploitation-layer evidence exists) to avoid duplication. */}
          {exploitationEvidence.length > 0 && (
          <Card id="evidence" className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 scroll-mt-4">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                <FileText className="w-4 h-4 text-zinc-500" />
                Evidence (Detection)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {detectionEvidence.length === 0 ? (
                <div className="text-sm text-zinc-500 dark:text-zinc-400 italic">
                  No detection evidence recorded.
                </div>
              ) : (
                detectionEvidence.map((ev, i) => (
                  <div key={i} className="space-y-2">
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                      Tool: {ev.tool_used}
                    </div>
                    <pre className="text-xs text-zinc-300 bg-zinc-950 border border-zinc-800 rounded p-3 font-mono overflow-x-auto max-h-80 overflow-y-auto">
                      {ev.raw_output.slice(0, 5000)}
                      {ev.raw_output.length > 5000 && "\n... [truncated]"}
                    </pre>
                    <div className="text-[10px] text-zinc-500 dark:text-zinc-500 font-mono">
                      hash: {ev.evidence_hash.slice(0, 32)}... · seal: {ev.custody_seal.slice(0, 32)}...
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
          )}

          {auditEvidence.length > 0 && (
            <div className="text-xs text-zinc-500 dark:text-zinc-400 italic px-1">
              + {auditEvidence.length} audit-layer evidence entries (omitted from view).
            </div>
          )}

          {/* ── 5. Remediation ── */}
          {finding.remediation && (
            <Card id="remediation" className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 scroll-mt-4">
              <CardHeader>
                <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                  <Lightbulb className="w-4 h-4 text-amber-500" />
                  Remediation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
                  {finding.remediation}
                </p>
              </CardContent>
            </Card>
          )}

          {/* ── 6. AI Confidence ── */}
          <Card id="confidence" className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 scroll-mt-4">
            <CardHeader>
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                <Shield className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                AI Confidence
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Overall confidence */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                    Overall confidence
                  </span>
                  <span className="text-lg font-bold text-zinc-900 dark:text-zinc-50">
                    {((finding.confidence_score || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      (finding.confidence_score || 0) >= 0.7
                        ? "bg-emerald-500"
                        : (finding.confidence_score || 0) >= 0.4
                        ? "bg-amber-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${(finding.confidence_score || 0) * 100}%` }}
                  />
                </div>
              </div>

              {/* W19-FIX5 Phase L v2: 4-dim confidence breakdown bars */}
              {finding.explanation?.confidence_breakdown && finding.explanation.confidence_breakdown.length > 0 ? (
                <div className="border-t border-zinc-200 dark:border-zinc-800 pt-3 space-y-2">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                    Confidence Breakdown
                  </div>
                  {finding.explanation.confidence_breakdown.map((dim) => (
                    <div key={dim.component} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-zinc-700 dark:text-zinc-300 font-medium">
                          {dim.component}
                          <span className="text-zinc-500 dark:text-zinc-500 ml-1">
                            ({(dim.weight * 100).toFixed(0)}% wt)
                          </span>
                        </span>
                        <span className="text-zinc-700 dark:text-zinc-200 font-mono">
                          {(dim.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            dim.score >= 0.7 ? "bg-emerald-500" :
                            dim.score >= 0.4 ? "bg-amber-500" :
                            "bg-red-500"
                          }`}
                          style={{ width: `${dim.score * 100}%` }}
                        />
                      </div>
                      <div className="text-[10px] text-zinc-500 dark:text-zinc-500">
                        {dim.description}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-zinc-500 dark:text-zinc-400 italic border-t border-zinc-200 dark:border-zinc-800 pt-3">
                  ⏳ 4-dim confidence breakdown not available for this finding (run a new scan after applying W19-FIX5 backend).
                </div>
              )}

              {/* Internet cross-check */}
              {finding.internet_verification && (
                <div className="border border-zinc-200 dark:border-zinc-800 rounded p-3 space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    {finding.internet_verified ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    ) : (
                      <XCircle className="w-4 h-4 text-zinc-500" />
                    )}
                    <span className={finding.internet_verified ? "text-emerald-700 dark:text-emerald-300" : "text-zinc-500"}>
                      {finding.internet_verified
                        ? "Confirmed by internet sources"
                        : "No consensus / insufficient references"}
                    </span>
                    <Badge variant="outline" className="text-xs ml-auto">
                      confidence: {(finding.internet_verification.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  {finding.internet_verification.summary && (
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 italic">
                      {finding.internet_verification.summary}
                    </p>
                  )}
                  {finding.internet_verification.references && finding.internet_verification.references.length > 0 && (
                    <div className="space-y-1">
                      <div className="text-xs text-zinc-500 dark:text-zinc-400">References:</div>
                      {finding.internet_verification.references.map((ref, i) => (
                        <a
                          key={i}
                          href={ref}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block text-xs text-blue-500 hover:text-blue-300 truncate font-mono"
                        >
                          → {ref}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* ── W19-FIX5 Phase L v2: Why we're confident + How to Disprove + Next Steps ── */}
          {finding.explanation && (
            <>
              {/* Why we're confident */}
              {finding.explanation.why_confident && (
                <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
                  <CardHeader>
                    <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                      <Shield className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                      Why we&apos;re confident
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-xs text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed">
                      {finding.explanation.why_confident}
                    </pre>
                  </CardContent>
                </Card>
              )}

              {/* How to Disprove */}
              {finding.explanation.how_to_disprove && (
                <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
                  <CardHeader>
                    <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-500" />
                      How to Disprove
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-xs text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded p-2">
                      {finding.explanation.how_to_disprove}
                    </pre>
                  </CardContent>
                </Card>
              )}

              {/* Recommended Next Steps */}
              {finding.explanation.recommended_next_steps && finding.explanation.recommended_next_steps.length > 0 && (
                <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
                  <CardHeader>
                    <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
                      <Lightbulb className="w-4 h-4 text-amber-500" />
                      Recommended Next Steps
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ol className="list-decimal list-inside space-y-1 text-sm text-zinc-700 dark:text-zinc-300">
                      {finding.explanation.recommended_next_steps.map((step, i) => (
                        <li key={i} className="leading-relaxed">{step}</li>
                      ))}
                    </ol>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Helper component for sidebar metadata items ──

function SidebarMetaItem({
  label,
  value,
  mono,
  small,
}: {
  label: string;
  value: string;
  mono?: boolean;
  small?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] text-zinc-500 dark:text-zinc-400 uppercase tracking-wider truncate">{label}</div>
      <div className={`text-zinc-900 dark:text-zinc-100 truncate ${mono ? "font-mono" : ""} ${small ? "text-[10px]" : "text-xs"}`}>
        {value}
      </div>
    </div>
  );
}