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
import { getFindings, type ProcessDetailRow } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

/**
 * Finding Detail View (Phase L v1+v2) — W19-FIX4 + W19-FIX5.
 *
 * Displays full detail for a single finding, following the EVVO report
 * template structure:
 *   1. Header (title, severity, CVSS, URL)
 *   2. Description (LLM-generated + "Independent verification" line)
 *   3. Proof of Concept (command + key output from evidence layer=exploitation)
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
  auditor_verdict?: string | null;
  confidence_score?: number;
  exploit_method?: string | null;
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
  critical: "bg-red-700 text-red-50 dark:bg-red-900 dark:text-red-100",
  high: "bg-orange-700 text-orange-50 dark:bg-orange-900 dark:text-orange-100",
  medium: "bg-amber-700 text-amber-50 dark:bg-amber-900 dark:text-amber-100",
  low: "bg-blue-700 text-blue-50 dark:bg-blue-900 dark:text-blue-100",
  info: "bg-zinc-700 text-zinc-100 dark:bg-zinc-800 dark:text-zinc-100",
};

export function FindingDetailView({ findingId, findingData, onBack }: FindingDetailProps) {
  const [finding, setFinding] = useState<Finding | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    // Phase L v1: use the finding object passed via props (no API fetch —
    // backend doesn't yet have GET /api/findings/{id} endpoint).
    if (findingData) {
      setFinding(findingData as Finding);
      setLoading(false);
      return;
    }
    // Fallback: if no findingData, show "not available" message.
    // Phase L v2 will fetch from GET /api/findings/{id} when backend adds it.
    setLoading(false);
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

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-4">
      {/* Back button */}
      <Button variant="ghost" onClick={onBack} className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100">
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to report history
      </Button>

      {/* ── 1. Header ── */}
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <Bug className="w-5 h-5 text-amber-600 dark:text-amber-400" />
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
              <CardTitle className="text-zinc-900 dark:text-zinc-50 text-xl">
                {finding.name}
              </CardTitle>
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 font-mono break-all">
                {finding.location}
              </div>
            </div>
            <div className="text-right shrink-0">
              <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                CVSS Score
              </div>
              <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
                {finding.cvss_score?.toFixed(1)}
              </div>
              {finding.cvss_vector && (
                <div className="text-[10px] text-zinc-500 dark:text-zinc-400 font-mono mt-1 max-w-[200px] break-all">
                  {finding.cvss_vector}
                </div>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* ── 2. Description ── */}
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
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
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-900 dark:text-zinc-50 text-base flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            Proof of Concept
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {exploitationEvidence.length === 0 ? (
            <div className="text-sm text-zinc-500 dark:text-zinc-400 italic">
              No exploitation evidence recorded for this finding.
              {finding.poc_status === "successful"
                ? " PoC status: successful (exploit chain recorded)."
                : finding.poc_status === "attempted"
                ? " PoC status: attempted (inconclusive)."
                : " PoC status: not_attempted."}
            </div>
          ) : (
            exploitationEvidence.map((ev, i) => (
              <div key={i} className="space-y-2">
                <div className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                  Tool used: {ev.tool_used}
                </div>
                <pre className="text-xs text-zinc-300 bg-zinc-950 border border-zinc-800 rounded p-3 font-mono overflow-x-auto max-h-80 overflow-y-auto">
                  {ev.raw_output.slice(0, 5000)}
                  {ev.raw_output.length > 5000 && "\n... [truncated]"}
                </pre>
                <div className="text-[10px] text-zinc-500 dark:text-zinc-500 font-mono">
                  evidence_hash: {ev.evidence_hash.slice(0, 32)}... · captured: {ev.captured_at}
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      {/* ── 4. Evidence (detection layer) ── */}
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
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
          {auditEvidence.length > 0 && (
            <div className="text-xs text-zinc-500 dark:text-zinc-400 italic mt-2">
              + {auditEvidence.length} audit-layer evidence entries (omitted from view).
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── 5. Remediation ── */}
      {finding.remediation && (
        <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
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
      <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800">
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
                  Why we're confident
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
  );
}