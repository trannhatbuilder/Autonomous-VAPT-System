'use client';

import { useEffect, useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Loader2, Bug, Filter, ShieldAlert, Download } from "lucide-react";
import { getFindings, exportVulnerabilities, downloadTextFile } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

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
  created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-700 text-red-50",
  high: "bg-orange-700 text-orange-50",
  medium: "bg-amber-700 text-amber-50",
  low: "bg-blue-700 text-blue-50",
  info: "bg-zinc-700 text-zinc-100",
};

export function FindingsView() {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scanIdFilter, setScanIdFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(null);
  const [exporting, setExporting] = useState(false);
  const { toast } = useToast();

  const loadFindings = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: any = { limit: 100 };
      if (scanIdFilter) params.scan_id = scanIdFilter;
      if (severityFilter !== "all") params.severity = severityFilter;
      const data = await getFindings(params);
      setFindings(data.findings || []);
      setTotal(data.total || 0);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadFindings(); }, []);

  const handleFilterApply = () => {
    loadFindings();
  };

  // Phase F6/F7: Markdown export (CyberStrikeAI pattern — on-demand,
  // not auto-generated after scan). User clicks button → backend builds
  // Markdown → frontend downloads via Blob + <a download>.
  const handleExportMarkdown = async () => {
    setExporting(true);
    try {
      const params: any = { group_by: "scan", mode: "summary", limit: 1000 };
      if (scanIdFilter) params.scan_id = scanIdFilter;
      if (severityFilter !== "all") params.severity = severityFilter;
      const resp = await exportVulnerabilities(params);
      if (!resp.files || resp.files.length === 0) {
        toast({ title: "No findings to export", description: resp.message || "No findings match the filter.", variant: "destructive" });
        return;
      }
      // Download each file (summary mode = 1 file, split mode = N files)
      for (const file of resp.files) {
        downloadTextFile(file.filename, file.content);
      }
      toast({ title: "Markdown exported", description: `${resp.total} findings in ${resp.files.length} file(s).` });
    } catch (err: any) {
      toast({ title: "Export failed", description: err.message, variant: "destructive" });
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-50 flex items-center gap-2">
            <Bug className="w-6 h-6 text-amber-400" />
          Findings
        </h2>
        <p className="text-sm text-zinc-400 mt-1">
          All vulnerability findings across scans. {total} total.
        </p>
        </div>
        {/* Phase F7: Export Markdown button (CyberStrikeAI pattern — on-demand) */}
        <Button
          onClick={handleExportMarkdown}
          disabled={exporting || total === 0}
          variant="outline"
          className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
        >
          {exporting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
          Export Markdown
        </Button>
      </div>

      {/* Filters */}
      <Card className="bg-zinc-900/60 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-50 text-base flex items-center gap-2">
            <Filter className="w-4 h-4 text-zinc-400" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div className="space-y-2">
            <Label htmlFor="scan_id" className="text-zinc-200">Scan ID</Label>
            <Input
              id="scan_id"
              placeholder="(all scans)"
              value={scanIdFilter}
              onChange={(e) => setScanIdFilter(e.target.value)}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-zinc-200">Severity</Label>
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-zinc-900 border-zinc-800">
                <SelectItem value="all">All severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button
            onClick={handleFilterApply}
            disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-500 text-zinc-50"
          >
            {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Filter className="w-4 h-4 mr-2" />}
            Apply
          </Button>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <Card className="bg-red-950/40 border-red-900">
          <CardContent className="pt-6 text-red-200">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 shrink-0" />
              {error}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading */}
      {loading && !error && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
        </div>
      )}

      {/* Findings list */}
      {!loading && !error && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* List */}
          <div className="lg:col-span-1 space-y-2">
            {findings.length === 0 ? (
              <Card className="bg-zinc-900/40 border-zinc-800">
                <CardContent className="pt-6 text-center text-zinc-500">
                  No findings yet. Start a scan to see results.
                </CardContent>
              </Card>
            ) : (
              findings.map((f) => (
                <Card
                  key={f.id}
                  className={`cursor-pointer hover:border-zinc-700 transition-colors ${
                    selectedFinding?.id === f.id ? "border-emerald-600 bg-zinc-900" : "bg-zinc-900/40 border-zinc-800"
                  }`}
                  onClick={() => setSelectedFinding(f)}
                >
                  <CardContent className="p-4 space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="text-sm font-medium text-zinc-100 leading-tight">{f.name}</h3>
                      <Badge
                        variant="secondary"
                        className={`shrink-0 ${SEVERITY_COLORS[(f.severity || "info").toLowerCase()] || SEVERITY_COLORS.info}`}
                      >
                        {f.severity}
                      </Badge>
                    </div>
                    <div className="text-xs text-zinc-500 font-mono truncate">
                      {f.vuln_type} @ {f.location}
                    </div>
                    <div className="text-xs text-zinc-600">
                      CVSS {f.cvss_score} | {f.verified ? "✓ verified" : "unverified"} | {f.poc_status}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          {/* Detail */}
          <div className="lg:col-span-2">
            {selectedFinding ? (
              <Card className="bg-zinc-900/60 border-zinc-800 sticky top-4">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <CardTitle className="text-zinc-50 text-lg">{selectedFinding.name}</CardTitle>
                      <CardDescription className="text-zinc-400 mt-1 font-mono text-xs">
                        {selectedFinding.id}
                      </CardDescription>
                    </div>
                    <Badge
                      className={`shrink-0 ${SEVERITY_COLORS[(selectedFinding.severity || "info").toLowerCase()] || SEVERITY_COLORS.info}`}
                    >
                      {selectedFinding.severity}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <Field label="Vulnerability type" value={selectedFinding.vuln_type} mono />
                    <Field label="Location" value={selectedFinding.location} mono />
                    <Field label="CVSS score" value={String(selectedFinding.cvss_score)} />
                    <Field label="CVSS vector" value={selectedFinding.cvss_vector || "—"} mono small />
                    <Field label="Verified" value={selectedFinding.verified ? "Yes" : "No"} />
                    <Field label="PoC status" value={selectedFinding.poc_status} />
                    <Field label="Scan ID" value={selectedFinding.scan_id} mono small />
                    <Field label="Detected" value={new Date(selectedFinding.created_at).toLocaleString()} small />
                  </div>

                  <div>
                    <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-2">Description</h4>
                    <p className="text-sm text-zinc-300 whitespace-pre-wrap">{selectedFinding.description}</p>
                  </div>

                  {selectedFinding.remediation && (
                    <div>
                      <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-2">Remediation</h4>
                      <p className="text-sm text-zinc-300 whitespace-pre-wrap">{selectedFinding.remediation}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-zinc-900/40 border-zinc-800">
                <CardContent className="pt-6 text-center text-zinc-500">
                  Select a finding to see details.
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, value, mono, small }: { label: string; value: string; mono?: boolean; small?: boolean }) {
  return (
    <div>
      <div className="text-xs text-zinc-500 uppercase tracking-wider">{label}</div>
      <div className={`text-zinc-200 ${mono ? "font-mono" : ""} ${small ? "text-xs" : "text-sm"} break-all`}>
        {value}
      </div>
    </div>
  );
}
