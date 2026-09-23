'use client';

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Input } from "../ui/input";
import { Loader2, Terminal } from "lucide-react";
import { getTools } from "../../lib/api";

interface Tool {
  name: string;
  command: string;
  category: string;
  short_description: string;
  wstg_ids?: string[];
  mitre_attack?: string[];
  safety_class: string;   
  parameters?: string[];
  timeout?: number;
}

const SAFETY_COLORS: Record<string, string> = {
  passive: "bg-blue-900/60 text-blue-200 border-blue-800",
  active: "bg-amber-900/60 text-amber-200 border-amber-800",
  destructive: "bg-red-900/60 text-red-200 border-red-800",
};

export function ToolsView() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    getTools()
      .then((data) => setTools(data.tools || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = tools.filter((t) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      t.name.toLowerCase().includes(q) ||
      t.command.toLowerCase().includes(q) ||
      t.category.toLowerCase().includes(q) ||
      t.short_description.toLowerCase().includes(q)
    );
  });

  const byCategory: Record<string, Tool[]> = filtered.reduce((acc, t) => {
    (acc[t.category] = acc[t.category] || []).push(t);
    return acc;
  }, {} as Record<string, Tool[]>);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-zinc-50 flex items-center gap-2">
          <Terminal className="w-6 h-6 text-zinc-400" />
          Pentest tools
        </h2>
        <p className="text-sm text-zinc-400 mt-1">
          {tools.length} tools registered. Driven by YAML definitions in <code className="text-zinc-500">app/tools/*.yaml</code>.
          Each tool is wrapped as an MCP tool on the FastMCP server.
        </p>
      </div>

      <Input
        placeholder="Search by name, category, command, or description..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600"
      />

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
        </div>
      )}

      {error && (
        <Card className="bg-red-950/40 border-red-900">
          <CardContent className="pt-6 text-red-200">{error}</CardContent>
        </Card>
      )}

      {!loading && !error && Object.entries(byCategory).map(([cat, catTools]) => (
        <div key={cat} className="space-y-3">
          <h3 className="text-sm uppercase tracking-wider text-zinc-500">
            {cat} <span className="text-zinc-600">({catTools.length})</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {catTools.map((t) => (
              <Card key={t.name} className="bg-zinc-900/40 border-zinc-800 hover:border-zinc-700 transition-colors">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-2">
                    <CardTitle className="text-zinc-100 text-sm font-mono">{t.name}</CardTitle>
                    <Badge
                      className={`shrink-0 ${SAFETY_COLORS[t.safety_class] || SAFETY_COLORS.active}`}
                      variant="outline"
                    >
                      {t.safety_class}
                    </Badge>
                  </div>
                  <CardDescription className="text-zinc-500 text-xs font-mono mt-1">{t.command}</CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-xs text-zinc-400 leading-relaxed">{t.short_description}</p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {t.wstg_ids?.map((w) => (
                      <Badge key={w} variant="outline" className="text-[10px] text-zinc-400 border-zinc-700">
                        {w}
                      </Badge>
                    ))}
                    {t.mitre_attack?.map((m) => (
                      <Badge key={m} variant="outline" className="text-[10px] text-zinc-400 border-zinc-700">
                        {m}
                      </Badge>
                    ))}
                  </div>
                  <div className="text-xs text-zinc-600 mt-2">
                    {t.parameters?.length ?? 0} params · timeout {t.timeout ?? 0}s
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
