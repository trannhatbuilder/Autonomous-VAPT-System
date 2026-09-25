'use client';

import { useEffect, useState, useCallback } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Alert, AlertDescription } from "../ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Slider } from "../ui/slider";
import { Switch } from "../ui/switch";
import { Loader2, Save, Plug, CheckCircle2, AlertCircle, Plus, Trash2, Star, Edit3 } from "lucide-react";
import {
  listChannels,
  createChannel,
  updateChannel,
  deleteChannel,
  testChannelInline,
  testSavedChannel,
  setDefaultChannel,
  type ChannelConfig,
  type ChannelTestResult,
} from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

// ---------- Provider presets (popular base_urls + models) ----------

const PROVIDER_PRESETS: Record<string, { base_url: string; model: string; name: string }> = {
  openai_compatible: { base_url: "https://api.openai.com/v1", model: "gpt-4o-mini", name: "OpenAI" },
  claude: { base_url: "https://api.anthropic.com", model: "claude-3-5-sonnet-20241022", name: "Anthropic Claude" },
};

const COMMON_BASE_URLS: { label: string; provider: "openai_compatible" | "claude"; base_url: string; models: string[]; tier?: string }[] = [
  { label: "OpenAI", provider: "openai_compatible", base_url: "https://api.openai.com/v1", models: ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o3-mini"], tier: "paid" },
  { label: "Anthropic Claude", provider: "claude", base_url: "https://api.anthropic.com", models: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"], tier: "paid" },
  { label: "DeepSeek", provider: "openai_compatible", base_url: "https://api.deepseek.com/v1", models: ["deepseek-chat", "deepseek-reasoner"], tier: "paid" },
  { label: "GLM (z.ai)", provider: "openai_compatible", base_url: "https://open.bigmodel.cn/api/paas/v4", models: ["glm-4-plus", "glm-4-air", "glm-4-flash"], tier: "paid" },
  { label: "Qwen (DashScope)", provider: "openai_compatible", base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", models: ["qwen3-max", "qwen-plus", "qwen-turbo"], tier: "paid" },
  { label: "Moonshot Kimi", provider: "openai_compatible", base_url: "https://api.moonshot.cn/v1", models: ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"], tier: "paid" },
  { label: "MiniMax", provider: "openai_compatible", base_url: "https://api.minimax.chat/v1", models: ["abab6.5-chat", "abab5.5-chat"], tier: "paid" },
  { label: "Groq", provider: "openai_compatible", base_url: "https://api.groq.com/openai/v1", models: ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"], tier: "free-tier" },
  // Phase C: free-tier aggregators that the user already uses
  { label: "OpenRouter (free tier)", provider: "openai_compatible", base_url: "https://openrouter.ai/api/v1", models: [
    // Free models on OpenRouter (as of 2026 — check openrouter.ai/models for current list)
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
  ], tier: "free" },
  { label: "NVIDIA build.nvidia.com", provider: "openai_compatible", base_url: "https://integrate.api.nvidia.com/v1", models: [
    // NVIDIA NIM models (free credit-based)
    "moonshotai/kimi-k3",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-r1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "qwen/qwen2.5-7b-instruct",
  ], tier: "free-credit" },
];

type Mode = "list" | "edit" | "create";

export function SettingsView() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [channels, setChannels] = useState<ChannelConfig[]>([]);
  const [defaultChannelId, setDefaultChannelId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("list");
  const [editingChannel, setEditingChannel] = useState<ChannelConfig | null>(null);

  // ---------- Load channels on mount ----------
  const reload = useCallback(async () => {
    try {
      const data = await listChannels();
      setChannels(data.channels || []);
      setDefaultChannelId(data.default_channel);
    } catch (err: any) {
      toast({
        title: "Failed to load channels",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    reload();
  }, [reload]);

  // ---------- Handlers ----------

  const handleSetDefault = async (channelId: string) => {
    try {
      await setDefaultChannel(channelId);
      setDefaultChannelId(channelId);
      toast({ title: "Default set", description: `Default channel: ${channelId}` });
    } catch (err: any) {
      toast({ title: "Failed to set default", description: err.message, variant: "destructive" });
    }
  };

  const handleDelete = async (channelId: string) => {
    if (!confirm(`Delete channel '${channelId}'? This cannot be undone.`)) return;
    try {
      await deleteChannel(channelId);
      await reload();
      toast({ title: "Channel deleted", description: channelId });
    } catch (err: any) {
      toast({ title: "Delete failed", description: err.message, variant: "destructive" });
    }
  };

  const handleTestSaved = async (channelId: string) => {
    toast({ title: "Testing…", description: `Pinging ${channelId}` });
    try {
      const result = await testSavedChannel(channelId);
      if (result.success) {
        toast({
          title: "Connection OK",
          description: `model=${result.model} latency=${result.latency_ms}ms`,
        });
      } else {
        toast({
          title: "Test failed",
          description: result.error || "Unknown error",
          variant: "destructive",
        });
      }
    } catch (err: any) {
      toast({ title: "Test error", description: err.message, variant: "destructive" });
    }
  };

  // ---------- Render ----------

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-zinc-600" />
      </div>
    );
  }

  if (mode === "edit" || mode === "create") {
    return (
      <ChannelEditor
        initial={editingChannel}
        isNew={mode === "create"}
        allChannels={channels}
        onClose={() => {
          setMode("list");
          setEditingChannel(null);
        }}
        onSaved={async () => {
          setMode("list");
          setEditingChannel(null);
          await reload();
        }}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">AI Channels</h2>
          <p className="text-xs text-zinc-500 mt-1">
            Multi-channel config (CyberStrikeAI pattern) — stored in <code className="text-zinc-400">config.yaml</code>
          </p>
        </div>
        <Button
          size="sm"
          onClick={() => {
            setEditingChannel(null);
            setMode("create");
          }}
          className="bg-emerald-700 hover:bg-emerald-600 text-white"
        >
          <Plus className="w-4 h-4 mr-1" />
          New channel
        </Button>
      </div>

      {/* Channels list — Phase C: pass allChannels to editor so it can render failover picker */}
      {channels.length === 0 ? (
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="py-12 text-center">
            <AlertCircle className="w-8 h-8 text-amber-400 mx-auto mb-3" />
            <p className="text-sm text-zinc-300 mb-1">No channels configured</p>
            <p className="text-xs text-zinc-500">
              Click <strong>New channel</strong> to add OpenAI, Anthropic, DeepSeek, GLM, etc.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {channels.map((ch) => {
            const isDefault = ch.id === defaultChannelId;
            return (
              <Card key={ch.id} className="bg-zinc-900 border-zinc-800 hover:border-zinc-700">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-zinc-100">{ch.name || ch.id}</span>
                        {isDefault && (
                          <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                            <Star className="w-2.5 h-2.5" />
                            Default
                          </span>
                        )}
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                          {ch.provider === "claude" ? "Claude" : "OpenAI-compat"}
                        </span>
                      </div>
                      <div className="text-xs text-zinc-500 font-mono truncate">
                        {ch.model} · {ch.base_url}
                      </div>
                      <div className="text-[10px] text-zinc-600 mt-1">
                        key: <span className="font-mono">{ch.api_key}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 ml-3">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleTestSaved(ch.id)}
                        title="Test connection"
                        className="text-zinc-300 hover:text-zinc-100 h-8 px-2"
                      >
                        <Plug className="w-3.5 h-3.5 mr-1" />
                        Test
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setEditingChannel(ch);
                          setMode("edit");
                        }}
                        title="Edit"
                        className="text-zinc-300 hover:text-zinc-100 h-8 px-2"
                      >
                        <Edit3 className="w-3.5 h-3.5 mr-1" />
                        Edit
                      </Button>
                      {!isDefault && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleSetDefault(ch.id)}
                          title="Set as default"
                          className="text-zinc-300 hover:text-zinc-100 h-8 px-2"
                        >
                          <Star className="w-3.5 h-3.5" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleDelete(ch.id)}
                        title="Delete"
                        className="text-red-400 hover:text-red-300 h-8 px-2"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Note */}
      <Alert className="bg-zinc-900/50 border-zinc-800 text-zinc-400">
        <AlertDescription className="text-xs">
          <strong className="text-zinc-300">Phase B migration:</strong> The old DB-stored per-user LLM settings
          ({"<Settings → API Key>"} in Phase A) have been replaced by YAML-based multi-channel config.
          New conversations use the default channel. Edit <code className="text-zinc-300">config.yaml</code> at the project root
          to view all channels (including those without api_key).
        </AlertDescription>
      </Alert>
    </div>
  );
}

// ============================================================
// Channel editor — Create / Edit form
// ============================================================

function ChannelEditor({
  initial,
  isNew,
  allChannels,
  onClose,
  onSaved,
}: {
  initial: ChannelConfig | null;
  isNew: boolean;
  allChannels: ChannelConfig[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const { toast } = useToast();
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ChannelTestResult | null>(null);

  // Form state
  const [id, setId] = useState(initial?.id || "");
  const [name, setName] = useState(initial?.name || "");
  const [provider, setProvider] = useState<"openai_compatible" | "claude">(initial?.provider || "openai_compatible");
  const [baseUrl, setBaseUrl] = useState(initial?.base_url || "https://api.openai.com/v1");
  const [apiKey, setApiKey] = useState(initial?.api_key || "");
  const [model, setModel] = useState(initial?.model || "gpt-4o-mini");
  const [maxTotalTokens, setMaxTotalTokens] = useState(initial?.max_total_tokens || 128000);
  const [maxCompletionTokens, setMaxCompletionTokens] = useState(initial?.max_completion_tokens || 4096);
  const [temperature, setTemperature] = useState(initial?.temperature ?? 0.7);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [reasoningMode, setReasoningMode] = useState<"auto" | "on" | "off">(initial?.reasoning?.mode || "auto");
  const [reasoningEffort, setReasoningEffort] = useState<"low" | "medium" | "high" | "max" | "xhigh">(initial?.reasoning?.effort || "medium");
  // Phase C: failover channels (list of channel IDs to try on retryable errors)
  const [failoverChannels, setFailoverChannels] = useState<string[]>(initial?.failover_channels || []);
  // Tracks whether the user typed the channel ID themselves. When they haven't,
  // picking a preset keeps the auto-generated ID in sync (and collision-free).
  const [idTouched, setIdTouched] = useState(false);

  // Compute detected reasoning for current model — purely cosmetic hint
  const reasoningHint = (() => {
    const m = (model || "").toLowerCase();
    if (m.includes("o3") || m.includes("o4") || /gpt-[56]/.test(m)) return "OpenAI reasoning_effort";
    if (m.includes("claude-3-7") || m.includes("claude-4") || m.includes("claude-opus-4")) return "Claude extended thinking";
    if (m.includes("deepseek-r") || m.includes("deepseek-reasoner")) return "DeepSeek thinking enabled";
    if (m.includes("qwq") || m.includes("qwen3-max")) return "Qwen reasoning";
    return "none (regular model)";
  })();

  // ---------- Preset handlers ----------

  /** "DeepSeek" → "deepseek", "NVIDIA build.nvidia.com" → "nvidia-build-nvidia-com". */
  const slugifyChannelId = (label: string) =>
    label.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "") || "channel";

  /** Avoid the "Channel already exists" 409 when the preset ID is taken. */
  const uniqueChannelId = (base: string) => {
    const taken = new Set(allChannels.map((c) => c.id));
    if (!taken.has(base)) return base;
    let n = 2;
    while (taken.has(`${base}-${n}`)) n += 1;
    return `${base}-${n}`;
  };

  const applyPreset = (preset: typeof COMMON_BASE_URLS[number]) => {
    setProvider(preset.provider);
    setBaseUrl(preset.base_url);
    if (preset.models.length > 0) setModel(preset.models[0]);
    if (!name) setName(preset.label);
    if (isNew && !idTouched) setId(uniqueChannelId(slugifyChannelId(preset.label)));
  };

  // ---------- Save ----------

  const handleSave = async () => {
    // Validate locally first so the user gets an actionable message instead of
    // a generic "Save failed" round-trip.
    const trimmedId = id.trim();
    const trimmedName = name.trim();
    const trimmedBaseUrl = baseUrl.trim();
    const trimmedModel = model.trim();
    const keyIsMasked = apiKey.startsWith("•");

    const fail = (description: string) => {
      toast({ title: "Save failed", description, variant: "destructive" });
    };
    if (!trimmedId) return fail("Channel ID is required (e.g. 'deepseek').");
    if (!trimmedName) return fail("Display name is required.");
    if (!trimmedBaseUrl) return fail("Base URL is required (e.g. https://api.deepseek.com/v1).");
    if (!apiKey.trim() && !keyIsMasked) return fail("API key is required.");
    if (!trimmedModel) return fail("Model is required (e.g. 'deepseek-chat').");
    if (isNew && allChannels.some((c) => c.id === trimmedId)) {
      return fail(
        `A channel with ID '${trimmedId}' already exists. Edit it instead, or choose a different ID.`,
      );
    }

    setSaving(true);
    setTestResult(null);
    const payload: ChannelConfig = {
      id: trimmedId,
      name: trimmedName,
      provider,
      base_url: trimmedBaseUrl,
      api_key: apiKey,
      model: trimmedModel,
      max_total_tokens: maxTotalTokens,
      max_completion_tokens: maxCompletionTokens,
      temperature,
      reasoning: { mode: reasoningMode, effort: reasoningEffort },
      failover_channels: failoverChannels,
    };
    try {
      if (isNew) {
        await createChannel(payload);
        toast({ title: "Channel created", description: `${trimmedName} (${trimmedId})` });
      } else {
        await updateChannel(id, payload);
        toast({ title: "Channel updated", description: `${trimmedName} (${trimmedId})` });
      }
      onSaved();
    } catch (err: any) {
      toast({ title: "Save failed", description: err.message, variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  // ---------- Test ----------

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    if (!baseUrl || !apiKey || !model) {
      setTestResult({ success: false, error: "Base URL, API Key and Model are required" });
      setTesting(false);
      return;
    }
    if (apiKey.startsWith("•")) {
      setTestResult({ success: false, error: "API key is masked. Re-enter the actual key to test." });
      setTesting(false);
      return;
    }
    try {
      const result = await testChannelInline({
        id,
        name,
        provider,
        base_url: baseUrl,
        api_key: apiKey,
        model,
        max_total_tokens: maxTotalTokens,
        // Reasoning models (DeepSeek flash/reasoner, o-series, Claude thinking)
        // spend tokens on hidden chain-of-thought first; a tiny cap makes the
        // visible content come back empty and looks like a connection failure.
        max_completion_tokens: 256,
        temperature: 0.0,
        reasoning: { mode: reasoningMode, effort: reasoningEffort },
        failover_channels: [],  // test only the primary, not failovers
      });
      setTestResult(result);
    } catch (err: any) {
      setTestResult({ success: false, error: err.message });
    } finally {
      setTesting(false);
    }
  };

  // ---------- Render ----------

  return (
    <div className="space-y-4 max-w-3xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">
            {isNew ? "New AI channel" : `Edit channel: ${initial?.name}`}
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            Configure provider, base URL, API key, model. Test before saving.
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={onClose} className="border-zinc-700 text-zinc-300">
          Back to list
        </Button>
      </div>

      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Quick presets</CardTitle>
          <CardDescription className="text-zinc-500">
            Click a provider to auto-fill base URL + a default model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {COMMON_BASE_URLS.map((p) => {
              const isActive = provider === p.provider && baseUrl === p.base_url;
              const tierBadge = p.tier === "free" ? "free" : p.tier === "free-credit" ? "credit" : p.tier === "free-tier" ? "free" : p.tier === "paid" ? "paid" : null;
              return (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => applyPreset(p)}
                  className={`px-2.5 py-1 rounded text-xs border transition-colors flex items-center gap-1.5 ${
                    isActive
                      ? "bg-emerald-950 text-emerald-200 border-emerald-800"
                      : "bg-zinc-950 text-zinc-300 border-zinc-800 hover:border-zinc-700"
                  }`}
                >
                  {p.label}
                  {tierBadge && (
                    <span
                      className={`text-[9px] px-1 rounded ${
                        tierBadge === "free" || tierBadge === "credit"
                          ? "bg-amber-950 text-amber-300"
                          : "bg-zinc-800 text-zinc-500"
                      }`}
                      title={tierBadge === "free" ? "Truly free models available" : tierBadge === "credit" ? "Free credit-based" : "Paid only"}
                    >
                      {tierBadge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <p className="text-[10px] text-zinc-600 mt-3">
            <strong className="text-amber-400">Tip for free tier:</strong> OpenRouter models with <code>:free</code> suffix are
            truly free (rate-limited). NVIDIA build.nvidia.com gives 1,000 free credits at sign-up — try
            <code>nvidia/llama-3.1-nemotron-70b-instruct</code> or <code>meta/llama-3.3-70b-instruct</code> (these usually work).
            Set up 2 channels + use the Failover feature so when NVIDIA returns 429, OpenRouter picks up.
          </p>
        </CardContent>
      </Card>

      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Channel details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Channel ID + Name */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <Label className="text-zinc-300 text-xs">Channel ID</Label>
              <Input
                value={id}
                onChange={(e) => { setId(e.target.value); setIdTouched(true); }}
                disabled={!isNew}
                placeholder="openai, claude, glm, ..."
                className="bg-zinc-950 border-zinc-800 text-zinc-100 font-mono text-sm"
              />
              <p className="text-[10px] text-zinc-600 mt-1">
                {isNew ? "Unique ID — cannot be changed later" : "Read-only (delete + recreate to rename)"}
              </p>
            </div>
            <div>
              <Label className="text-zinc-300 text-xs">Display name</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="OpenAI (default)"
                className="bg-zinc-950 border-zinc-800 text-zinc-100"
              />
            </div>
          </div>

          {/* Provider */}
          <div>
            <Label className="text-zinc-300 text-xs">Provider</Label>
            <Select value={provider} onValueChange={(v: "openai_compatible" | "claude") => setProvider(v)}>
              <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-100">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="openai_compatible">openai_compatible (OpenAI / DeepSeek / GLM / Qwen / Kimi / etc.)</SelectItem>
                <SelectItem value="claude">claude (Anthropic native Messages API)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-[10px] text-zinc-600 mt-1">
              {provider === "claude"
                ? "Will call POST {base_url}/v1/messages with x-api-key header"
                : "Will call POST {base_url}/chat/completions with Bearer auth"}
            </p>
          </div>

          {/* Base URL */}
          <div>
            <Label className="text-zinc-300 text-xs">Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.openai.com/v1"
              className="bg-zinc-950 border-zinc-800 text-zinc-100 font-mono text-sm"
            />
            <p className="text-[10px] text-zinc-600 mt-1">
              No trailing slash. For OpenAI-compat, the path should usually include <code>/v1</code>.
            </p>
          </div>

          {/* API Key */}
          <div>
            <Label className="text-zinc-300 text-xs">API Key</Label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="bg-zinc-950 border-zinc-800 text-zinc-100 font-mono text-sm"
            />
            <p className="text-[10px] text-zinc-600 mt-1">
              {apiKey.startsWith("•")
                ? "Currently masked — re-enter the actual key to test or save"
                : "Stored in config.yaml at project root (plaintext — single-tenant tool)"}
            </p>
          </div>

          {/* Model */}
          <div>
            <Label className="text-zinc-300 text-xs">Model</Label>
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gpt-4o-mini"
              className="bg-zinc-950 border-zinc-800 text-zinc-100 font-mono text-sm"
            />
            <p className="text-[10px] text-zinc-600 mt-1">
              Use the model ID as documented by the provider (e.g. <code>deepseek-chat</code>, <code>claude-3-5-sonnet-20241022</code>).
            </p>
          </div>

          {/* Failover channels (Phase C) */}
          <div className="pt-3 border-t border-zinc-800">
            <Label className="text-zinc-300 text-xs">Failover channels (optional)</Label>
            <p className="text-[10px] text-zinc-600 mt-1 mb-2">
              If this channel returns 429 / 5xx / timeout, try these channels in order.
              Useful when mixing free-tier providers (NVIDIA → OpenRouter as backup).
            </p>
            <div className="flex flex-wrap gap-1.5">
              {allChannels
                .filter((c) => c.id !== id)
                .map((c) => {
                  const selected = failoverChannels.includes(c.id);
                  return (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => {
                        if (selected) {
                          setFailoverChannels(failoverChannels.filter((x) => x !== c.id));
                        } else {
                          setFailoverChannels([...failoverChannels, c.id]);
                        }
                      }}
                      className={`px-2 py-0.5 rounded text-[11px] border transition-colors ${
                        selected
                          ? "bg-emerald-950 text-emerald-200 border-emerald-800"
                          : "bg-zinc-950 text-zinc-400 border-zinc-800 hover:border-zinc-700"
                      }`}
                    >
                      {selected ? "✓ " : ""}{c.id} <span className="opacity-60">· {c.model}</span>
                    </button>
                  );
                })}
              {allChannels.filter((c) => c.id !== id).length === 0 && (
                <span className="text-[10px] text-zinc-600 italic">
                  No other channels available. Create another channel first.
                </span>
              )}
            </div>
            {failoverChannels.length > 0 && (
              <p className="text-[10px] text-emerald-500 mt-2">
                ✓ Will try: {id} → {failoverChannels.join(" → ")}
              </p>
            )}
          </div>

          {/* Advanced toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-zinc-400 hover:text-zinc-200 mt-2"
          >
            {showAdvanced ? "▼ Hide advanced" : "▶ Show advanced (tokens, temperature, reasoning)"}
          </button>

          {showAdvanced && (
            <div className="space-y-4 pt-2 border-t border-zinc-800">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-zinc-300 text-xs">Max total tokens (context window)</Label>
                  <Input
                    type="number"
                    value={maxTotalTokens}
                    onChange={(e) => setMaxTotalTokens(Number(e.target.value) || 0)}
                    className="bg-zinc-950 border-zinc-800 text-zinc-100"
                  />
                </div>
                <div>
                  <Label className="text-zinc-300 text-xs">Max completion tokens</Label>
                  <Input
                    type="number"
                    value={maxCompletionTokens}
                    onChange={(e) => setMaxCompletionTokens(Number(e.target.value) || 0)}
                    className="bg-zinc-950 border-zinc-800 text-zinc-100"
                  />
                </div>
              </div>

              <div>
                <Label className="text-zinc-300 text-xs">
                  Temperature: <span className="font-mono text-zinc-100">{temperature.toFixed(2)}</span>
                </Label>
                <Slider
                  value={[temperature]}
                  onValueChange={(v) => setTemperature(v[0])}
                  min={0}
                  max={2}
                  step={0.05}
                  className="mt-2"
                />
              </div>

              {/* Reasoning (Phase C: auto-detect hint shown) */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-zinc-800">
                <div>
                  <Label className="text-zinc-300 text-xs">Reasoning mode</Label>
                  <Select value={reasoningMode} onValueChange={(v: "auto" | "on" | "off") => setReasoningMode(v)}>
                    <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">auto (detect from model name)</SelectItem>
                      <SelectItem value="on">on (force extended thinking)</SelectItem>
                      <SelectItem value="off">off (no reasoning)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-[10px] text-zinc-600 mt-1">
                    <strong>auto</strong> (recommended): detects <code>o3/gpt-5</code>, <code>claude-3-7+</code>,
                    <code>deepseek-r/reasoner</code>, <code>qwq</code> by model name. Current model → <span className="text-emerald-400">{reasoningHint}</span>.
                  </p>
                </div>
                <div>
                  <Label className="text-zinc-300 text-xs">Reasoning effort</Label>
                  <Select value={reasoningEffort} onValueChange={(v: any) => setReasoningEffort(v)}>
                    <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">low</SelectItem>
                      <SelectItem value="medium">medium</SelectItem>
                      <SelectItem value="high">high</SelectItem>
                      <SelectItem value="max">max (Claude / OpenAI)</SelectItem>
                      <SelectItem value="xhigh">xhigh (OpenAI gpt-5)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-[10px] text-zinc-600 mt-1">
                    Sent as <code>reasoning_effort</code> (OpenAI) or controls Claude thinking budget.
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex flex-col items-stretch gap-2 border-t border-zinc-800 pt-4">
          {testResult && (
            <div
              className={`rounded p-2 text-xs border ${
                testResult.success
                  ? "bg-emerald-950/50 border-emerald-900 text-emerald-300"
                  : "bg-red-950/50 border-red-900 text-red-300"
              }`}
            >
              <div className="flex items-center gap-2">
                {testResult.success ? (
                  <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                )}
                <span className="font-mono">
                  {testResult.success
                    ? `OK — model=${testResult.model} latency=${testResult.latency_ms}ms${
                        testResult.response_preview ? ` → ${testResult.response_preview}` : ""
                      }`
                    : testResult.error}
                  {testResult.status_code ? ` [HTTP ${testResult.status_code}]` : ""}
                </span>
              </div>
              {testResult.response_body && !testResult.success && (
                <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-all text-[10px] bg-zinc-950 p-2 rounded border border-zinc-800 text-red-200">
{testResult.response_body}
                </pre>
              )}
            </div>
          )}
          <div className="flex items-center justify-end gap-2">
            <Button
              variant="outline"
              onClick={handleTest}
              disabled={testing || saving}
              className="border-zinc-700 text-zinc-200 hover:bg-zinc-800"
            >
              {testing ? (
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
              ) : (
                <Plug className="w-4 h-4 mr-1" />
              )}
              Test
            </Button>
            <Button
              onClick={handleSave}
              disabled={saving || testing}
              className="bg-emerald-700 hover:bg-emerald-600 text-white"
            >
              {saving ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Save className="w-4 h-4 mr-1" />}
              {isNew ? "Create" : "Save"}
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}