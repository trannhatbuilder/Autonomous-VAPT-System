'use client';

import { useEffect, useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Alert, AlertDescription } from "../ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Slider } from "../ui/slider";
import { Switch } from "../ui/switch";
import { Loader2, Save, Plug, CheckCircle2, AlertCircle } from "lucide-react";
import { getLLMSettings, saveLLMSettings, testLLMConnection } from "../../lib/api";
import { useToast } from "../../hooks/use-toast";

const PROVIDERS = [
  { value: "openai", label: "OpenAI (official)" },
  { value: "anthropic", label: "Anthropic Claude" },
  { value: "glm", label: "GLM (z.ai)" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "groq", label: "Groq" },
  { value: "google", label: "Google Gemini" },
  { value: "ollama", label: "Ollama (local)" },
  { value: "minimax", label: "MiniMax" },
];

const POPULAR_MODELS: Record<string, string[]> = {
  openai: ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
  anthropic: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
  glm: ["glm-4-plus", "glm-4-air", "glm-4-flash"],
  deepseek: ["deepseek-chat", "deepseek-coder"],
  groq: ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
  google: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
  ollama: ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
  minimax: ["abab6.5-chat", "abab5.5-chat"],
};

export function SettingsView() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  const [provider, setProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o-mini");
  const [baseUrl, setBaseUrl] = useState("");
  const [maxTotalTokens, setMaxTotalTokens] = useState(200000);
  const [maxCompletionTokens, setMaxCompletionTokens] = useState(4096);
  const [temperature, setTemperature] = useState(0.7);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Load existing settings on mount
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const data = await getLLMSettings();
        if (!mounted) return;
        const llm = data.llm || {};
        if (llm.provider) setProvider(llm.provider);
        if (llm.api_key) setApiKey(llm.api_key);  // masked
        if (llm.model) setModel(llm.model);
        if (llm.base_url) setBaseUrl(llm.base_url);
        if (llm.max_total_tokens) setMaxTotalTokens(llm.max_total_tokens);
        if (llm.max_completion_tokens) setMaxCompletionTokens(llm.max_completion_tokens);
        if (llm.temperature !== undefined) setTemperature(llm.temperature);
      } catch (err: any) {
        // Silently ignore — settings page can show empty form
        console.warn("Failed to load LLM settings:", err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setTestResult(null);
    try {
      await saveLLMSettings({
        provider,
        api_key: apiKey,
        model,
        base_url: baseUrl,
        max_total_tokens: maxTotalTokens,
        max_completion_tokens: maxCompletionTokens,
        temperature,
      });
      toast({
        title: "Settings saved",
        description: `LLM provider: ${provider}, model: ${model}`,
      });
    } catch (err: any) {
      toast({
        title: "Save failed",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testLLMConnection({
        provider,
        api_key: apiKey,
        model,
        base_url: baseUrl,
      });
      setTestResult({
        success: result.success,
        message: result.success ? (result.message || "Connection OK") : (result.error || "Test failed"),
      });
      if (result.success) {
        toast({
          title: "Connection OK",
          description: result.message,
        });
      } else {
        toast({
          title: "Connection failed",
          description: result.error,
          variant: "destructive",
        });
      }
    } catch (err: any) {
      setTestResult({ success: false, message: err.message });
      toast({
        title: "Connection test failed",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setTesting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-zinc-50">Settings</h2>
        <p className="text-sm text-zinc-400 mt-1">
          Configure your LLM provider. VAPT-AI uses these credentials to drive the
          pentest agent. Keys are stored in your user profile (single-user tool).
        </p>
      </div>

      <Card className="bg-zinc-900/60 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-50 flex items-center gap-2">
            <Plug className="w-4 h-4 text-emerald-400" />
            LLM Provider
          </CardTitle>
          <CardDescription className="text-zinc-400">
            OpenAI is the recommended provider. Other providers work via litellm routing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Provider */}
          <div className="space-y-2">
            <Label htmlFor="provider" className="text-zinc-200">Provider</Label>
            <Select value={provider} onValueChange={setProvider}>
              <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-zinc-900 border-zinc-800">
                {PROVIDERS.map((p) => (
                  <SelectItem key={p.value} value={p.value} className="text-zinc-100">
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* API Key */}
          <div className="space-y-2">
            <Label htmlFor="api_key" className="text-zinc-200">API Key</Label>
            <Input
              id="api_key"
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono"
            />
            <p className="text-xs text-zinc-500">
              {apiKey?.startsWith("•") ? "Saved key (masked). Re-enter to replace." : "Get your API key from your provider's dashboard."}
            </p>
          </div>

          {/* Model */}
          <div className="space-y-2">
            <Label htmlFor="model" className="text-zinc-200">Model</Label>
            <Input
              id="model"
              type="text"
              placeholder="gpt-4o-mini"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono"
              list="model-suggestions"
            />
            <datalist id="model-suggestions">
              {(POPULAR_MODELS[provider] || []).map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            <p className="text-xs text-zinc-500">
              Popular: {(POPULAR_MODELS[provider] || []).join(", ")}
            </p>
          </div>

          {/* Base URL (optional) */}
          <div className="space-y-2">
            <Label htmlFor="base_url" className="text-zinc-200">Base URL <span className="text-zinc-500">(optional)</span></Label>
            <Input
              id="base_url"
              type="text"
              placeholder="https://api.openai.com/v1 (leave empty for default)"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              className="bg-zinc-950 border-zinc-800 text-zinc-50 placeholder-zinc-600 font-mono"
            />
            <p className="text-xs text-zinc-500">
              For OpenAI-compatible providers (GLM, MiniMax, Ollama, self-hosted), enter the API endpoint.
            </p>
          </div>

          {/* Advanced toggle */}
          <div className="flex items-center justify-between pt-2">
            <Label htmlFor="advanced" className="text-zinc-200">Advanced settings</Label>
            <Switch
              id="advanced"
              checked={showAdvanced}
              onCheckedChange={setShowAdvanced}
            />
          </div>

          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-zinc-800">
              {/* Temperature */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-zinc-200">Temperature</Label>
                  <span className="text-sm text-zinc-400 tabular-nums">{temperature.toFixed(2)}</span>
                </div>
                <Slider
                  value={[temperature]}
                  onValueChange={(v) => setTemperature(v[0])}
                  min={0}
                  max={2}
                  step={0.1}
                  className="py-2"
                />
                <p className="text-xs text-zinc-500">
                  Lower = deterministic (recommended for pentest). Higher = creative.
                </p>
              </div>

              {/* Max tokens */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="max_total" className="text-zinc-200">Max total tokens</Label>
                  <Input
                    id="max_total"
                    type="number"
                    value={maxTotalTokens}
                    onChange={(e) => setMaxTotalTokens(parseInt(e.target.value) || 200000)}
                    className="bg-zinc-950 border-zinc-800 text-zinc-50"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="max_completion" className="text-zinc-200">Max completion tokens</Label>
                  <Input
                    id="max_completion"
                    type="number"
                    value={maxCompletionTokens}
                    onChange={(e) => setMaxCompletionTokens(parseInt(e.target.value) || 4096)}
                    className="bg-zinc-950 border-zinc-800 text-zinc-50"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Test result */}
          {testResult && (
            <Alert className={testResult.success ? "bg-emerald-950/40 border-emerald-900" : "bg-red-950/40 border-red-900"}>
              <AlertDescription className={testResult.success ? "text-emerald-200" : "text-red-200"}>
                <div className="flex items-start gap-2">
                  {testResult.success ? (
                    <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  )}
                  <span>{testResult.message}</span>
                </div>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button
            onClick={handleTest}
            disabled={testing || !apiKey || apiKey.startsWith("•") || !model}
            variant="outline"
            className="border-zinc-700 text-zinc-200 hover:bg-zinc-800"
          >
            {testing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Plug className="w-4 h-4 mr-2" />}
            Test connection
          </Button>
          <Button
            onClick={handleSave}
            disabled={saving || !provider || !model || (!apiKey || apiKey.startsWith("•") ? false : !apiKey)}
            className="bg-emerald-600 hover:bg-emerald-500 text-zinc-50"
          >
            {saving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
            Save settings
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
