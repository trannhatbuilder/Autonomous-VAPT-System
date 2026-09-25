/**
 * VAPT-AI API client — Phase A (frontend/backend split).
 *
 * Single mode: RELATIVE paths.
 *  - Local dev:  Next.js rewrites() in next.config.ts proxies /api/* → http://localhost:8000
 *  - Sandbox:    Caddy `?XTransformPort=8000` query routes to FastAPI on port 8000.
 *
 * No env vars required. Set NEXT_PUBLIC_API_BASE_URL only if you want to bypass
 * the rewrite layer and call an absolute URL (e.g. a remote backend).
 *
 * Auth: JWT access + refresh tokens stored in localStorage.
 *  - On 401 response, automatically refreshes once + retries the call.
 *  - On refresh failure, clears tokens + redirects to login.
 */

/**
 * Build the API base URL.
 * - Default: empty string → relative paths (proxied by Next.js / Caddy).
 * - Override: set NEXT_PUBLIC_API_BASE_URL to use absolute URLs.
 */
function getApiConfig(): { baseUrl: string } {
  const envBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (envBaseUrl) {
    return { baseUrl: envBaseUrl.replace(/\/$/, "") };
  }
  return { baseUrl: "" };
}

const SANDBOX_BACKEND_PORT = "8000";

/** Helper: build URL — relative by default, absolute if NEXT_PUBLIC_API_BASE_URL set. */
function apiUrl(path: string, params?: Record<string, string | number | boolean | undefined>): string {
  const { baseUrl } = getApiConfig();
  // If env override is set → use absolute URL
  if (baseUrl) {
    const url = new URL(path, baseUrl);
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    return url.toString();
  }
  // Sandbox preview: detect preview.space-z.ai → use XTransformPort query param.
  // Local dev: just use relative path, Next.js rewrites will proxy it.
  const isSandboxPreview =
    typeof window !== "undefined" &&
    window.location.hostname.endsWith(".space-z.ai");
  if (isSandboxPreview) {
    const url = new URL(path, window.location.origin);
    url.searchParams.set("XTransformPort", SANDBOX_BACKEND_PORT);
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    return url.toString().replace(window.location.origin, "");
  }
  // Local dev: relative path
  const search = params
    ? "?" + Object.entries(params)
        .filter(([_, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&")
    : "";
  return `${path}${search}`;
}

/** Token storage helpers (localStorage — single-user internal tool). */
export const tokenStorage = {
  getAccessToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("vapt_access_token");
  },
  getRefreshToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("vapt_refresh_token");
  },
  setTokens(accessToken: string, refreshToken: string): void {
    if (typeof window === "undefined") return;
    localStorage.setItem("vapt_access_token", accessToken);
    localStorage.setItem("vapt_refresh_token", refreshToken);
  },
  clearTokens(): void {
    if (typeof window === "undefined") return;
    localStorage.removeItem("vapt_access_token");
    localStorage.removeItem("vapt_refresh_token");
  },
};

export class ApiError extends Error {
  status: number;
  detail: any;
  constructor(message: string, status: number, detail?: any) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

/** Refresh the access token using the refresh token. Returns new tokens or throws. */
async function refreshAccessToken(): Promise<{ access_token: string; refresh_token: string }> {
  const refreshToken = tokenStorage.getRefreshToken();
  if (!refreshToken) {
    throw new ApiError("No refresh token", 401);
  }
  const res = await fetch(apiUrl("/api/auth/refresh"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new ApiError("Refresh failed", res.status, detail);
  }
  const data = await res.json();
  tokenStorage.setTokens(data.access_token, data.refresh_token);
  return data;
}

/**
 * Authenticated fetch wrapper.
 * - Adds Authorization: Bearer <access_token> header
 * - On 401: tries refreshAccessToken() once, then retries the original call.
 * - On repeated 401: clears tokens + throws ApiError(401) — caller should redirect to login.
 */
export async function apiFetch<T = any>(
  path: string,
  options: RequestInit = {},
  params?: Record<string, string | number | boolean | undefined>,
): Promise<T> {
  const accessToken = tokenStorage.getAccessToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
    headers["Content-Type"] = "application/json";
  }

  const doFetch = async (): Promise<Response> => {
    return fetch(apiUrl(path, params), {
      ...options,
      headers,
    });
  };

  let res = await doFetch();

  if (res.status === 401) {
    // Try refresh + retry once
    try {
      await refreshAccessToken();
      headers["Authorization"] = `Bearer ${tokenStorage.getAccessToken()}`;
      res = await doFetch();
    } catch (refreshErr) {
      tokenStorage.clearTokens();
      throw new ApiError("Session expired. Please log in again.", 401);
    }
  }

  if (!res.ok) {
    // Read body ONCE as text, then try to parse as JSON
    // (avoids "body stream already read" error from calling both .json() + .text())
    let detail: any;
    const text = await res.text();
    try { detail = JSON.parse(text); } catch { detail = text; }
    const message =
      (detail && (detail.detail || detail.message)) ||
      `HTTP ${res.status}: ${res.statusText}`;
    throw new ApiError(message, res.status, detail);
  }

  // 204 No Content
  if (res.status === 204) return undefined as T;
  // Empty body
  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

/** Unauthenticated fetch — used only for /api/auth/login. */
export async function apiLogin(email: string, password: string): Promise<{
  access_token: string;
  refresh_token: string;
  user: { id: string; email: string; role: string; display_name: string | null };
}> {
  const res = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    // Read body ONCE as text, then try to parse as JSON
    let detail: any;
    const text = await res.text();
    try { detail = JSON.parse(text); } catch { detail = text; }
    const message =
      (detail && (detail.detail || detail.message)) ||
      `HTTP ${res.status}: ${res.statusText}`;
    throw new ApiError(message, res.status, detail);
  }
  return res.json();
}

/** GET /api/auth/me — verify access token validity. */
export async function getMe(): Promise<{ id: string; email: string; role: string; display_name: string | null }> {
  return apiFetch("/api/auth/me");
}

/** POST /api/scans/start — start a scan, returns scan_id immediately. */
export async function startScan(target: string, userPrompt: string): Promise<{
  scan_id: string;
  target: string;
  user_prompt: string;
  status: string;
  message: string;
}> {
  return apiFetch("/api/scans/start", {
    method: "POST",
    body: JSON.stringify({ target, user_prompt: userPrompt }),
  });
}

/** GET /api/scans/{scan_id}/blackboard — get blackboard summary. */
export async function getBlackboard(scanId: string): Promise<any> {
  return apiFetch(`/api/scans/${scanId}/blackboard`);
}

/** GET /api/scans/{scan_id}/facts — get facts for a scan. */
export async function getFacts(scanId: string, factType?: string): Promise<any> {
  const params: Record<string, string | undefined> = {};
  if (factType) params.fact_type = factType;
  return apiFetch(`/api/scans/${scanId}/facts`, undefined, params);
}

/** GET /mcp/tools/definitions — list all 32 YAML tool definitions. */
export async function getTools(): Promise<{
  tools_count: number;
  tools: Array<{
    name: string;
    command: string;
    category: string;
    short_description: string;
    wstg_ids: string[];
    mitre_attack: string[];
    safety_class: string;
    parameters: string[];
    timeout: number;
  }>;
}> {
  return apiFetch("/mcp/tools/definitions");
}

/** GET /api/orchestration/agents — list all 16 agents (3 orchestrators + 13 specialists). */
export async function getAgents(): Promise<{
  agents_count: number;
  agents: Array<{
    name: string;
    display_name: string;
    description: string;
    safety_class: string;
    tool_allowlist: string[];
    is_destructive: boolean;
    prompt_file: string;
  }>;
}> {
  return apiFetch("/api/orchestration/agents");
}

/** One tool execution, mirrors app/mcp/execution_service.py ToolExecution.to_dict(). */
export interface ToolExecution {
  id: string;
  tool_name: string;
  target: string;
  scan_id: string | null;
  actor_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "hard_timeout"
    | "background_running" | "orphaned";
  started_at: string | null;
  completed_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  duration_seconds: number;
}

/** GET /api/mcp/executions — list tool executions. */
export async function getExecutions(scanId?: string, limit: number = 50): Promise<{
  executions_count: number;
  executions: ToolExecution[];
}> {
  const params: Record<string, string | number | undefined> = { limit };
  if (scanId) params.scan_id = scanId;
  return apiFetch("/api/mcp/executions", undefined, params);
}

/** POST /api/mcp/executions/{id}/cancel — panic button for single execution. */
export async function cancelExecution(executionId: string): Promise<{
  cancelled: boolean;
  execution_id: string;
  final_status: string | null;
  reason: string;
}> {
  return apiFetch(`/api/mcp/executions/${executionId}/cancel`, { method: "POST" });
}

/** POST /api/mcp/scans/{scan_id}/abort — bulk panic button. */
export async function abortScan(scanId: string): Promise<{
  scan_id: string;
  cancelled_executions: number;
  message: string;
}> {
  return apiFetch(`/api/mcp/scans/${scanId}/abort`, { method: "POST" });
}

/** GET /api/settings/llm — fetch current user's LLM config (api_key masked). */
export async function getLLMSettings(): Promise<{
  llm: {
    provider?: string;
    api_key?: string;  // masked
    model?: string;
    base_url?: string;
    max_total_tokens?: number;
    max_completion_tokens?: number;
    temperature?: number;
    hitl_audit_api_key?: string;  // masked
  };
}> {
  return apiFetch("/api/settings/llm");
}

/** POST /api/settings/llm — save LLM config. */
export async function saveLLMSettings(config: {
  provider: string;
  api_key: string;
  model: string;
  base_url?: string;
  max_total_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
}): Promise<{ status: string; message: string }> {
  return apiFetch("/api/settings/llm", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/** POST /api/settings/llm/test — test LLM connection (Phase A, deprecated). */
export async function testLLMConnection(config: {
  provider: string;
  api_key: string;
  model: string;
  base_url?: string;
}): Promise<{ success: boolean; message?: string; error?: string }> {
  return apiFetch("/api/settings/llm/test", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// ============================================================
// Channels API (Phase B — CyberStrikeAI pattern)
// Replaces the single LLM settings endpoints above.
// ============================================================

export interface ChannelConfig {
  id: string;
  name: string;
  provider: "openai_compatible" | "claude";
  base_url: string;
  api_key: string;           // masked (••••1234) when read from server
  model: string;
  max_total_tokens: number;
  max_completion_tokens: number;
  temperature: number;
  reasoning?: {
    mode?: "auto" | "on" | "off";
    effort?: "low" | "medium" | "high" | "max" | "xhigh";
    budget_tokens?: number;
  };
  failover_channels?: string[];   // Phase C: list of channel IDs for fallback
}

export interface ChannelTestResult {
  success: boolean;
  model?: string;
  latency_ms?: number;
  error?: string;
  status_code?: number | null;
  response_preview?: string;
  response_body?: string | null;  // Phase C: first 500 chars of upstream error body
  channel_id?: string;
}

/** GET /api/channels — list all channels (api_key masked). */
export async function listChannels(): Promise<{
  channels_count: number;
  default_channel: string | null;
  config_path: string;
  channels: ChannelConfig[];
}> {
  return apiFetch("/api/channels");
}

/** GET /api/channels/{id} — fetch one channel (api_key masked). */
export async function getChannel(channelId: string): Promise<ChannelConfig> {
  return apiFetch(`/api/channels/${channelId}`);
}

/** POST /api/channels — create a new channel. */
export async function createChannel(channel: ChannelConfig): Promise<{
  status: string;
  channel: ChannelConfig;
}> {
  return apiFetch("/api/channels", {
    method: "POST",
    body: JSON.stringify(channel),
  });
}

/** PUT /api/channels/{id} — update an existing channel. */
export async function updateChannel(channelId: string, channel: ChannelConfig): Promise<{
  status: string;
  channel: ChannelConfig;
}> {
  return apiFetch(`/api/channels/${channelId}`, {
    method: "PUT",
    body: JSON.stringify(channel),
  });
}

/** DELETE /api/channels/{id} — delete a channel. */
export async function deleteChannel(channelId: string): Promise<{ status: string; deleted: string }> {
  return apiFetch(`/api/channels/${channelId}`, { method: "DELETE" });
}

/** POST /api/channels/test — test channel config inline (no save). */
export async function testChannelInline(channel: ChannelConfig): Promise<ChannelTestResult> {
  return apiFetch("/api/channels/test", {
    method: "POST",
    // 256 tokens: reasoning models need headroom for their hidden
    // chain-of-thought before any visible content is produced.
    body: JSON.stringify({ ...channel, max_completion_tokens: 256, temperature: 0.0 }),
  });
}

/** POST /api/channels/{id}/test — test an already-saved channel. */
export async function testSavedChannel(channelId: string): Promise<ChannelTestResult> {
  return apiFetch(`/api/channels/${channelId}/test`, { method: "POST" });
}

/** GET /api/channels/default — get default channel id. */
export async function getDefaultChannel(): Promise<{ default_channel: string | null }> {
  return apiFetch("/api/channels/default");
}

/** POST /api/channels/default/{id} — set default channel. */
export async function setDefaultChannel(channelId: string): Promise<{ status: string; default_channel: string }> {
  return apiFetch(`/api/channels/default/${channelId}`, { method: "POST" });
}

/** GET /api/findings — list all findings (paginated). */
export async function getFindings(params?: {
  scan_id?: string;
  severity?: string;
  verified?: "true" | "false";
  limit?: number;
  offset?: number;
}): Promise<{
  findings: Array<{
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
  }>;
  total: number;
}> {
  return apiFetch("/api/findings", undefined, params as any);
}

/** GET /api/scans/active — list active scans. */
export async function getActiveScans(): Promise<{
  active_scans: Array<{
    scan_id: string;
    target: string;
    started_at: string;
    user_prompt: string;
    status: string;
    progress: number;
  }>;
  count: number;
}> {
  return apiFetch("/api/scans/active");
}

/** POST /api/scans/{scan_id}/abort — abort a scan (separate from abort-tools). */
export async function abortScanPipeline(scanId: string): Promise<any> {
  return apiFetch(`/api/scans/${scanId}/abort`, { method: "POST" });
}

/** GET /api/hitl/pending/{scan_id} — list pending HITL approvals. */
export async function getPendingHITLApprovals(scanId: string): Promise<{
  approvals: Array<any>;
  count: number;
}> {
  return apiFetch(`/api/hitl/pending/${scanId}`);
}

/** POST /api/hitl/{approval_id}/approve — approve a HITL request. */
export async function approveHITL(approvalId: string, comment?: string): Promise<any> {
  return apiFetch(`/api/hitl/${approvalId}/approve`, {
    method: "POST",
    body: JSON.stringify({ comment: comment || "" }),
  });
}

/** POST /api/hitl/{approval_id}/abort — reject a HITL request. */
export async function rejectHITL(approvalId: string, comment?: string): Promise<any> {
  return apiFetch(`/api/hitl/${approvalId}/abort`, {
    method: "POST",
    body: JSON.stringify({ comment: comment || "" }),
  });
}

/**
 * Build SSE endpoint URL for scan events.
 *
 * CRITICAL — Next.js dev proxy bug:
 *   The Next.js `rewrites()` proxy (http-proxy under the hood) BUFFERS
 *   streaming responses. SSE streams hang indefinitely through the
 *   proxy — the browser connects (200 OK) but receives no `data:`
 *   frames until the proxy buffer flushes (which never happens for
 *   a long-lived SSE stream).
 *   Symptom: backend log shows "SSE subscriber connected" + events
 *   being published, but the frontend shows "Waiting for events..."
 *   forever. Browser Network tab shows the events request as
 *   "(pending)" with no chunks arriving.
 *
 * Workaround:
 *   - In LOCAL DEV (window.location.hostname is localhost or 127.0.0.1),
 *     return an ABSOLUTE URL pointing directly at the FastAPI backend
 *     (default http://localhost:8000). The browser connects directly
 *     to uvicorn, bypassing the Next.js dev proxy entirely.
 *   - Override via NEXT_PUBLIC_API_BASE_URL env var if the backend
 *     runs on a different port/host.
 *   - In SANDBOX PREVIEW (preview.space-z.ai), keep the relative
 *     path — Caddy's `?XTransformPort=8000` query handles SSE
 *     streaming correctly.
 *
 * Auth: JWT access token passed as `?token=<access_token>` query
 * param (EventSource browser API doesn't support custom headers).
 */
export function getScanEventsUrl(scanId: string): string {
  const accessToken = tokenStorage.getAccessToken();
  const { baseUrl } = getApiConfig();

  // Resolve the SSE backend origin.
  // Priority:
  //   1. NEXT_PUBLIC_API_BASE_URL (explicit override)
  //   2. localhost dev bypass — point directly at FastAPI :8000
  //      to skip the Next.js dev proxy (which buffers SSE)
  //   3. Sandbox preview — relative path (Caddy handles SSE fine)
  let sseOrigin: string;
  if (baseUrl) {
    sseOrigin = baseUrl.replace(/\/$/, "");
  } else if (
    typeof window !== "undefined" &&
    (window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "0.0.0.0")
  ) {
    // Local dev bypass — connect directly to FastAPI.
    // Read port from env if the user runs backend on a non-default port.
    sseOrigin = process.env.NEXT_PUBLIC_SSE_BASE_URL || "http://localhost:8000";
  } else if (
    typeof window !== "undefined" &&
    window.location.hostname.endsWith(".space-z.ai")
  ) {
    // Sandbox preview — relative path; Caddy routes via ?XTransformPort=8000
    sseOrigin = "";
  } else {
    // Unknown host — assume backend is on the same origin (production)
    sseOrigin = typeof window !== "undefined" ? window.location.origin : "";
  }

  const url = new URL(`/api/scans/${scanId}/events`, sseOrigin || "http://localhost");

  // Sandbox preview needs the XTransformPort query so Caddy routes to FastAPI
  if (
    !baseUrl &&
    typeof window !== "undefined" &&
    window.location.hostname.endsWith(".space-z.ai")
  ) {
    url.searchParams.set("XTransformPort", SANDBOX_BACKEND_PORT);
  }
  if (accessToken) {
    url.searchParams.set("token", accessToken);
  }
  // Return absolute URL — EventSource requires absolute URL when origin differs
  // from the page origin (which is exactly what we want for the dev bypass).
  return url.toString();
}