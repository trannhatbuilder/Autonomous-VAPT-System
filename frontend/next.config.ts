import type { NextConfig } from "next";
import os from "node:os";

/**
 * VAPT-AI Next.js config — Phase A (frontend/backend split).
 *
 * - `npm run dev` → http://localhost:3000 (UI)
 * - FastAPI backend → http://localhost:8000 (pure JSON API, no SPA)
 * - In dev, Next.js rewrites() proxies /api/*, /mcp/*, /docs, /health, /openapi.json
 *   to the FastAPI port, so the frontend never needs absolute URLs or env vars.
 * - In sandbox preview (preview.space-z.ai), Caddy already routes via
 *   `?XTransformPort=8000`, so rewrites only fire in dev mode.
 */
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

/**
 * Next.js 16 blocks cross-origin requests to dev-only resources (HMR and
 * /_next/* chunks) unless the browser's origin is listed here. When the UI is
 * opened via the LAN "Network" URL (e.g. http://192.168.1.11:3000) instead of
 * localhost, every JS chunk is blocked and the page renders BLANK:
 *
 *   "Blocked cross-origin request to Next.js dev resource /_next/hmr"
 *
 * We auto-collect this machine's non-internal IPv4 addresses so the UI keeps
 * working when the box's IP changes (DHCP/VPN), and merge in anything passed
 * via ALLOWED_DEV_ORIGINS (comma-separated) for proxied/preview hostnames.
 */
function devOrigins(): string[] {
  const fromEnv = (process.env.ALLOWED_DEV_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const lan: string[] = [];
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const addr of addrs ?? []) {
      if (addr.family === "IPv4" && !addr.internal) {
        lan.push(addr.address);
      }
    }
  }

  return Array.from(
    new Set(["localhost", "127.0.0.1", ...lan, ...fromEnv]),
  );
}

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: devOrigins(),
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  async rewrites() {
    // Only apply in dev — production uses Caddy gateway.
    if (process.env.NODE_ENV === "production") {
      return [];
    }
    return [
      // API endpoints (all /api/* routes)
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      // MCP tool endpoints (FastAPI serves /mcp/* directly)
      {
        source: "/mcp/:path*",
        destination: `${BACKEND_URL}/mcp/:path*`,
      },
      // Swagger / OpenAPI / health (useful during dev)
      {
        source: "/docs",
        destination: `${BACKEND_URL}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${BACKEND_URL}/openapi.json`,
      },
      {
        source: "/health",
        destination: `${BACKEND_URL}/health`,
      },
    ];
  },
};

export default nextConfig;