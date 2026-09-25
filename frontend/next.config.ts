import type { NextConfig } from "next";

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

const nextConfig: NextConfig = {
  output: "standalone",
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