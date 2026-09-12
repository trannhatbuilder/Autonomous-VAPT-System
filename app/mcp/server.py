"""
VAPT-AI MCP (Model Context Protocol) server skeleton — W1-G.

This is the W1 skeleton: responds to `tools/list` (empty list for now) and
`tools/call` (returns "no tools registered yet"). W2-W5 will register 30+
security tool wrappers here.

The legacy EVVO Metasploit MCP server (mcp_servers/metasploit/) is kept
as-is for W6 task (Metasploit RPC integration). It's a separate MCP server
that talks to msfrpcd via msgpack — different concern from this VAPT-AI
MCP server which wraps 30+ CLI tools.

Transports supported:
    - stdio  : for spawning as subprocess by LangGraph agents
    - HTTP   : for direct testing (GET /mcp/tools/list)
    - SSE    : for streaming (deferred to W3)

W1-G acceptance: MCP server responds to `tools/list` with an empty list.
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ---------- MCP server instance ----------

mcp_server = FastMCP(
    "vapt-ai",
    instructions="VAPT-AI v3.2.1 MCP server — W1 skeleton. 30+ security tool wrappers will be added in W2-W5.",
)


# ---------- W1 skeleton: empty tools/list ----------

@mcp_server.tool()
async def ping() -> str:
    """Health check — returns 'pong'.

    Useful for verifying the MCP server is alive without needing any
    external dependencies (msfrpcd, security tools, etc.).
    """
    return "pong"


@mcp_server.tool()
async def server_info() -> str:
    """Return VAPT-AI MCP server info (version, registered tools count)."""
    import json
    return json.dumps({
        "name": "vapt-ai",
        "version": "3.2.1",
        "tools_registered": 2,  # ping + server_info (W1 skeleton)
        "note": "W1 skeleton — 30+ security tool wrappers will be added in W2-W5",
    })


# ---------- W1-G verification helpers ----------

async def list_tools_async() -> list[dict[str, Any]]:
    """Return the list of registered MCP tools (for HTTP endpoint + tests).

    This is the canonical way to verify `tools/list` works.
    """
    # FastMCP stores tools in mcp_server._tool_manager._tools
    # The exact attribute may vary by mcp SDK version — use list_tools() if available
    try:
        # mcp SDK >= 1.0
        tools = await mcp_server.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else None,
            }
            for t in tools
        ]
    except Exception:
        # Fallback: inspect internal state
        try:
            tools_dict = mcp_server._tool_manager._tools  # type: ignore[attr-defined]
            return [
                {
                    "name": name,
                    "description": getattr(tool, "description", None),
                }
                for name, tool in tools_dict.items()
            ]
        except Exception as e:
            logger.error("Failed to list MCP tools: %s", e)
            return []


def list_tools_sync() -> list[dict[str, Any]]:
    """Synchronous version of list_tools_async (for FastAPI routes)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context — create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, list_tools_async()).result()
    except RuntimeError:
        pass
    return asyncio.run(list_tools_async())


# ---------- Stdio entry point ----------

def run_stdio() -> None:
    """Run MCP server in stdio transport mode.

    Use this when spawning VAPT-AI MCP server as a subprocess from
    LangGraph agents:
        subprocess.Popen(['python', '-m', 'app.mcp.server'], stdin=PIPE, stdout=PIPE)

    Logs go to stderr (NEVER stdout — would corrupt JSON-RPC).
    """
    import sys
    print("[vapt-ai-mcp] Starting stdio transport", file=sys.stderr)
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()
