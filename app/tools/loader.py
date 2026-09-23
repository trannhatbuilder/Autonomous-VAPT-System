"""
VAPT-AI tool loader — reads YAML tool definitions + registers as MCP tools.

Loads all *.yaml files from app/tools/ at startup, parses them into ToolDef
dataclasses, and registers each as an MCP tool on the FastMCP server.

Usage:
    from app.tools.loader import load_all_tools, get_tool, list_tools
    tools = load_all_tools()  # call once at app startup

    # Then register with MCP server (see register_tools_with_mcp below)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).parent


# ---------- Data classes ----------

@dataclass
class ParameterSpec:
    """Specification for a single tool parameter."""
    name: str
    type: str  # string, integer, bool, array
    description: str
    required: bool = False
    flag: str | None = None  # e.g. "-p", "--data"
    position: int | None = None  # positional arg position (1-indexed)
    default: Any = None
    options: list[Any] | None = None  # enum options

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema property for MCP input schema."""
        type_map = {
            "string": "string",
            "integer": "integer",
            "bool": "boolean",
            "boolean": "boolean",
            "array": "array",
        }
        prop: dict[str, Any] = {
            "type": type_map.get(self.type, "string"),
            "description": self.description,
        }
        if self.default is not None:
            prop["default"] = self.default
        if self.options:
            prop["enum"] = self.options
        return prop


@dataclass
class OutputSpec:
    """Output handling configuration."""
    max_bytes: int = 50000
    format: str = "text"
    parse_hints: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolDef:
    """Definition of a single MCP tool (parsed from YAML)."""
    name: str
    command: str
    enabled: bool = True
    description: str = ""
    short_description: str = ""
    category: str = ""
    wstg_ids: list[str] = field(default_factory=list)
    mitre_attack: list[str] = field(default_factory=list)
    safety_class: str = "active"  # passive, active, destructive
    parameters: list[ParameterSpec] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    timeout: int = 300
    allowed_exit_codes: list[int] = field(default_factory=lambda: [0])
    forbidden_args: list[str] = field(default_factory=list)  # blocked by scope_guard

    def to_mcp_input_schema(self) -> dict[str, Any]:
        """Build MCP input schema (JSON Schema) from parameters."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def build_command_args(self, **kwargs: Any) -> list[str]:
        """Build CLI args list from kwargs based on parameter specs.

        Returns a list suitable for subprocess.Popen: [arg1, arg2, ...]
        (without the command itself — caller prepends self.command).
        """
        args: list[str] = []
        # Build lookup by name
        spec_by_name = {p.name: p for p in self.parameters}

        # Sort: positional args first (by position), then flagged args
        positional = sorted(
            [p for p in self.parameters if p.position is not None],
            key=lambda p: p.position,
        )
        flagged = [p for p in self.parameters if p.flag is not None]
        # "additional_args" is special — appended last
        additional = spec_by_name.get("additional_args")

        # Add positional args in order
        for p in positional:
            val = kwargs.get(p.name, p.default)
            if val is not None and val != "" and val is not False:
                args.append(str(val))

        # Add flagged args
        for p in flagged:
            if p.name == "additional_args":
                continue  # handle last
            val = kwargs.get(p.name, p.default)
            if val is None or val == "" or val is False:
                continue
            if p.type in ("bool", "boolean"):
                if val:  # True
                    args.append(p.flag)
            elif p.type == "integer" and val == 0:
                # Skip integer 0 (default value — don't pass)
                if p.default != 0:
                    args.append(p.flag)
                    args.append(str(val))
            else:
                args.append(p.flag)
                args.append(str(val))

        # Append additional_args last
        if additional:
            val = kwargs.get("additional_args", additional.default)
            if val:
                # Shell-split the additional args string
                import shlex
                args.extend(shlex.split(str(val)))

        return args


# ---------- Loader ----------

_loaded_tools: dict[str, ToolDef] = {}


def _parse_yaml(data: dict[str, Any]) -> ToolDef:
    """Parse a YAML dict into a ToolDef."""
    params = []
    for p in data.get("parameters", []):
        params.append(ParameterSpec(
            name=p["name"],
            type=p.get("type", "string"),
            description=p.get("description", ""),
            required=p.get("required", False),
            flag=p.get("flag"),
            position=p.get("position"),
            default=p.get("default"),
            options=p.get("options"),
        ))

    output_data = data.get("output", {})
    output = OutputSpec(
        max_bytes=output_data.get("max_bytes", 50000),
        format=output_data.get("format", "text"),
        parse_hints=output_data.get("parse_hints", {}),
    )

    return ToolDef(
        name=data["name"],
        command=data["command"],
        enabled=data.get("enabled", True),
        description=data.get("description", ""),
        short_description=data.get("short_description", ""),
        category=data.get("category", ""),
        wstg_ids=data.get("wstg_ids", []),
        mitre_attack=data.get("mitre_attack", []),
        safety_class=data.get("safety_class", "active"),
        parameters=params,
        output=output,
        timeout=data.get("timeout", 300),
        allowed_exit_codes=data.get("allowed_exit_codes", [0]),
        forbidden_args=data.get("forbidden_args", []),
    )


def load_all_tools() -> dict[str, ToolDef]:
    """Load all enabled tool YAMLs from app/tools/.

    Returns dict: tool_name -> ToolDef
    """
    global _loaded_tools
    _loaded_tools.clear()

    yaml_files = sorted(TOOLS_DIR.glob("*.yaml"))
    logger.info("Loading tool definitions from %s (%d YAML files)", TOOLS_DIR, len(yaml_files))

    for yaml_file in yaml_files:
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            tool = _parse_yaml(data)
            if not tool.enabled:
                logger.info("Skipping disabled tool: %s", tool.name)
                continue
            _loaded_tools[tool.name] = tool
            logger.info("Loaded tool: %s (%s, %s, %d params)",
                        tool.name, tool.category, tool.safety_class, len(tool.parameters))
        except Exception as e:
            logger.error("Failed to load %s: %s", yaml_file, e)

    logger.info("Loaded %d tools: %s", len(_loaded_tools), list(_loaded_tools.keys()))
    return _loaded_tools


def get_tool(name: str) -> ToolDef | None:
    """Get a loaded ToolDef by name. Returns None if not found."""
    return _loaded_tools.get(name)


def list_tools() -> list[ToolDef]:
    """Get all loaded ToolDefs."""
    return list(_loaded_tools.values())


# ---------- MCP registration ----------

def _build_scope_for_target(target: str):
    """Build a permissive-but-safe ScopeGuard for one scan target.

    P1: allows the target host + any subdomain of its base domain.
    Example: target="https://pentest-ground.com:4280/" allows:
        - pentest-ground.com
        - *.pentest-ground.com  (any subdomain)
    IPs discovered via DNS resolution during recon still need to be added
    explicitly (P2 will add a runtime scope-expand API).

    Returns a configured ScopeGuard instance.
    """
    from app.sandbox.scope_guard import ScopeGuard, ScopeRule
    from urllib.parse import urlparse
    import ipaddress

    if not target:
        return ScopeGuard(declared_scope=[])  # empty scope = blocks all target checks

    # Extract host from URL/IP
    host = target
    if "://" in target:
        try:
            parsed = urlparse(target)
            host = parsed.hostname or target
        except Exception:
            pass

    rules: list[ScopeRule] = []

    # If it's an IP, allow that exact IP
    try:
        ip = ipaddress.ip_address(host)
        rules.append(ScopeRule(host=str(ip)))
    except ValueError:
        # It's a hostname — allow exact + wildcard subdomain
        rules.append(ScopeRule(host=host))
        # Also allow *.host (subdomain glob)
        # E.g. host="pentest-ground.com" → allow "*.pentest-ground.com"
        if "." in host:
            rules.append(ScopeRule(host=f"*.{host}"))

    return ScopeGuard(declared_scope=rules)


def register_tools_with_mcp(mcp_server) -> None:
    """Register all loaded tools as MCP tools on the FastMCP server.

    Each tool becomes callable via MCP tools/call.
    The MCP tool's input schema is derived from the YAML parameters.

    P1: real subprocess execution via SubprocessExecutor.
    P2: tool_func now routes through ExecutionService — the single
    substrate for all tool calls (both MCP HTTP + agent loop).
    Benefits:
        - Hard timeout (per-tool — kills the asyncio.Task if SubprocessExecutor
          hangs on I/O)
        - Cancellation (panic button — cancel by execution_id)
        - Concurrency cap (default 16 — semaphore limits parallel tools)
        - In-memory status dict (queried via get_tool_execution meta-tool)
        - Normalized result shape (all tool results funnel through one place)

    Per-call scope: built from the `target` kwarg (every YAML tool defines
    `target` as the first positional parameter). Subdomain glob is added
    automatically so recon-discovered subdomains of the declared target
    are still in scope.

    Usage (in app/main.py or app/mcp/server.py):
        from app.tools.loader import load_all_tools, register_tools_with_mcp
        from app.mcp.server import mcp_server
        load_all_tools()
        register_tools_with_mcp(mcp_server)
    """
    from app.sandbox.executor import SubprocessExecutor
    from app.mcp.execution_service import (
        get_execution_service, ExecutionStatus,
    )

    svc = get_execution_service()

    for tool in _loaded_tools.values():
        def make_tool_func(tool_def: ToolDef):
            async def tool_func(**kwargs: Any) -> str:
                """Execute the tool via ExecutionService + SubprocessExecutor.

                P2: routes through ExecutionService for unified timeout/cancel/
                status tracking. Same substrate as the agent loop's
                execute_tool_call() — both paths share identical semantics.

                Args (passed by LLM via MCP tools/call):
                    target: Target IP/hostname/URL (required — always param #1)
                    ...other params per YAML schema (ports, scan_type, etc.)

                Returns:
                    JSON string. Shape depends on final execution status:
                        COMPLETED      → {status:"executed", stdout, exit_code, ...}
                        HARD_TIMEOUT   → {status:"hard_timeout", error, execution_id}
                        CANCELLED      → {status:"cancelled", error, execution_id}
                        FAILED         → {status:"failed", error, execution_id}
                """
                import json

                target = kwargs.get("target", "")
                args = tool_def.build_command_args(**kwargs)
                cmd = [tool_def.command] + args
                cmd_str = " ".join(cmd)

                scope_guard = _build_scope_for_target(target)
                executor = SubprocessExecutor(scope_guard=scope_guard)

                logger.info(
                    "MCP tool execute | tool=%s | target=%s | cmd=%s",
                    tool_def.name, target, cmd_str[:120],
                )

                # The run closure — ExecutionService wraps this with timeout
                # + cancel + concurrency cap. Closure receives cancel_event
                # but SubprocessExecutor doesn't poll it (relies on asyncio.Task
                # cancellation propagating to subprocess via process-group kill).
                async def run(cancel_event) -> dict[str, Any]:
                    result = await executor.execute(
                        command=cmd,
                        target=target,
                        timeout=tool_def.timeout,
                        allowed_exit_codes=tool_def.allowed_exit_codes,
                        scan_id=kwargs.get("_scan_id"),
                        actor_id=kwargs.get("_actor_id", "mcp_caller"),
                    )
                    return result.to_dict()

                execution = await svc.submit(
                    tool_name=tool_def.name,
                    arguments=kwargs,
                    target=target,
                    run=run,
                    scan_id=kwargs.get("_scan_id"),
                    actor_id=kwargs.get("_actor_id", "mcp_caller"),
                    hard_timeout=tool_def.timeout,
                )

                # Build response based on final status
                response: dict[str, Any] = {
                    "tool": tool_def.name,
                    "command": cmd_str,
                    "target": target,
                    "wstg_ids": tool_def.wstg_ids,
                    "mitre_attack": tool_def.mitre_attack,
                    "safety_class": tool_def.safety_class,
                    "execution_id": execution.id,
                    "status": execution.status.value,
                    "duration_seconds": execution.duration_seconds,
                }

                if execution.status == ExecutionStatus.COMPLETED and execution.result:
                    result = execution.result
                    response.update({
                        "exit_code": result.get("exit_code"),
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "truncated": result.get("truncated", False),
                        "spill_path": result.get("spill_path"),
                        "scope_violation": result.get("scope_violation", False),
                    })
                    # Override "status" to "executed" for backward compat with
                    # the field the agent loop expects (was "executed" | "error")
                    response["status"] = "executed" if result.get("success") else (
                        "scope_violation" if result.get("scope_violation") else "error"
                    )
                else:
                    # Terminal but not completed (timeout/cancel/fail)
                    response["error"] = execution.error or "Unknown error"
                    response["hint"] = (
                        "Poll status via get_tool_execution(execution_id="
                        f"{execution.id}) or retry with different parameters."
                    )

                return json.dumps(response, indent=2, ensure_ascii=False)

            tool_func.__name__ = tool.name
            tool_func.__doc__ = tool.description
            return tool_func

        func = make_tool_func(tool)
        mcp_server.tool(name=tool.name, description=tool.short_description or tool.description[:200])(func)
        logger.info("Registered MCP tool: %s (ExecutionService-backed — P2)", tool.name)


if __name__ == "__main__":
    # Quick test — load + print all tools
    logging.basicConfig(level=logging.INFO)
    tools = load_all_tools()
    print(f"\nLoaded {len(tools)} tools:")
    for name, t in tools.items():
        print(f"  - {name}: {t.short_description}")
        print(f"    category={t.category}, safety={t.safety_class}, wstg={t.wstg_ids}")
        print(f"    params: {[p.name for p in t.parameters]}")
        print()