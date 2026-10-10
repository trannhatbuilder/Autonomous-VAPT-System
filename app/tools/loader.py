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

# Bundled fallback wordlist (shipped with the repo). Used when a tool YAML
# requests a path like /usr/share/seclists/... that isn't installed on the host,
# so fuzzing tools still run instead of burning agent turns hunting for files.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLED_WORDLIST = _PROJECT_ROOT / "data" / "wordlists" / "common.txt"


def _resolve_wordlist(value: str | None) -> str | None:
    """Return a usable wordlist path for a `wordlist` parameter.

    Order:
      1. the caller-provided path, if it exists on disk;
      2. the bundled repo wordlist (data/wordlists/common.txt);
      3. the original value (so the tool still reports a meaningful error).

    A warning is logged on substitution so operators can install the real
    SecLists/dirb wordlists and get richer results.
    """
    if value:
        try:
            if Path(value).is_file():
                return value
        except OSError:
            pass
    if BUNDLED_WORDLIST.is_file():
        logger.warning(
            "Wordlist %r not found — substituting bundled %s",
            value, BUNDLED_WORDLIST,
        )
        return str(BUNDLED_WORDLIST)
    return value


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
    # Literal text appended to the arg VALUE (e.g. "; exit" for a shell/REPL
    # command). Lets a wrapper guarantee non-interactive execution.
    suffix: str = ""
    # Go-duration flag (e.g. gobuster's `--timeout`/`--to`, which wants "10s").
    # When true, a bare numeric value is completed with a "s" unit: 15 -> "15s".
    # A value that already carries a unit ("10s", "2m") is left untouched.
    # Needed because the LLM frequently sends a bare number despite a
    # `type: string` schema, and a flag-name heuristic is ambiguous —
    # `--timeout` is a Go duration for gobuster but plain seconds for dalfox.
    duration: bool = False

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
            # Skip unset values. NOTE: a bare `val is not False` was NOT enough
            # — 0 is a distinct object from False, so 0 was emitted as the
            # literal "0". 0 means "unset" for numeric params (mirrors the
            # flagged-arg loop below), so drop it here too. Every positional
            # param is a string today, but this keeps the two loops consistent
            # (and stops a future positional `port` defaulting to 0 from
            # passing a literal "0"). Bools are unaffected: `True == 0` is
            # False, and False is already skipped above.
            if val is None or val == "" or val is False:
                continue
            if isinstance(val, (int, float)) and val == 0:
                continue
            args.append(str(val))

        # Add flagged args
        for p in flagged:
            if p.name == "additional_args":
                continue  # handle last
            val = kwargs.get(p.name, p.default)
            # Wordlist params: substitute the bundled list when the requested
            # path doesn't exist (e.g. SecLists not installed). Without this,
            # ffuf/gobuster abort immediately and the agent wastes turns
            # searching the filesystem for wordlists.
            if val and "wordlist" in p.name.lower():
                val = _resolve_wordlist(str(val))
            if val is None or val == "" or val is False:
                continue
            if p.type in ("bool", "boolean"):
                if val:  # True
                    args.append(p.flag)
            elif p.type == "integer" and val == 0:
                # 0 means "not set" for integer params — never pass it
                # explicitly. The previous version emitted `-t 0` whenever the
                # LLM sent 0 for a param whose default was non-zero, which
                # makes tools like gobuster die ("threads must be > 0").
                continue
            else:
                # Go-duration flags need a time unit. The LLM frequently sends a
                # bare number — as an int (15) or a string ("15") — despite a
                # `type: string` schema; gobuster 3.8+ rejects it ("invalid
                # value ... for flag -to"). Coerce bare numbers to "<n>s" while
                # leaving an already-formatted "10s"/"2m" untouched.
                if p.duration:
                    if isinstance(val, (int, float)):
                        val = f"{int(val)}s"
                    else:
                        sval = str(val).strip()
                        if sval.isdigit():
                            val = sval + "s"
                args.append(p.flag)
                # `suffix` appends literal text to the value — e.g. msfconsole's
                # `-x` command gets "; exit" so the REPL never blocks.
                args.append(str(val) + p.suffix)

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
_loaded_tools_mtime: float = 0.0  # mtime of the newest YAML file at last load


def _newest_yaml_mtime(yaml_files: list[Path] | None = None) -> float:
    """Return the newest mtime across the tool YAMLs (0.0 if none/unreadable).

    Cheap — a glob + one stat per file, no YAML parsing. Shared by
    load_all_tools() (cache key) and _yaml_mtime_changed() (the stale-schema
    guard consulted by tool_bridge.build_tool_schemas). Pass `yaml_files` to
    reuse an already-globbed list and avoid a redundant directory scan.
    """
    if yaml_files is None:
        try:
            yaml_files = sorted(TOOLS_DIR.glob("*.yaml"))
        except OSError:
            return 0.0
    if not yaml_files:
        return 0.0
    try:
        return max(f.stat().st_mtime for f in yaml_files)
    except OSError:
        return 0.0


def _yaml_mtime_changed() -> bool:
    """True if a tool YAML was edited since the last successful load.

    Stat-only (no re-parse). Consumed by
    app.agents.tool_bridge.build_tool_schemas() so that a memoized schema is
    NOT served after a live YAML edit: that function early-returns on a cache
    hit and would otherwise never reach load_all_tools() — whose re-parse is
    the ONLY thing that runs the schema-cache invalidation. In a long-running
    process (the MCP server) that meant an edited YAML stayed invisible until
    a restart.

    Returns False on a cold cache (nothing loaded yet) so we never report a
    spurious change before the first load.
    """
    if not _loaded_tools:
        return False
    return _newest_yaml_mtime() != _loaded_tools_mtime


def _parse_yaml(data: dict[str, Any]) -> ToolDef:
    """Parse a YAML dict into a ToolDef."""
    params = []
    for p in data.get("parameters", []):
        # Guard: a param declaring BOTH `position` and `flag` is appended TWICE
        # by build_command_args (once in the positional loop, once in the
        # flagged loop) — producing a duplicated arg. Warn at parse time so the
        # author notices; the YAML should use one or the other, not both.
        if p.get("position") is not None and p.get("flag"):
            logger.warning(
                "Tool %s param %s has BOTH position and flag — it will be "
                "emitted twice in the command args",
                data.get("name", "?"), p.get("name", "?"),
            )
        params.append(ParameterSpec(
            name=p["name"],
            type=p.get("type", "string"),
            description=p.get("description", ""),
            required=p.get("required", False),
            flag=p.get("flag"),
            position=p.get("position"),
            default=p.get("default"),
            options=p.get("options"),
            suffix=p.get("suffix", ""),
            duration=p.get("duration", False),
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


def load_all_tools(force_reload: bool = False) -> dict[str, ToolDef]:
    """Load all enabled tool YAMLs from app/tools/.

    CACHED — only re-parses YAML files when the directory contents or any
    YAML file's mtime has changed since the last call. Subsequent calls
    during the same scan (including one ReAct loop's many iterations)
    return the cached dict in O(1) without touching disk.

    Args:
        force_reload: bypass the mtime check and re-parse from disk.
            Useful for tests / hot-reload during development.

    Returns dict: tool_name -> ToolDef
    """
    global _loaded_tools, _loaded_tools_mtime

    # Compute the newest mtime across the YAML directory contents.
    # If unchanged and cache is populated, return cache immediately.
    yaml_files = sorted(TOOLS_DIR.glob("*.yaml"))
    if not yaml_files:
        # Empty / missing dir — return whatever we have (likely empty)
        return _loaded_tools

    newest_mtime = _newest_yaml_mtime(yaml_files)

    if (
        not force_reload
        and _loaded_tools
        and newest_mtime == _loaded_tools_mtime
    ):
        # Cache hit — log at debug level only to keep scan logs clean
        logger.debug(
            "Tool cache hit (%d tools, mtime=%s) — skipping re-parse",
            len(_loaded_tools), _loaded_tools_mtime,
        )
        return _loaded_tools

    # Cache miss — re-parse all YAMLs from disk
    _loaded_tools.clear()
    _loaded_tools_mtime = newest_mtime

    logger.info(
        "Loading tool definitions from %s (%d YAML files, mtime=%s)",
        TOOLS_DIR, len(yaml_files), newest_mtime,
    )

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

    # The OpenAI tool schemas the LLM sees are memoized in tool_bridge. A YAML
    # edit changes mtime → we just re-parsed the ToolDefs above, so drop the
    # memoized schemas too; otherwise the ReAct loop keeps advertising the OLD
    # description/parameters until someone explicitly calls reload_tools().
    # Lazy import avoids a circular import (tool_bridge imports this module).
    try:
        from app.agents.tool_bridge import invalidate_tool_schema_cache
        invalidate_tool_schema_cache()
        logger.info("Invalidated tool schema cache (YAML mtime changed)")
    except Exception as exc:  # pragma: no cover — defensive: loader must not
        # hard-fail if tool_bridge is unavailable (e.g. partial install).
        logger.warning("Could not invalidate tool schema cache: %s", exc)

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
    """Build a permissive ScopeGuard for one scan target.

    Phase 3: ScopeGuard is disabled via env VAPT_AI_SCOPE_GUARD_DISABLED=1
    (default). This function still returns a ScopeGuard instance for backward
    compatibility with SubprocessExecutor (which expects a scope_guard arg),
    but the guard's validate_command() always returns allowed=True.

    Returns a configured ScopeGuard instance (permissive — all targets allowed).
    """
    # Phase 3: keep importing ScopeGuard for now (module still exists).
    # When scope_guard.py is deleted in a future phase, this will be replaced
    # with a no-op stub class.
    try:
        from app.sandbox.scope_guard import ScopeGuard
        return ScopeGuard(declared_scope=[])  # empty = permissive (env flag disables checks)
    except ImportError:
        # Fallback: scope_guard.py already deleted — return a no-op stub
        class _NoopScopeGuard:
            def validate_command(self, command, target=""):
                from collections import namedtuple
                R = namedtuple("ValidationResult", ["allowed", "command", "target", "reason", "severity"])
                return R(allowed=True, command=command, target=target, reason="noop", severity="info")
        return _NoopScopeGuard()


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


def reload_tools() -> dict[str, ToolDef]:
    """Force-reload tool definitions from disk (alias for load_all_tools(force_reload=True)).

    Useful when a tool YAML has been edited at runtime (e.g. during dev) —
    safe to call from any thread, idempotent.

    Also invalidates the agent's memoized LLM tool-schema cache
    (app.agents.tool_bridge._tool_schema_cache). Without this, an edited
    YAML would reload ToolDefs but the ReAct loop would keep sending the
    STALE schema (old description/parameters) to the LLM. The import is
    deferred to avoid a circular import at module load time (tool_bridge
    imports this module at the top level).
    """
    tools = load_all_tools(force_reload=True)
    try:
        from app.agents.tool_bridge import invalidate_tool_schema_cache
        invalidate_tool_schema_cache()
    except Exception as exc:  # pragma: no cover — defensive: loader must not
        # hard-fail if tool_bridge is unavailable (e.g. partial install).
        logger.warning("Could not invalidate tool schema cache after reload: %s", exc)
    return tools


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