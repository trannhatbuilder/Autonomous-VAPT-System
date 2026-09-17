"""
VAPT-AI Audit Agent — LLM critic that reviews destructive ops proposed by the
main agent. Ports CyberStrikeAI internal/handler/hitl_audit_agent.go (360 LOC Go).

Design (Option C — A-in-the-Loop):
    Instead of blocking tool execution to wait for a human approval, every
    destructive op is reviewed by a second LLM (the "audit agent"). The audit
    agent decides approve / reject / suggest_alternative within ~2-3 seconds,
    so the scan flow stays autonomous. Decisions are logged to
    vapt_hitl_approvals + vapt_audit_log (D23 chain of custody) for full
    traceability. A human panic button (POST /api/scans/{id}/abort) is still
    available as a safety net.

Audit-decision outcomes:
    approve              -> tool runs as-is
    reject               -> tool does NOT run; agent receives the comment as
                            feedback and must justify or pick an alternative
    suggest_alternative  -> tool runs with `suggested_args` substituted in
                            (narrower scope, safer flags)

Graceful degradation:
    If the LLM call fails (timeout, API error, malformed JSON), the audit
    agent falls back to `reject` with a clear comment. This is the safe
    default — better to skip one tool call than to run a destructive op
    without review. The behavior matches CyberStrikeAI's "保守拒绝" policy.
    Set VAPT_AI_HITL_AUDIT_FALLBACK=approve to flip the policy (debug only).

Usage:
    from app.hitl.audit_agent import AuditAgent, AuditDecision

    agent = AuditAgent()  # singleton via get_audit_agent()
    decision = await agent.review(
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit", "module_name": "exploit/windows/smb/ms17_010_eternalblue"},
        predicted_impact="destructive",
        agent_reasoning="Exploit MS17-010 to obtain a Meterpreter session",
        scan_context={
            "declared_scope": ["10.10.10.0/24"],
            "blackboard_facts": [...],
            "destructive_ops_count": 1,
            "max_destructive_ops": 5,
        },
    )
    if decision.decision == "approve":
        # proceed with original args
    elif decision.decision == "reject":
        # skip tool; feed decision.comment back to agent
    elif decision.decision == "suggest_alternative":
        # use decision.suggested_args instead
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------- Defaults ----------

DEFAULT_AUDIT_MODEL = "gpt-4o-mini"
DEFAULT_AUDIT_TEMPERATURE = 0.1
DEFAULT_AUDIT_MAX_TOKENS = 1024
DEFAULT_AUDIT_TIMEOUT_SECONDS = 90

# Cap payload fields sent to LLM (avoid token blowup)
MAX_PAYLOAD_FIELD_CHARS = 2000
MAX_BLACKBOARD_FACTS = 5
MAX_KG_PATHS = 3


# ---------- Audit-decision prompt (English per §2.3 — code/docs in English) ----------

AUDIT_AGENT_PROMPT_BASE = """You are the VAPT-AI Human-in-the-Loop audit agent. Your role is to review a destructive operation proposed by another AI agent and decide whether it should run.

You will receive a JSON object containing: hitlMode, toolName, target, args, predictedImpact, agentReasoning, and scanContext (declared scope, blackboard facts, KG paths, destructive-ops count).

Default stance (lean toward approve):
- Routine, low-risk pentest operations -> approve (information gathering, port/service scanning, directory enumeration, read-only queries, harmless probes)
- Operations consistent with the user's declared scope and the current task goal, with no clear high-risk signal -> approve
- Only reject when there is a real possibility of substantial harm to the target system

Must-reject high-risk scenarios (examples, not exhaustive):
- Database deletion, table clearing, bulk data deletion, disk formatting, irreversible destruction
- Modifying or resetting passwords, creating or tampering with administrator accounts, persisting backdoors, auto-start on boot
- Writing malicious payloads to production, ransomware encryption, stopping critical services, modifying core system config
- Obvious privilege escalation beyond task scope: destructive ops unrelated to the task or declared scope

Should NOT be a standalone reason to reject:
- Routine nmap/curl/grep/read-file/enumeration commands themselves
- Slightly broad parameters but no clear destructive intent (in review_edit mode you may narrow parameters then approve)
- "Insufficient information" alone -- if no high-risk signal above, you should approve and may add a cautionary comment

Important: this is single-user VAPT-AI on owned/authorized targets. False rejections slow down legitimate pentest work. Approve by default; reject only on clear risk."""

AUDIT_AGENT_PROMPT_APPROVAL_OUTPUT = """

Output a single JSON line, no markdown code fences:
{"decision":"approve"|"reject","comment":"short rationale"}"""

AUDIT_AGENT_PROMPT_REVIEW_EDIT_OUTPUT = """

Output a single JSON line, no markdown code fences:
{"decision":"approve"|"reject","comment":"short rationale","suggested_args":{...}}

suggested_args rules (only fill when decision=approve AND args need narrowing; otherwise omit the field):
- Provide the FULL replacement args object (same keys as the input args)
- Make only the minimal changes needed to narrow scope or remove risky flags
- NEVER widen attack surface: do not expand target range, escalate privileges, or introduce destructive parameters
- If you cannot safely narrow the args, reject instead -- do not勉强 approve"""


# ---------- Data structures ----------

@dataclass
class AuditDecision:
    """Outcome of an audit-agent review."""

    decision: str  # approve | reject | suggest_alternative
    comment: str
    suggested_args: dict[str, Any] | None = None
    risk_assessment: str | None = None
    confidence: float = 0.0
    decided_by: str = "audit_agent"
    duration_seconds: float = 0.0
    raw_llm_response: str | None = None  # for debugging only — never expose to user

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "comment": self.comment,
            "suggested_args": self.suggested_args,
            "risk_assessment": self.risk_assessment,
            "confidence": self.confidence,
            "decided_by": self.decided_by,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AuditReviewInput:
    """Input bundle for an audit review (sent to LLM as user message)."""

    hitl_mode: str = "approval"
    tool_name: str = ""
    target: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    predicted_impact: str = ""
    agent_reasoning: str = ""
    scan_context: dict[str, Any] = field(default_factory=dict)

    def to_user_message(self) -> str:
        """Render as pretty-printed JSON for the LLM user message."""
        payload: dict[str, Any] = {
            "hitlMode": self.hitl_mode,
            "toolName": self.tool_name,
            "target": self.target,
            "args": _truncate_dict(self.args),
            "predictedImpact": self.predicted_impact,
            "agentReasoning": _truncate_str(self.agent_reasoning),
        }
        # Scan context (only the fields useful for review)
        ctx = self.scan_context or {}
        scan_subset: dict[str, Any] = {}
        if "declared_scope" in ctx:
            scan_subset["declaredScope"] = ctx["declared_scope"]
        if "destructive_ops_count" in ctx and "max_destructive_ops" in ctx:
            scan_subset["destructiveOpsCount"] = ctx["destructive_ops_count"]
            scan_subset["maxDestructiveOps"] = ctx["max_destructive_ops"]
        facts = ctx.get("blackboard_facts") or []
        if facts:
            scan_subset["blackboardFacts"] = facts[:MAX_BLACKBOARD_FACTS]
        paths = ctx.get("kg_paths") or []
        if paths:
            scan_subset["kgTopPaths"] = paths[:MAX_KG_PATHS]
        if scan_subset:
            payload["scanContext"] = scan_subset
        return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


# ---------- Audit Agent ----------

class AuditAgent:
    """LLM-based audit agent that reviews destructive ops.

    Singleton via get_audit_agent(). Tests should construct directly with
    custom llm_caller to inject mocks.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = DEFAULT_AUDIT_TEMPERATURE,
        max_tokens: int = DEFAULT_AUDIT_MAX_TOKENS,
        timeout_seconds: int = DEFAULT_AUDIT_TIMEOUT_SECONDS,
        mode: str = "approval",  # approval | review_edit
        fallback_decision: str = "reject",  # reject (safe) | approve (debug only)
        llm_caller: Any = None,  # inject for tests
    ):
        self.model = (model or _resolve_audit_model()).strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.mode = _normalize_mode(mode)
        self.fallback_decision = _normalize_fallback(fallback_decision)
        self._llm_caller = llm_caller  # if None, use litellm at call time

    # ---------- Public API ----------

    async def review(
        self,
        tool_name: str,
        target: str,
        args: dict[str, Any],
        predicted_impact: str,
        agent_reasoning: str,
        scan_context: dict[str, Any] | None = None,
    ) -> AuditDecision:
        """Review a destructive op. Returns an AuditDecision (never raises)."""
        start = time.time()
        review_input = AuditReviewInput(
            hitl_mode=self.mode,
            tool_name=tool_name,
            target=target,
            args=args or {},
            predicted_impact=predicted_impact,
            agent_reasoning=agent_reasoning,
            scan_context=scan_context or {},
        )

        # If model not configured (no API key), fall back immediately
        if not self._is_llm_configured():
            logger.warning(
                "Audit agent LLM not configured (model=%s) — falling back to %s",
                self.model, self.fallback_decision,
            )
            return AuditDecision(
                decision=self.fallback_decision,
                comment=f"audit agent: LLM not configured (model={self.model}), "
                        f"fallback={self.fallback_decision}",
                decided_by="audit_agent_fallback",
                duration_seconds=time.time() - start,
            )

        system_prompt = self._build_system_prompt()
        user_content = review_input.to_user_message()

        try:
            raw_response = await self._call_llm(system_prompt, user_content)
        except Exception as exc:
            logger.warning(
                "Audit agent LLM call failed (tool=%s, model=%s): %s: %s",
                tool_name, self.model, type(exc).__name__, exc,
            )
            return AuditDecision(
                decision=self.fallback_decision,
                comment=f"audit agent: LLM call failed ({type(exc).__name__}: {exc}), "
                        f"fallback={self.fallback_decision}",
                decided_by="audit_agent_fallback",
                duration_seconds=time.time() - start,
            )

        # Parse the LLM response
        decision = self._parse_response(raw_response, tool_name=tool_name)
        decision.duration_seconds = time.time() - start
        decision.raw_llm_response = raw_response
        return decision

    # ---------- Internal: LLM call ----------

    async def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """Call the configured LLM. Returns the raw text content.

        If self._llm_caller is set (test mock), use it.
        Otherwise use litellm.acompletion().
        """
        if self._llm_caller is not None:
            # Test injection: caller is responsible for the full call signature.
            result = self._llm_caller(
                model=self.model,
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
            )
            # Allow either sync str or async coroutine
            if hasattr(result, "__await__"):
                result = await result
            return result

        # Production path: litellm.acompletion
        try:
            import asyncio
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "litellm is required for AuditAgent but is not installed. "
                "Run: pip install litellm"
            ) from exc

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # litellm.acompletion is async; we run it directly since this method is async.
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_seconds,
        )
        # Extract content from standard OpenAI-shaped response
        try:
            choices = response.choices if hasattr(response, "choices") else response["choices"]
            if not choices:
                return ""
            msg = choices[0].message if hasattr(choices[0], "message") else choices[0]["message"]
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            # Some providers put content in reasoning_content when thinking is enabled
            if not content:
                reasoning = getattr(msg, "reasoning_content", None) if hasattr(msg, "reasoning_content") else msg.get("reasoning_content", "")
                content = reasoning or ""
            return content or ""
        except (AttributeError, KeyError, IndexError) as exc:
            logger.warning("Audit agent: unexpected LLM response shape: %s", exc)
            return ""

    def _is_llm_configured(self) -> bool:
        """Check if we have an API key for the configured model provider.

        litellm resolves keys from env vars based on the model prefix
        (e.g. openai/gpt-4o-mini -> OPENAI_API_KEY, glm/glm-4 -> GLM_API_KEY).
        We do a lightweight check here; the actual call may still fail later
        if the key is invalid.
        """
        if self._llm_caller is not None:
            return True  # tests always treat as configured
        if not self.model:
            return False
        # Map common model prefixes to their env var
        prefix_map = {
            "openai/": "OPENAI_API_KEY",
            "gpt-": "OPENAI_API_KEY",
            "o1-": "OPENAI_API_KEY",
            "o3-": "OPENAI_API_KEY",
            "anthropic/": "ANTHROPIC_API_KEY",
            "claude-": "ANTHROPIC_API_KEY",
            "glm/": "GLM_API_KEY",
            "glm-": "GLM_API_KEY",
            "minimax/": "MINIMAX_API_KEY",
            "groq/": "GROQ_API_KEY",
            "deepseek/": "DEEPSEEK_API_KEY",
            "gemini/": "GEMINI_API_KEY",
            "ollama/": "OLLAMA_API_KEY",  # local — usually no key needed
        }
        for prefix, env_var in prefix_map.items():
            if self.model.lower().startswith(prefix):
                if env_var == "OLLAMA_API_KEY":
                    return True  # local ollama, no key needed
                return bool(os.environ.get(env_var))
        # Unknown provider — assume configured (litellm will raise if not)
        return True

    # ---------- Internal: prompt + parsing ----------

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt based on the configured mode."""
        if self.mode == "review_edit":
            return AUDIT_AGENT_PROMPT_BASE + AUDIT_AGENT_PROMPT_REVIEW_EDIT_OUTPUT
        return AUDIT_AGENT_PROMPT_BASE + AUDIT_AGENT_PROMPT_APPROVAL_OUTPUT

    def _parse_response(self, raw: str, tool_name: str = "") -> AuditDecision:
        """Parse the LLM response into an AuditDecision.

        Tries multiple JSON candidates (raw, strip-code-fence, first-JSON-object).
        On any failure, returns the fallback decision.
        """
        if not raw or not raw.strip():
            return AuditDecision(
                decision=self.fallback_decision,
                comment=f"audit agent: empty LLM response, fallback={self.fallback_decision}",
                decided_by="audit_agent_fallback",
            )

        for candidate in _json_candidates(raw):
            parsed = _try_parse_decision(candidate, mode=self.mode)
            if parsed is not None:
                decision, comment, suggested_args = parsed
                # In approval mode, ignore any suggested_args (only review_edit allows edits)
                if self.mode != "review_edit" and suggested_args:
                    logger.info(
                        "Audit agent returned suggested_args in approval mode (tool=%s) — ignoring",
                        tool_name,
                    )
                    suggested_args = None
                # Normalize comment prefix
                if not comment:
                    comment = f"audit agent: {decision}"
                elif not comment.lower().startswith("audit agent"):
                    comment = f"audit agent: {comment}"
                # If suggested_args present in review_edit, treat as suggest_alternative
                final_decision = decision
                if decision == "approve" and suggested_args:
                    final_decision = "suggest_alternative"
                return AuditDecision(
                    decision=final_decision,
                    comment=comment,
                    suggested_args=suggested_args,
                    decided_by="audit_agent",
                )

        snippet = raw[:240] + "..." if len(raw) > 240 else raw
        logger.warning(
            "Audit agent: failed to parse LLM response (tool=%s, mode=%s, snippet=%r)",
            tool_name, self.mode, snippet,
        )
        return AuditDecision(
            decision=self.fallback_decision,
            comment=f"audit agent: response unparseable, fallback={self.fallback_decision}",
            decided_by="audit_agent_fallback",
        )


# ---------- Singleton ----------

_audit_agent_singleton: AuditAgent | None = None


def get_audit_agent() -> AuditAgent:
    """Get the singleton AuditAgent instance.

    Reads config from settings/env on first call. Subsequent calls return
    the same instance. Tests that need a fresh instance should construct
    AuditAgent() directly.
    """
    global _audit_agent_singleton
    if _audit_agent_singleton is None:
        _audit_agent_singleton = AuditAgent(
            model=_resolve_audit_model(),
            temperature=_resolve_audit_temperature(),
            max_tokens=_resolve_audit_max_tokens(),
            timeout_seconds=_resolve_audit_timeout(),
            mode=_resolve_audit_mode(),
            fallback_decision=_resolve_audit_fallback(),
        )
    return _audit_agent_singleton


def reset_audit_agent_singleton() -> None:
    """Reset the singleton (for tests)."""
    global _audit_agent_singleton
    _audit_agent_singleton = None


# ---------- Helpers: config resolution ----------

def _resolve_audit_model() -> str:
    """Resolve the audit-agent model from env vars.

    Priority:
      1. VAPT_AI_HITL_AUDIT_MODEL (canonical)
      2. VAPT_AI_MODEL_PROVIDER + default model (e.g. glm -> glm-4-flash, openai -> gpt-4o-mini)
      3. DEFAULT_AUDIT_MODEL (gpt-4o-mini)
    """
    explicit = os.environ.get("VAPT_AI_HITL_AUDIT_MODEL", "").strip()
    if explicit:
        return explicit
    provider = (settings.model_provider or "openai").lower()
    # Map provider to a sensible default model (cheap + structured)
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "glm": "glm-4-flash",
        "minimax": "abab6.5s-chat",
        "deepseek": "deepseek-chat",
        "groq": "groq/llama-3.1-8b-instant",
        "google": "gemini/gemini-1.5-flash",
        "ollama": "ollama/llama3.1:8b",
    }
    return defaults.get(provider, DEFAULT_AUDIT_MODEL)


def _resolve_audit_temperature() -> float:
    val = os.environ.get("VAPT_AI_HITL_AUDIT_TEMPERATURE", "").strip()
    try:
        return float(val) if val else DEFAULT_AUDIT_TEMPERATURE
    except ValueError:
        return DEFAULT_AUDIT_TEMPERATURE


def _resolve_audit_max_tokens() -> int:
    val = os.environ.get("VAPT_AI_HITL_AUDIT_MAX_TOKENS", "").strip()
    try:
        return int(val) if val else DEFAULT_AUDIT_MAX_TOKENS
    except ValueError:
        return DEFAULT_AUDIT_MAX_TOKENS


def _resolve_audit_timeout() -> int:
    val = os.environ.get("VAPT_AI_HITL_AUDIT_TIMEOUT_SECONDS", "").strip()
    try:
        return int(val) if val else DEFAULT_AUDIT_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_AUDIT_TIMEOUT_SECONDS


def _resolve_audit_mode() -> str:
    """approval (default) | review_edit"""
    return _normalize_mode(os.environ.get("VAPT_AI_HITL_AUDIT_MODE", "approval"))


def _resolve_audit_fallback() -> str:
    return _normalize_fallback(
        os.environ.get("VAPT_AI_HITL_AUDIT_FALLBACK", "reject")
    )


def _normalize_mode(mode: str) -> str:
    v = (mode or "").lower().strip()
    if v in ("review_edit", "review-edit"):
        return "review_edit"
    # approval, off, "", anything else -> approval
    return "approval"


def _normalize_fallback(v: str) -> str:
    s = (v or "").lower().strip()
    return "approve" if s in ("approve", "approved", "yes", "true", "1") else "reject"


# ---------- Helpers: JSON parsing (port from Go) ----------

def _truncate_str(s: str | None, max_chars: int = MAX_PAYLOAD_FIELD_CHARS) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...(truncated)"


def _truncate_dict(d: dict[str, Any] | None) -> dict[str, Any]:
    if not d:
        return {}
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = _truncate_str(v)
        elif isinstance(v, (dict, list)):
            try:
                s = json.dumps(v, default=str, ensure_ascii=False)
                out[k] = _truncate_str(s)
            except (TypeError, ValueError):
                out[k] = "<unserializable>"
        else:
            out[k] = v
    return out


def _json_candidates(s: str) -> list[str]:
    """Generate candidate JSON strings to try parsing.

    Ports auditAgentJSONCandidates from Go:
      - raw string
      - strip markdown code fence (```json ... ``` or ``` ... ```)
      - first JSON object in raw
      - first JSON object in fence-stripped
    Dedupes while preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(c: str) -> None:
        c = (c or "").strip()
        if not c or c in seen:
            return
        seen.add(c)
        out.append(c)

    add(s)
    fenced = _strip_markdown_code_fence(s)
    add(fenced)
    obj1 = _extract_first_json_object(s)
    add(obj1)
    obj2 = _extract_first_json_object(fenced)
    add(obj2)
    return out


def _strip_markdown_code_fence(s: str) -> str:
    s = (s or "").strip()
    for fence in ("```json", "```JSON", "```Json", "```"):
        if s.startswith(fence):
            s = s[len(fence):]
            break
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _extract_first_json_object(s: str) -> str:
    """Find the first balanced {...} substring (string-aware)."""
    s = s or ""
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


def _try_parse_decision(
    json_text: str,
    mode: str = "approval",
) -> tuple[str, str, dict[str, Any] | None] | None:
    """Parse a JSON candidate into (decision, comment, suggested_args).

    Returns None if the candidate is not a valid decision object.
    Ports parseAuditAgentDecisionObject from Go.
    """
    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    raw_decision = _pick_string(
        parsed,
        "decision", "Decision", "result", "action", "verdict",
    )
    decision = _normalize_decision(raw_decision)
    if not decision:
        return None

    comment = _pick_string(
        parsed,
        "comment", "Comment", "reason", "message", "rationale",
    )

    suggested_args = None
    if mode == "review_edit":
        suggested_args = _pick_object(
            parsed,
            "suggested_args", "suggestedArgs", "editedArguments", "edited_arguments", "editedArgs",
        )

    return decision, comment.strip(), suggested_args


def _pick_string(d: dict, *keys: str) -> str:
    for k in keys:
        if k in d and d[k] is not None:
            s = str(d[k]).strip()
            if s:
                return s
    return ""


def _pick_object(d: dict, *keys: str) -> dict[str, Any] | None:
    for k in keys:
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if isinstance(v, dict):
            if v:
                return v
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s or s == "{}":
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and obj:
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
    return None


# Approve synonyms (English only — Vietnamese/Chinese comments are not used in code)
_APPROVE_SYNONYMS = {
    "approve", "approved", "pass", "passed", "allow", "allowed",
    "yes", "ok", "accept", "accepted",
}
_REJECT_SYNONYMS = {
    "reject", "rejected", "deny", "denied", "no",
    "block", "blocked", "refuse", "refused",
}


def _normalize_decision(v: str) -> str:
    s = (v or "").lower().strip()
    if s in _APPROVE_SYNONYMS:
        return "approve"
    if s in _REJECT_SYNONYMS:
        return "reject"
    return ""
