"""
Tests for VAPT-AI Audit Agent — W7-A.

Ports the scenarios covered by CyberStrikeAI's hitl_audit_agent_test.go and
adds the 5 W7-A acceptance scenarios:
    1. approve        — LLM returns {"decision":"approve", "comment":"..."} -> AuditDecision.decision == "approve"
    2. reject         — LLM returns {"decision":"reject",  "comment":"..."} -> AuditDecision.decision == "reject"
    3. suggest_alt    — review_edit mode, LLM returns approve + suggested_args -> decision == "suggest_alternative"
    4. LLM timeout    — caller raises TimeoutError -> fallback decision (reject by default)
    5. malformed JSON — LLM returns prose -> fallback decision

Plus: parser unit tests (json_candidates, extract_first_json_object, strip code fence).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.hitl.audit_agent import (
    AuditAgent,
    AuditDecision,
    AuditReviewInput,
    _extract_first_json_object,
    _json_candidates,
    _normalize_decision,
    _normalize_fallback,
    _normalize_mode,
    _strip_markdown_code_fence,
    _try_parse_decision,
)


# ---------- Test fixtures ----------

def _make_review_input(**overrides: Any) -> AuditReviewInput:
    base = dict(
        hitl_mode="approval",
        tool_name="metasploit",
        target="10.10.10.5",
        args={
            "module_type": "exploit",
            "module_name": "exploit/windows/smb/ms17_010_eternalblue",
        },
        predicted_impact="destructive",
        agent_reasoning="Exploit MS17-010 to obtain a Meterpreter session",
        scan_context={
            "declared_scope": ["10.10.10.0/24"],
            "destructive_ops_count": 1,
            "max_destructive_ops": 5,
            "blackboard_facts": [{"type": "asset", "key": "host", "value": "10.10.10.5"}],
            "kg_paths": [{"path": "smb -> ms17-010 -> meterpreter", "prob": 0.7}],
        },
    )
    base.update(overrides)
    return AuditReviewInput(**base)


def _make_sync_caller(response_text: str):
    """Build a sync LLM caller that always returns the given text."""
    def _caller(**kwargs: Any) -> str:
        return response_text
    return _caller


def _make_async_caller(response_text: str):
    """Build an async LLM caller that always returns the given text."""
    async def _caller(**kwargs: Any) -> str:
        return response_text
    return _caller


def _make_failing_caller(exc: Exception):
    """Build a sync caller that always raises."""
    def _caller(**kwargs: Any) -> str:
        raise exc
    return _caller


def _make_async_failing_caller(exc: Exception):
    """Build an async caller that always raises."""
    async def _caller(**kwargs: Any) -> str:
        raise exc
    return _caller


# ============================================================
# Scenario 1: approve
# ============================================================

@pytest.mark.asyncio
async def test_review_approve_sync_caller():
    """LLM returns approve -> AuditDecision.decision == 'approve'."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller('{"decision":"approve","comment":"low-risk recon, target in scope"}'),
    )
    decision = await agent.review(
        tool_name="sqlmap",
        target="http://target.example.com",
        args={"url": "http://target.example.com/?id=1", "level": 1},
        predicted_impact="active_probe",
        agent_reasoning="Test for SQLi on id parameter",
    )
    assert decision.decision == "approve"
    assert "audit agent" in decision.comment.lower()
    assert decision.decided_by == "audit_agent"
    assert decision.suggested_args is None
    assert decision.duration_seconds >= 0


@pytest.mark.asyncio
async def test_review_approve_async_caller():
    """Same as above but caller is async (matches chat_completion shape)."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_async_caller('{"decision":"approve","comment":"ok"}'),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={"target": "10.10.10.5"},
        predicted_impact="read_only",
        agent_reasoning="Port scan",
    )
    assert decision.decision == "approve"


# ============================================================
# Scenario 2: reject
# ============================================================

@pytest.mark.asyncio
async def test_review_reject():
    """LLM returns reject -> AuditDecision.decision == 'reject'."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller(
            '{"decision":"reject","comment":"DROP TABLE detected — irreversible data loss"}'
        ),
    )
    decision = await agent.review(
        tool_name="custom_rce",
        target="10.10.10.5",
        args={"command": "psql -c 'DROP TABLE users;'"},
        predicted_impact="data_modification",
        agent_reasoning="Cleanup DB before exit",
    )
    assert decision.decision == "reject"
    assert "DROP TABLE" in decision.comment or "audit agent" in decision.comment.lower()


@pytest.mark.asyncio
async def test_review_reject_synonym():
    """LLM uses 'deny' instead of 'reject' — should normalize."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller('{"decision":"deny","comment":"no"}'),
    )
    decision = await agent.review(
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit"},
        predicted_impact="destructive",
        agent_reasoning="x",
    )
    assert decision.decision == "reject"


# ============================================================
# Scenario 3: suggest_alternative (review_edit mode)
# ============================================================

@pytest.mark.asyncio
async def test_review_suggest_alternative():
    """In review_edit mode, approve + suggested_args -> decision == 'suggest_alternative'."""
    raw = json.dumps({
        "decision": "approve",
        "comment": "narrowing sqlmap level to 1 to reduce noise",
        "suggested_args": {
            "url": "http://target.example.com/?id=1",
            "level": 1,
            "risk": 1,
        },
    })
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="review_edit",
        llm_caller=_make_sync_caller(raw),
    )
    decision = await agent.review(
        tool_name="sqlmap",
        target="http://target.example.com",
        args={"url": "http://target.example.com/?id=1", "level": 5, "risk": 3},
        predicted_impact="active_probe",
        agent_reasoning="Test for SQLi",
    )
    assert decision.decision == "suggest_alternative"
    assert decision.suggested_args is not None
    assert decision.suggested_args["level"] == 1
    assert decision.suggested_args["risk"] == 1


@pytest.mark.asyncio
async def test_review_edit_mode_approve_without_args_stays_approve():
    """In review_edit mode, approve WITHOUT suggested_args stays 'approve'."""
    raw = json.dumps({"decision": "approve", "comment": "args already minimal"})
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="review_edit",
        llm_caller=_make_sync_caller(raw),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={"target": "10.10.10.5"},
        predicted_impact="read_only",
        agent_reasoning="port scan",
    )
    assert decision.decision == "approve"
    assert decision.suggested_args is None


@pytest.mark.asyncio
async def test_approval_mode_ignores_suggested_args():
    """In approval mode, suggested_args in response is ignored (no edit)."""
    raw = json.dumps({
        "decision": "approve",
        "comment": "ok",
        "suggested_args": {"level": 1},  # should be ignored in approval mode
    })
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller(raw),
    )
    decision = await agent.review(
        tool_name="sqlmap",
        target="http://target.example.com",
        args={"url": "x", "level": 5},
        predicted_impact="active_probe",
        agent_reasoning="x",
    )
    assert decision.decision == "approve"  # NOT suggest_alternative
    assert decision.suggested_args is None


# ============================================================
# Scenario 4: LLM call fails (timeout / API error)
# ============================================================

@pytest.mark.asyncio
async def test_review_llm_timeout_fallback_reject():
    """LLM caller raises TimeoutError -> fallback to reject (safe default)."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        fallback_decision="reject",
        llm_caller=_make_async_failing_caller(TimeoutError("LLM timed out after 90s")),
    )
    decision = await agent.review(
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit"},
        predicted_impact="destructive",
        agent_reasoning="x",
    )
    assert decision.decision == "reject"
    assert "fallback" in decision.comment.lower() or "timed out" in decision.comment.lower()
    assert decision.decided_by == "audit_agent_fallback"


@pytest.mark.asyncio
async def test_review_llm_timeout_fallback_approve():
    """Same as above but fallback=approve (debug mode)."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        fallback_decision="approve",
        llm_caller=_make_async_failing_caller(RuntimeError("API 500")),
    )
    decision = await agent.review(
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit"},
        predicted_impact="destructive",
        agent_reasoning="x",
    )
    assert decision.decision == "approve"
    assert decision.decided_by == "audit_agent_fallback"


# ============================================================
# Scenario 5: malformed JSON
# ============================================================

@pytest.mark.asyncio
async def test_review_malformed_json_fallback():
    """LLM returns prose instead of JSON -> fallback to reject."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        fallback_decision="reject",
        llm_caller=_make_sync_caller("I think this is safe to run."),
    )
    decision = await agent.review(
        tool_name="sqlmap",
        target="http://target.example.com",
        args={"url": "x"},
        predicted_impact="active_probe",
        agent_reasoning="x",
    )
    assert decision.decision == "reject"
    assert decision.decided_by == "audit_agent_fallback"


@pytest.mark.asyncio
async def test_review_empty_response_fallback():
    """LLM returns empty string -> fallback to reject."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        fallback_decision="reject",
        llm_caller=_make_sync_caller(""),
    )
    decision = await agent.review(
        tool_name="sqlmap",
        target="http://x",
        args={},
        predicted_impact="active_probe",
        agent_reasoning="x",
    )
    assert decision.decision == "reject"
    assert "empty" in decision.comment.lower()


@pytest.mark.asyncio
async def test_review_markdown_code_fence_stripped():
    """LLM wraps JSON in ```json ... ``` — should still parse correctly."""
    raw = '```json\n{"decision":"approve","comment":"ok"}\n```'
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller(raw),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={"target": "10.10.10.5"},
        predicted_impact="read_only",
        agent_reasoning="port scan",
    )
    assert decision.decision == "approve"


@pytest.mark.asyncio
async def test_review_json_embedded_in_prose():
    """LLM returns prose + JSON — should extract the first JSON object."""
    raw = 'Looking at this request, my assessment is:\n{"decision":"approve","comment":"ok"}\nThat is my final answer.'
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller(raw),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={},
        predicted_impact="read_only",
        agent_reasoning="x",
    )
    assert decision.decision == "approve"


# ============================================================
# Comment normalization
# ============================================================

@pytest.mark.asyncio
async def test_comment_gets_audit_agent_prefix():
    """Comment without 'audit agent' prefix gets it prepended."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller('{"decision":"approve","comment":"low risk"}'),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={},
        predicted_impact="read_only",
        agent_reasoning="x",
    )
    assert decision.comment.lower().startswith("audit agent")


@pytest.mark.asyncio
async def test_empty_comment_gets_default():
    """Empty comment -> 'audit agent: <decision>'."""
    agent = AuditAgent(
        model="gpt-4o-mini",
        mode="approval",
        llm_caller=_make_sync_caller('{"decision":"approve"}'),
    )
    decision = await agent.review(
        tool_name="nmap",
        target="10.10.10.5",
        args={},
        predicted_impact="read_only",
        agent_reasoning="x",
    )
    assert decision.comment.lower().startswith("audit agent")
    assert "approve" in decision.comment.lower()


# ============================================================
# AuditReviewInput.to_user_message
# ============================================================

def test_review_input_renders_json():
    """to_user_message returns valid JSON with expected keys."""
    inp = _make_review_input()
    msg = inp.to_user_message()
    parsed = json.loads(msg)
    assert parsed["hitlMode"] == "approval"
    assert parsed["toolName"] == "metasploit"
    assert parsed["target"] == "10.10.10.5"
    assert "args" in parsed
    assert "predictedImpact" in parsed
    assert "agentReasoning" in parsed
    assert "scanContext" in parsed
    assert parsed["scanContext"]["destructiveOpsCount"] == 1
    assert parsed["scanContext"]["maxDestructiveOps"] == 5
    assert len(parsed["scanContext"]["blackboardFacts"]) == 1


def test_review_input_truncates_long_strings():
    """Long agent_reasoning should be truncated."""
    inp = _make_review_input(agent_reasoning="x" * 5000)
    msg = inp.to_user_message()
    parsed = json.loads(msg)
    assert len(parsed["agentReasoning"]) < 5000
    assert "truncated" in parsed["agentReasoning"].lower()


def test_review_input_caps_blackboard_facts():
    """More than 5 blackboard facts -> only first 5 sent."""
    facts = [{"type": "asset", "key": f"h{i}", "value": str(i)} for i in range(20)]
    inp = _make_review_input(scan_context={"blackboard_facts": facts})
    msg = inp.to_user_message()
    parsed = json.loads(msg)
    assert len(parsed["scanContext"]["blackboardFacts"]) == 5


# ============================================================
# Parser unit tests (port from Go auditAgentJSONCandidates tests)
# ============================================================

def test_strip_markdown_code_fence_json():
    s = '```json\n{"a":1}\n```'
    assert _strip_markdown_code_fence(s) == '{"a":1}'


def test_strip_markdown_code_fence_plain():
    s = '```\n{"a":1}\n```'
    assert _strip_markdown_code_fence(s) == '{"a":1}'


def test_strip_markdown_code_fence_none():
    s = '{"a":1}'
    assert _strip_markdown_code_fence(s) == '{"a":1}'


def test_extract_first_json_object_simple():
    s = 'prefix {"a":1} suffix'
    assert _extract_first_json_object(s) == '{"a":1}'


def test_extract_first_json_object_nested():
    s = 'text {"a": {"b": 2}, "c": 3} trailing'
    assert _extract_first_json_object(s) == '{"a": {"b": 2}, "c": 3}'


def test_extract_first_json_object_with_braces_in_string():
    """Braces inside string literals must not affect depth counting."""
    s = '{"a": "this { is } a brace"}'
    assert _extract_first_json_object(s) == s


def test_extract_first_json_object_no_object():
    assert _extract_first_json_object("no json here") == ""


def test_json_candidates_dedup():
    """Same candidate should not appear twice."""
    s = '{"a":1}'
    cands = _json_candidates(s)
    # Should produce unique candidates
    assert len(cands) == len(set(cands))
    assert '{"a":1}' in cands


def test_json_candidates_fenced():
    """Fenced JSON should produce both fenced and unfenced candidates."""
    s = '```json\n{"a":1}\n```'
    cands = _json_candidates(s)
    assert len(cands) >= 1
    assert any('{"a":1}' in c for c in cands)


def test_try_parse_decision_approve():
    res = _try_parse_decision('{"decision":"approve","comment":"ok"}')
    assert res is not None
    assert res[0] == "approve"
    assert res[1] == "ok"
    assert res[2] is None


def test_try_parse_decision_reject():
    res = _try_parse_decision('{"decision":"reject"}')
    assert res is not None
    assert res[0] == "reject"


def test_try_parse_decision_synonyms():
    """All approve synonyms should normalize correctly."""
    for syn in ("approve", "approved", "pass", "yes", "ok", "accept"):
        res = _try_parse_decision(f'{{"decision":"{syn}"}}')
        assert res is not None and res[0] == "approve", f"Failed for synonym: {syn}"
    for syn in ("reject", "deny", "no", "block", "refuse"):
        res = _try_parse_decision(f'{{"decision":"{syn}"}}')
        assert res is not None and res[0] == "reject", f"Failed for synonym: {syn}"


def test_try_parse_decision_alternate_keys():
    """LLM uses 'result' instead of 'decision' — should still parse."""
    res = _try_parse_decision('{"result":"approve","reason":"ok"}')
    assert res is not None
    assert res[0] == "approve"
    assert res[1] == "ok"


def test_try_parse_decision_invalid_json():
    assert _try_parse_decision("not json") is None


def test_try_parse_decision_missing_decision():
    assert _try_parse_decision('{"comment":"no decision field"}') is None


def test_try_parse_decision_unknown_decision():
    assert _try_parse_decision('{"decision":"maybe"}') is None


def test_try_parse_decision_review_edit_picks_suggested_args():
    res = _try_parse_decision(
        '{"decision":"approve","suggested_args":{"level":1}}',
        mode="review_edit",
    )
    assert res is not None
    assert res[0] == "approve"
    assert res[2] == {"level": 1}


def test_try_parse_decision_approval_mode_ignores_suggested_args():
    res = _try_parse_decision(
        '{"decision":"approve","suggested_args":{"level":1}}',
        mode="approval",
    )
    assert res is not None
    assert res[2] is None  # suggested_args not extracted in approval mode


# ============================================================
# Normalizers
# ============================================================

def test_normalize_mode():
    assert _normalize_mode("approval") == "approval"
    assert _normalize_mode("review_edit") == "review_edit"
    assert _normalize_mode("review-edit") == "review_edit"
    assert _normalize_mode("") == "approval"
    assert _normalize_mode("off") == "approval"
    assert _normalize_mode("anything") == "approval"


def test_normalize_fallback():
    assert _normalize_fallback("reject") == "reject"
    assert _normalize_fallback("approve") == "approve"
    assert _normalize_fallback("yes") == "approve"
    assert _normalize_fallback("true") == "approve"
    assert _normalize_fallback("1") == "approve"
    assert _normalize_fallback("") == "reject"
    assert _normalize_fallback("anything") == "reject"


def test_normalize_decision():
    assert _normalize_decision("approve") == "approve"
    assert _normalize_decision("APPROVE") == "approve"
    assert _normalize_decision("Approved") == "approve"
    assert _normalize_decision("reject") == "reject"
    assert _normalize_decision("DENY") == "reject"
    assert _normalize_decision("maybe") == ""
    assert _normalize_decision("") == ""


# ============================================================
# AuditDecision dataclass
# ============================================================

def test_audit_decision_to_dict():
    d = AuditDecision(
        decision="approve",
        comment="ok",
        suggested_args={"a": 1},
        risk_assessment="low",
        confidence=0.9,
        decided_by="audit_agent",
        duration_seconds=1.5,
    )
    out = d.to_dict()
    assert out["decision"] == "approve"
    assert out["comment"] == "ok"
    assert out["suggested_args"] == {"a": 1}
    assert out["risk_assessment"] == "low"
    assert out["confidence"] == 0.9
    assert out["decided_by"] == "audit_agent"
    assert out["duration_seconds"] == 1.5
    # raw_llm_response must NOT be in dict (debug only)
    assert "raw_llm_response" not in out