"""
Tests for VAPT-AI HITLManager (W7-B) — 3-mode branching + decision flow.

Covers:
    1. Mode resolution (audit_agent default / human_block / auto_approve / invalid → default)
    2. is_hitl_required() classification (DESTRUCTIVE_TOOLS, SAFE_TOOLS, C2 L3+, sqlmap forbidden args)
    3. request_and_wait() in auto_approve mode → returns approve immediately
    4. request_and_wait() in audit_agent mode → calls AuditAgent.review() + applies decision
    5. request_and_wait() in human_block mode → returns approval_timeout (NotImplementedError caught)
    6. _apply_decision() — updates HITLApproval row correctly for each decision type
    7. SSE events emitted: hitl_approval_required + hitl_decision_made
    8. Legacy request_approval() still works (backward compat with W6-D)
    9. HITLDecision dataclass + to_dict()
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hitl.manager import (
    HITLManager,
    HITLDecision,
    HITL_TIMEOUT_SECONDS,
    DESTRUCTIVE_TOOLS,
    SAFE_TOOLS,
    HITL_REQUIRED_C2_LEVELS,
    VALID_MODES,
    DEFAULT_MODE,
    _resolve_mode,
)
from app.hitl.audit_agent import AuditDecision


def _make_fake_session():
    """Build a fake AsyncSession that auto-assigns UUID id on flush.

    The real SQLAlchemy flush() populates server-side gen_random_uuid() into
    the id column. Our mock needs to mimic this so tests can assert on
    approval_id after request_and_wait().
    """
    fake_session = MagicMock()
    fake_session.add = MagicMock()
    fake_session.flush = AsyncMock()

    async def _flush_side_effect(*args, **kwargs):
        # Set id on any pending HITLApproval objects added via session.add()
        for call_obj in fake_session.add.call_args_list:
            obj = call_obj.args[0] if call_obj.args else call_obj.kwargs.get("instance")
            if obj is not None and getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    fake_session.flush.side_effect = _flush_side_effect
    return fake_session


# ============================================================

class FakeAuditAgent:
    """Test double for AuditAgent — returns a preset AuditDecision."""

    def __init__(self, decision: AuditDecision):
        self._decision = decision
        self.review_calls: list[dict[str, Any]] = []

    async def review(self, **kwargs: Any) -> AuditDecision:
        self.review_calls.append(kwargs)
        return self._decision


def _make_audit_decision(
    decision: str = "approve",
    comment: str = "low risk",
    suggested_args: dict | None = None,
    decided_by: str = "audit_agent",
) -> AuditDecision:
    return AuditDecision(
        decision=decision,
        comment=comment,
        suggested_args=suggested_args,
        decided_by=decided_by,
        duration_seconds=0.5,
    )


# ============================================================
# Mode resolution
# ============================================================

def test_resolve_mode_default():
    """No explicit + no env → DEFAULT_MODE (audit_agent)."""
    assert _resolve_mode(None) == "audit_agent"
    assert DEFAULT_MODE == "audit_agent"


def test_resolve_mode_explicit_overrides_env():
    """Explicit param takes priority over env."""
    assert _resolve_mode("auto_approve") == "auto_approve"
    assert _resolve_mode("human_block") == "human_block"
    assert _resolve_mode("audit_agent") == "audit_agent"


def test_resolve_mode_invalid_falls_back_to_default():
    """Invalid explicit value falls back to default."""
    assert _resolve_mode("invalid_mode") == "audit_agent"
    assert _resolve_mode("") == "audit_agent"


def test_valid_modes_constant():
    assert VALID_MODES == {"audit_agent", "human_block", "auto_approve"}


# ============================================================
# is_hitl_required
# ============================================================

def test_is_hitl_required_destructive_tool():
    mgr = HITLManager.__new__(HITLManager)  # skip __init__ (no session needed)
    mgr.session = None
    mgr.mode = "audit_agent"
    mgr._audit_agent = None
    for tool in DESTRUCTIVE_TOOLS:
        assert mgr.is_hitl_required(tool) is True, f"{tool} should require HITL"


def test_is_hitl_required_safe_tool():
    mgr = HITLManager.__new__(HITLManager)
    mgr.session = None
    mgr.mode = "audit_agent"
    mgr._audit_agent = None
    for tool in SAFE_TOOLS:
        assert mgr.is_hitl_required(tool) is False, f"{tool} should NOT require HITL"


def test_is_hitl_required_c2_level():
    mgr = HITLManager.__new__(HITLManager)
    mgr.session = None
    mgr.mode = "audit_agent"
    mgr._audit_agent = None
    for level in HITL_REQUIRED_C2_LEVELS:
        assert mgr.is_hitl_required("c2_task", c2_task_level=level) is True
    assert mgr.is_hitl_required("c2_task", c2_task_level=1) is False
    assert mgr.is_hitl_required("c2_task", c2_task_level=2) is False


def test_is_hitl_required_sqlmap_forbidden_args():
    mgr = HITLManager.__new__(HITLManager)
    mgr.session = None
    mgr.mode = "audit_agent"
    mgr._audit_agent = None
    for forbidden in ["--os-shell", "--dump", "--sql-shell", "--os-pwn", "--priv-esc"]:
        assert mgr.is_hitl_required("sqlmap", args={"extra": forbidden}) is True
    # Plain sqlmap (detect mode) — no HITL
    assert mgr.is_hitl_required("sqlmap", args={"url": "http://x"}) is False


# ============================================================
# HITLDecision dataclass
# ============================================================

def test_hitl_decision_to_dict():
    d = HITLDecision(
        decision="approve",
        comment="ok",
        suggested_args={"a": 1},
        risk_assessment="low",
        confidence=0.9,
        decided_by="audit_agent",
        duration_seconds=1.5,
        approval_id=uuid.uuid4(),
    )
    out = d.to_dict()
    assert out["decision"] == "approve"
    assert out["comment"] == "ok"
    assert out["suggested_args"] == {"a": 1}
    assert out["decided_by"] == "audit_agent"
    assert out["approval_id"]  # str of uuid
    assert "raw_llm_response" not in out  # debug-only field excluded


def test_hitl_decision_minimal():
    d = HITLDecision(decision="reject", comment="no")
    assert d.suggested_args is None
    assert d.risk_assessment is None
    assert d.confidence == 0.0
    assert d.decided_by == "audit_agent"
    assert d.approval_id is None
    assert d.raw_llm_response is None


# ============================================================
# request_and_wait — auto_approve mode
# ============================================================

@pytest.mark.asyncio
async def test_request_and_wait_auto_approve_mode():
    """In auto_approve mode, returns approve immediately without LLM call."""
    fake_session = _make_fake_session()

    mgr = HITLManager(session=fake_session, mode="auto_approve")
    decision = await mgr.request_and_wait(
        scan_id="scan_test_1",
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit"},
        predicted_impact="destructive",
        agent_reasoning="test",
    )
    assert decision.decision == "approve"
    assert decision.decided_by == "auto"
    assert "auto-approved" in decision.comment
    assert decision.approval_id is not None  # row was created
    # Session.add called once (for the HITLApproval row)
    assert fake_session.add.call_count == 1
    # flush called twice (after create, after apply)
    assert fake_session.flush.call_count >= 2


# ============================================================
# request_and_wait — audit_agent mode
# ============================================================

@pytest.mark.asyncio
async def test_request_and_wait_audit_agent_approve():
    """audit_agent mode with approve decision → HITLDecision.decision == 'approve'."""
    fake_session = _make_fake_session()

    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve", comment="low risk"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    decision = await mgr.request_and_wait(
        scan_id="scan_test_2",
        tool_name="metasploit",
        target="10.10.10.5",
        args={"module_type": "exploit"},
        predicted_impact="destructive",
        agent_reasoning="Exploit MS17-010",
        scan_context={"declared_scope": ["10.10.10.0/24"]},
    )
    assert decision.decision == "approve"
    assert decision.decided_by == "audit_agent"
    assert "low risk" in decision.comment
    assert decision.approval_id is not None
    # Audit agent was called once with expected args
    assert len(fake_audit.review_calls) == 1
    call = fake_audit.review_calls[0]
    assert call["tool_name"] == "metasploit"
    assert call["target"] == "10.10.10.5"
    assert call["predicted_impact"] == "destructive"
    assert call["scan_context"]["declared_scope"] == ["10.10.10.0/24"]


@pytest.mark.asyncio
async def test_request_and_wait_audit_agent_reject():
    """audit_agent mode with reject decision → HITLDecision.decision == 'reject'."""
    fake_session = _make_fake_session()

    fake_audit = FakeAuditAgent(_make_audit_decision(decision="reject", comment="DROP TABLE detected"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    decision = await mgr.request_and_wait(
        scan_id="scan_test_3",
        tool_name="custom_rce",
        target="10.10.10.5",
        args={"command": "DROP TABLE users;"},
        predicted_impact="data_modification",
        agent_reasoning="Cleanup",
    )
    assert decision.decision == "reject"
    assert "DROP TABLE" in decision.comment or "audit agent" in decision.comment.lower()


@pytest.mark.asyncio
async def test_request_and_wait_audit_agent_suggest_alternative():
    """audit_agent mode with suggest_alternative → suggested_args propagated."""
    fake_session = _make_fake_session()

    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="suggest_alternative",
        comment="narrowing scope",
        suggested_args={"level": 1, "risk": 1},
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    decision = await mgr.request_and_wait(
        scan_id="scan_test_4",
        tool_name="sqlmap",
        target="http://x",
        args={"url": "http://x/?id=1", "level": 5, "risk": 3},
        predicted_impact="active_probe",
        agent_reasoning="test",
    )
    assert decision.decision == "suggest_alternative"
    assert decision.suggested_args == {"level": 1, "risk": 1}


@pytest.mark.asyncio
async def test_request_and_wait_audit_agent_fallback_on_exception():
    """If AuditAgent raises unexpected exception → fallback to reject."""
    fake_session = _make_fake_session()

    class ExplodingAuditAgent:
        async def review(self, **kwargs):
            raise RuntimeError("LLM service is down")

    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=ExplodingAuditAgent())

    decision = await mgr.request_and_wait(
        scan_id="scan_test_5",
        tool_name="metasploit",
        target="10.10.10.5",
        args={},
        predicted_impact="destructive",
        agent_reasoning="test",
    )
    assert decision.decision == "reject"
    assert decision.decided_by == "audit_agent_fallback"
    assert "RuntimeError" in decision.comment or "LLM service is down" in decision.comment


# ============================================================
# request_and_wait — human_block mode (deferred)
# ============================================================

@pytest.mark.asyncio
async def test_request_and_wait_human_block_returns_timeout():
    """human_block mode raises NotImplementedError → caught, returns approval_timeout."""
    fake_session = _make_fake_session()

    mgr = HITLManager(session=fake_session, mode="human_block")
    decision = await mgr.request_and_wait(
        scan_id="scan_test_6",
        tool_name="metasploit",
        target="10.10.10.5",
        args={},
        predicted_impact="destructive",
        agent_reasoning="test",
    )
    assert decision.decision == "approval_timeout"
    assert decision.decided_by == "timeout"
    assert "not implemented" in decision.comment.lower()


# ============================================================
# _apply_decision
# ============================================================

@pytest.mark.asyncio
async def test_apply_decision_approve_sets_status_approved():
    """_apply_decision with approve → approval.status == 'approved'."""
    from app.db.models.hitl import HITLApproval
    approval = HITLApproval(
        scan_id="scan_x",
        tool_name="metasploit",
        target="10.10.10.5",
        args_json={},
        predicted_impact="destructive",
        agent_reasoning="test",
        kg_confidence=0.5,
        status="pending",
        expires_at=datetime.now(UTC),
    )
    approval.id = uuid.uuid4()

    mgr = HITLManager.__new__(HITLManager)
    mgr.session = MagicMock()
    mgr.session.flush = AsyncMock()

    decision = HITLDecision(decision="approve", comment="ok", approval_id=approval.id)
    await mgr._apply_decision(approval, decision)

    assert approval.status == "approved"
    assert approval.user_decision == "approve"
    assert approval.decided_at is not None
    assert approval.edited_args_json is None


@pytest.mark.asyncio
async def test_apply_decision_reject_sets_status_aborted():
    """_apply_decision with reject → approval.status == 'aborted'."""
    from app.db.models.hitl import HITLApproval
    approval = HITLApproval(
        scan_id="scan_x",
        tool_name="metasploit",
        target="10.10.10.5",
        args_json={},
        predicted_impact="destructive",
        agent_reasoning="test",
        kg_confidence=0.5,
        status="pending",
        expires_at=datetime.now(UTC),
    )
    approval.id = uuid.uuid4()

    mgr = HITLManager.__new__(HITLManager)
    mgr.session = MagicMock()
    mgr.session.flush = AsyncMock()

    decision = HITLDecision(decision="reject", comment="no", approval_id=approval.id)
    await mgr._apply_decision(approval, decision)

    assert approval.status == "aborted"
    assert approval.user_decision == "abort"


@pytest.mark.asyncio
async def test_apply_decision_suggest_alternative_sets_edited_args():
    """_apply_decision with suggest_alternative → edited_args_json populated."""
    from app.db.models.hitl import HITLApproval
    approval = HITLApproval(
        scan_id="scan_x",
        tool_name="sqlmap",
        target="http://x",
        args_json={"level": 5},
        predicted_impact="active_probe",
        agent_reasoning="test",
        kg_confidence=0.5,
        status="pending",
        expires_at=datetime.now(UTC),
    )
    approval.id = uuid.uuid4()

    mgr = HITLManager.__new__(HITLManager)
    mgr.session = MagicMock()
    mgr.session.flush = AsyncMock()

    decision = HITLDecision(
        decision="suggest_alternative",
        comment="narrowing",
        suggested_args={"level": 1},
        approval_id=approval.id,
    )
    await mgr._apply_decision(approval, decision)

    assert approval.status == "approved"
    assert approval.user_decision == "approve_with_edit"
    assert approval.edited_args_json == {"level": 1}


@pytest.mark.asyncio
async def test_apply_decision_timeout_sets_status_timeout():
    """_apply_decision with approval_timeout → approval.status == 'timeout'."""
    from app.db.models.hitl import HITLApproval
    approval = HITLApproval(
        scan_id="scan_x",
        tool_name="metasploit",
        target="10.10.10.5",
        args_json={},
        predicted_impact="destructive",
        agent_reasoning="test",
        kg_confidence=0.5,
        status="pending",
        expires_at=datetime.now(UTC),
    )
    approval.id = uuid.uuid4()

    mgr = HITLManager.__new__(HITLManager)
    mgr.session = MagicMock()
    mgr.session.flush = AsyncMock()

    decision = HITLDecision(decision="approval_timeout", comment="5 min", approval_id=approval.id)
    await mgr._apply_decision(approval, decision)

    assert approval.status == "timeout"
    assert approval.user_decision == "timeout"


# ============================================================
# SSE events emitted
# ============================================================

@pytest.mark.asyncio
async def test_sse_events_emitted_in_audit_agent_mode():
    """Both hitl_approval_required + hitl_decision_made events are published."""
    fake_session = _make_fake_session()

    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    published_events: list[dict[str, Any]] = []

    async def fake_publish_required(**kwargs):
        published_events.append({"event": "hitl_approval_required", **kwargs})

    async def fake_publish_decision(**kwargs):
        published_events.append({"event": "hitl_decision_made", **kwargs})

    with patch("app.hitl.manager.emit_hitl_required", new=fake_publish_required), \
         patch("app.hitl.manager.emit_hitl_decision", new=fake_publish_decision):
        await mgr.request_and_wait(
            scan_id="scan_sse_1",
            tool_name="metasploit",
            target="10.10.10.5",
            args={},
            predicted_impact="destructive",
            agent_reasoning="test",
        )

    # Two events should be published: hitl_approval_required + hitl_decision_made
    assert len(published_events) == 2
    assert published_events[0]["event"] == "hitl_approval_required"
    assert published_events[0]["tool_name"] == "metasploit"
    assert published_events[1]["event"] == "hitl_decision_made"
    assert published_events[1]["decision"] == "approve"
    assert published_events[1]["decided_by"] == "audit_agent"


@pytest.mark.asyncio
async def test_sse_decision_event_includes_suggested_args():
    """hitl_decision_made event includes suggested_args when present."""
    fake_session = _make_fake_session()

    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="suggest_alternative",
        suggested_args={"level": 1},
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    captured_decision_event: dict[str, Any] = {}

    async def fake_publish_required(**kwargs):
        pass

    async def fake_publish_decision(**kwargs):
        captured_decision_event.update(kwargs)

    with patch("app.hitl.manager.emit_hitl_required", new=fake_publish_required), \
         patch("app.hitl.manager.emit_hitl_decision", new=fake_publish_decision):
        await mgr.request_and_wait(
            scan_id="scan_sse_2",
            tool_name="sqlmap",
            target="http://x",
            args={},
            predicted_impact="active_probe",
            agent_reasoning="test",
        )

    assert captured_decision_event["decision"] == "suggest_alternative"
    assert captured_decision_event["suggested_args"] == {"level": 1}


# ============================================================
# Legacy: request_approval (W3-C compat)
# ============================================================

@pytest.mark.asyncio
async def test_legacy_request_approval_still_works():
    """W6-D code calls request_approval() — must still work after W7-B refactor."""
    fake_session = _make_fake_session()

    mgr = HITLManager(session=fake_session, mode="audit_agent")

    async def fake_publish(**kwargs):
        pass

    with patch("app.hitl.manager.emit_hitl_required", new=fake_publish):
        approval = await mgr.request_approval(
            scan_id="scan_legacy_1",
            tool_name="metasploit",
            target="10.10.10.5",
            args={"module_type": "exploit"},
            predicted_impact="destructive",
            agent_reasoning="legacy",
        )

    assert approval is not None
    assert approval.status == "pending"
    assert approval.scan_id == "scan_legacy_1"
    assert approval.tool_name == "metasploit"
    assert approval.expires_at is not None


# ============================================================
# Constants sanity check
# ============================================================

def test_constants():
    assert HITL_TIMEOUT_SECONDS == 300
    assert "metasploit" in DESTRUCTIVE_TOOLS
    assert "sqlmap_exploit" in DESTRUCTIVE_TOOLS
    assert "custom_rce" in DESTRUCTIVE_TOOLS
    assert "payload_gen" in DESTRUCTIVE_TOOLS
    assert "webshell_deploy" in DESTRUCTIVE_TOOLS
    assert "nmap" in SAFE_TOOLS
    assert "nuclei" in SAFE_TOOLS
    assert HITL_REQUIRED_C2_LEVELS == {3, 4, 5}
