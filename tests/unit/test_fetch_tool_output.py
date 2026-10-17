"""
Unit tests — `fetch_tool_output` plus the two agent-loop fixes that came
with it:

  * `fetch_tool_output(execution_id)` actually reads back the full output
    (so the truncation notices are no longer a promise the system cannot
    keep),
  * the advisory round-robin falls through instead of burning a turn's
    advisory slot,
  * the premature-exit gate is SOFT (challenge once, then comply).
"""
from __future__ import annotations

import json

import pytest

from app.agents import tool_artifacts as art
from app.agents.tool_bridge import _fetch_tool_output, execute_tool_call
from app.agents.react_agent import (
    EXIT_SOFT_GATE_MIN_COMMANDS,
    EXIT_SOFT_GATE_MIN_TURNS,
    KEEP_RECENT_RESULTS,
    _exit_needs_challenge,
    _prune_old_tool_results,
    _select_advisory,
)

SCAN = "scan_fetch_test"


@pytest.fixture(autouse=True)
def _clean_store():
    art.clear_scan_artifacts(SCAN)
    yield
    art.clear_scan_artifacts(SCAN)


# ── fetch_tool_output handler ─────────────────────────────────────────


class TestFetchToolOutput:
    def _seed(self, body: str = "0123456789" * 1000) -> str:
        art.record_artifact(
            SCAN, "exec-abc", "sqlmap", body,
            spill_path=None, command="sqlmap -u http://t/?id=1 --batch",
        )
        return body

    def test_returns_full_content_by_execution_id(self):
        body = self._seed()
        payload = json.loads(_fetch_tool_output({"execution_id": "exec-abc"}, SCAN))
        assert payload["status"] == "ok"
        assert payload["tool"] == "sqlmap"
        assert payload["total_chars"] == len(body)
        assert payload["content"] == body[:4000]
        assert payload["has_more"] is True
        assert payload["next_offset"] == 4000

    def test_paging_with_offset(self):
        body = self._seed()
        payload = json.loads(
            _fetch_tool_output({"execution_id": "exec-abc", "offset": 4000}, SCAN)
        )
        assert payload["content"] == body[4000:8000]
        assert payload["offset"] == 4000

    def test_limit_is_capped(self):
        self._seed("x" * 50_000)
        payload = json.loads(
            _fetch_tool_output({"execution_id": "exec-abc", "limit": 10**9}, SCAN)
        )
        assert payload["returned_chars"] == 20_000

    def test_tool_name_fallback(self):
        self._seed()
        payload = json.loads(_fetch_tool_output({"tool_name": "sqlmap"}, SCAN))
        assert payload["status"] == "ok"
        assert payload["execution_id"] == "exec-abc"

    def test_missing_arguments_is_actionable(self):
        payload = json.loads(_fetch_tool_output({}, SCAN))
        assert payload["status"] == "error"
        assert "execution_id" in payload["message"]

    def test_unknown_execution_id(self):
        payload = json.loads(_fetch_tool_output({"execution_id": "ghost"}, SCAN))
        assert payload["status"] == "not_found"

    def test_offset_past_end(self):
        self._seed("short")
        payload = json.loads(
            _fetch_tool_output({"execution_id": "exec-abc", "offset": 999}, SCAN)
        )
        assert payload["status"] == "offset_out_of_range"
        assert payload["total_chars"] == 5

    def test_empty_body(self):
        art.record_artifact(SCAN, "exec-empty", "httpx", "")
        payload = json.loads(_fetch_tool_output({"execution_id": "exec-empty"}, SCAN))
        assert payload["status"] == "empty"


class TestFetchToolOutputWiring:
    @pytest.mark.asyncio
    async def test_dispatched_through_execute_tool_call(self):
        """The tool must be reachable from the agent's only entry point."""
        art.record_artifact(SCAN, "exec-wired", "nuclei", "MATCH: sqli found")
        out = await execute_tool_call(
            tool_name="fetch_tool_output",
            tool_args={"execution_id": "exec-wired"},
            target="http://example.test",
            scan_id=SCAN,
            executor=None,  # must not be touched — no subprocess involved
        )
        payload = json.loads(out)
        assert payload["status"] == "ok"
        assert payload["content"] == "MATCH: sqli found"


# ── prune note points at the retrieval tool ───────────────────────────


class TestPruneNoteReferencesFetchTool:
    def test_note_carries_execution_id(self):
        messages = [
            {"role": "tool", "content": "x" * 5_000 + "\n[execution_id] exec-42"}
            for _ in range(KEEP_RECENT_RESULTS + 3)
        ]
        pruned = _prune_old_tool_results(messages)
        assert pruned == 3
        note = messages[0]["content"]
        assert "fetch_tool_output" in note
        assert "exec-42" in note

    def test_note_without_id_is_still_honest(self):
        messages = [
            {"role": "tool", "content": "y" * 5_000}
            for _ in range(KEEP_RECENT_RESULTS + 1)
        ]
        _prune_old_tool_results(messages)
        note = messages[0]["content"]
        assert "fetch_tool_output(execution_id=" not in note
        assert "scan artifacts" in note


# ── advisory rotation fall-through ────────────────────────────────────


RL = {"role": "system", "content": "rl"}
TDA = {"role": "system", "content": "tda"}
EGATS = {"role": "system", "content": "egats"}


class TestAdvisoryRotation:
    def test_slot_order(self):
        assert _select_advisory(1, rl_msg=RL, tda_msg=TDA, egats_msg=EGATS)[0] == "rl"
        assert _select_advisory(2, rl_msg=RL, tda_msg=TDA, egats_msg=EGATS)[0] == "tda"
        assert _select_advisory(3, rl_msg=RL, tda_msg=TDA, egats_msg=EGATS)[0] == "egats"
        assert _select_advisory(4, rl_msg=RL, tda_msg=TDA, egats_msg=EGATS)[0] == "rl"

    def test_falls_through_when_slot_advisor_is_silent(self):
        """EGATS has not been seeded yet (egats_select() -> None)."""
        picked, msg = _select_advisory(3, rl_msg=RL, tda_msg=TDA, egats_msg=None)
        assert picked == "rl"
        assert msg is RL

    def test_falls_through_multiple_slots(self):
        picked, msg = _select_advisory(2, rl_msg=None, tda_msg=None, egats_msg=EGATS)
        assert picked == "egats"
        assert msg is EGATS

    def test_nothing_available(self):
        assert _select_advisory(1, rl_msg=None, tda_msg=None, egats_msg=None) == (None, None)


# ── premature-exit soft gate ──────────────────────────────────────────


class TestExitSoftGate:
    def _call(self, **kwargs):
        base = dict(
            exit_challenged=False,
            findings_count=0,
            commands_run=EXIT_SOFT_GATE_MIN_COMMANDS - 1,
            iteration=EXIT_SOFT_GATE_MIN_TURNS - 1,
        )
        base.update(kwargs)
        return _exit_needs_challenge(**base)

    def test_challenges_early_exit_with_no_work(self):
        assert self._call() is True

    def test_never_challenges_twice(self):
        """The second exit is always honoured — no infinite loop."""
        assert self._call(exit_challenged=True) is False

    def test_findings_make_an_early_exit_legitimate(self):
        assert self._call(findings_count=1) is False

    def test_enough_commands_is_accepted(self):
        assert self._call(commands_run=EXIT_SOFT_GATE_MIN_COMMANDS) is False
        assert self._call(commands_run=EXIT_SOFT_GATE_MIN_COMMANDS + 20) is False

    def test_enough_turns_is_accepted(self):
        assert self._call(iteration=EXIT_SOFT_GATE_MIN_TURNS) is False
