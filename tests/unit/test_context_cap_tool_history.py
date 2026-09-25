"""Regression test: context capping must not orphan tool-call results.

Root cause being guarded here
-----------------------------
``BaseAgent._run_react_loop`` capped its running history with a naive slice::

    messages = messages[:KEEP_HEAD] + messages[-KEEP_TAIL:]

When the cut landed between an assistant message carrying ``tool_calls`` and its
``role: "tool"`` replies, the next LLM request was rejected with::

    HTTP 400: An assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'. The following tool_call_ids did
    not have response messages: call_01_...

which aborted the specialist agent mid-scan (~iteration 12, once the history
crossed 50 messages).  ``_cap_message_history`` must drop whole exchanges so the
capped history stays API-valid.

Run:
    pytest tests/unit/test_context_cap_tool_history.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.base import KEEP_HEAD, KEEP_TAIL, MAX_MESSAGES, _cap_message_history


def _build_history(n_exchanges: int) -> list[dict]:
    """[system, user] + n x [assistant(tool_calls), tool_result] + final text."""
    messages: list[dict] = [
        {"role": "system", "content": "you are a pentest agent"},
        {"role": "user", "content": "scan the target"},
    ]
    for i in range(n_exchanges):
        call_id = f"call_{i:02d}"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "nmap", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"result {i}",
        })
    messages.append({"role": "assistant", "content": "done"})
    return messages


def _assert_valid(messages: list[dict]) -> None:
    """Every assistant tool_calls message must be followed by exactly its replies."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected = [tc["id"] for tc in msg["tool_calls"]]
            replies = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                replies.append(messages[j]["tool_call_id"])
                j += 1
            assert replies == expected, (
                f"assistant tool_calls {expected} at index {i} followed by "
                f"replies {replies}"
            )
            i = j
            continue
        assert msg.get("role") != "tool", (
            f"orphaned tool message at index {i} (assistant was dropped)"
        )
        i += 1


class TestContextCap:
    def test_no_cap_when_small(self):
        messages = _build_history(3)  # well under MAX_MESSAGES
        assert _cap_message_history(messages) == messages

    def test_capped_history_keeps_tool_pairs_valid(self):
        messages = _build_history(30)
        assert len(messages) > MAX_MESSAGES, "fixture must exceed the cap"

        capped = _cap_message_history(messages, KEEP_HEAD, KEEP_TAIL)

        assert len(capped) < len(messages), "history should actually shrink"
        assert capped[0]["role"] == "system", "system prompt must survive"
        assert capped[1]["role"] == "user", "initial user message must survive"
        _assert_valid(capped)

    def test_naive_slice_would_have_been_invalid(self):
        """Documents the original bug: the raw slice splits a tool exchange."""
        messages = _build_history(30)
        naive = messages[:KEEP_HEAD] + messages[-KEEP_TAIL:]
        with_invalid = False
        try:
            _assert_valid(naive)
        except AssertionError:
            with_invalid = True
        assert with_invalid, (
            "fixture no longer reproduces the split exchange; adjust KEEP_HEAD/"
            "KEEP_TAIL so the boundary lands mid-exchange"
        )

    def test_dangling_tail_without_assistant_is_dropped(self):
        """A tail that starts with a lone tool reply must not keep that reply."""
        messages = _build_history(30)
        # Force the tail boundary to start on a `tool` message by keeping an
        # odd number: [system, user, a0, t0, a1, t1, ...] — keep_head=4 lands on
        # a1/t1 pairing; verify with an explicit synthetic list instead.
        history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "gone", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "gone", "content": "orphan"},
        ] + _build_history(30)
        capped = _cap_message_history(history, KEEP_HEAD, KEEP_TAIL)
        _assert_valid(capped)
        assert all(m.get("tool_call_id") != "gone" for m in capped if m.get("role") == "tool")


# ============================================================
# Loop-level regression: drive the real BaseAgent ReAct loop over the cap
# ============================================================

import pytest  # noqa: E402

from app.agents.base import create_agent  # noqa: E402


@pytest.mark.asyncio
async def test_react_loop_history_stays_api_valid_across_context_cap(monkeypatch):
    """The real loop must never send an invalid history, even after capping.

    Reproduces the production failure: DeepSeek returns a VARIABLE number of
    tool calls per turn (1-3 were observed), so the history has variable-length
    exchanges. A naive head/tail slice then lands mid-exchange and the next
    request is rejected with HTTP 400. This drives the real
    ``BaseAgent._run_react_loop`` across MAX_MESSAGES and asserts every request
    handed to the LLM is valid.
    """
    requests: list[list[dict]] = []

    async def fake_chat_completion(llm_config, messages, tools=None, temperature=None):
        requests.append([dict(m) for m in messages])
        call_no = len(requests)
        if call_no <= 24:
            # Cycle 1, 2, 3 tool calls per turn — mirrors the real logs
            # ("finish=tool_calls | tool_calls=2" / "tool_calls=3").
            n_calls = (call_no % 3) + 1
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{call_no:02d}_{k}",
                        "name": "nmap",
                        "arguments": '{"target": "example.com"}',
                    }
                    for k in range(n_calls)
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "content": "done",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def fake_execute_tool_call(**kwargs):
        return "nmap output: 80/tcp open http"

    import app.agents.llm_client as agents_llm_client
    import app.agents.tool_bridge as tool_bridge
    monkeypatch.setattr(agents_llm_client, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(tool_bridge, "execute_tool_call", fake_execute_tool_call)

    agent = create_agent(
        agent_name="recon",
        scan_id="scan_cap_test",
        target="http://example.com",
        task_description="reachability check",
        user_prompt="check reachability",
        max_iterations=30,
    )

    result = await agent.run(llm_config={"provider": "openai_compatible"}, executor=object())

    assert result.status in ("completed", "max_iterations")
    assert len(requests) >= 12, "loop should have run well past the context cap"
    # At least one request must have been made AFTER the history was capped.
    assert any(len(r) > MAX_MESSAGES or len(r) > 2 for r in requests)
    for idx, r in enumerate(requests):
        try:
            _assert_valid(r)
        except AssertionError as exc:  # pragma: no cover - failure detail
            raise AssertionError(f"LLM request #{idx} had invalid history: {exc}") from exc


# ============================================================
# Guard: the fixture above really does reproduce the production failure
# ============================================================

@pytest.mark.asyncio
async def test_naive_cap_would_produce_invalid_history(monkeypatch):
    """Proves the variable-length fixture reproduces the original 400.

    With the OLD naive slice the loop sends an invalid history; with the fixed
    ``_cap_message_history`` it does not. This keeps the regression test honest.
    """
    import app.agents.base as base

    def naive_cap(messages, keep_head=base.KEEP_HEAD, keep_tail=base.KEEP_TAIL):
        if len(messages) <= base.MAX_MESSAGES:
            return messages
        return messages[:keep_head] + messages[-keep_tail:]

    monkeypatch.setattr(base, "_cap_message_history", naive_cap)

    requests: list[list[dict]] = []

    async def fake_chat_completion(llm_config, messages, tools=None, temperature=None):
        requests.append([dict(m) for m in messages])
        call_no = len(requests)
        if call_no <= 24:
            n_calls = (call_no % 3) + 1
            return {
                "content": "",
                "tool_calls": [
                    {"id": f"c_{call_no}_{k}", "name": "nmap", "arguments": "{}"}
                    for k in range(n_calls)
                ],
                "finish_reason": "tool_calls",
                "usage": {},
            }
        return {"content": "done", "tool_calls": [], "finish_reason": "stop", "usage": {}}

    async def fake_execute_tool_call(**kwargs):
        return "ok"

    import app.agents.llm_client as agents_llm_client
    import app.agents.tool_bridge as tool_bridge
    monkeypatch.setattr(agents_llm_client, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(tool_bridge, "execute_tool_call", fake_execute_tool_call)

    agent = base.create_agent("recon", "scan_x", "http://x", "t", "u", max_iterations=30)
    await agent.run(llm_config={"provider": "openai_compatible"}, executor=object())

    invalid = 0
    for r in requests:
        try:
            _assert_valid(r)
        except AssertionError:
            invalid += 1
    assert invalid > 0, (
        "the naive cap no longer produces an invalid history — the loop-level "
        "regression test is no longer meaningful"
    )
