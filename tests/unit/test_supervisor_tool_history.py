"""Regression test: supervisor tool-call history must stay API-valid.

Root cause being guarded here
-----------------------------
``SupervisorOrchestrator._supervisor_node_llm`` appended an assistant message
carrying ``tool_calls`` (the synthetic ``transfer``/``exit`` routing tools) to
its running history, but never appended the ``role: "tool"`` result messages
that OpenAI-compatible APIs (OpenAI, DeepSeek, GLM, ...) require.  The next
supervisor turn therefore failed with::

    HTTP 400: An assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'.

which aborted every scan after the first expert transfer -- i.e. the agent could
call tools but the scan could not proceed.

Run:
    pytest tests/unit/test_supervisor_tool_history.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.orchestration.langgraph_supervisor import SupervisorOrchestrator


def _transfer_response(call_id: str = "call_1"):
    return {
        "content": "I will transfer to recon.",
        "reasoning_content": "",
        "tool_calls": [{
            "id": call_id,
            "name": "transfer",
            "arguments": json.dumps({
                "target_agent": "recon",
                "task_description": "Check the target is reachable.",
            }),
        }],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _assert_tool_calls_are_answered(messages: list[dict]) -> None:
    """Every assistant tool_calls message must be followed by a tool reply per id."""
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        expected = [tc["id"] for tc in msg["tool_calls"]]
        # Collect the contiguous run of `tool` messages directly after this one.
        answered = []
        cursor = idx + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            answered.append(messages[cursor].get("tool_call_id"))
            cursor += 1
        assert answered == expected, (
            f"assistant tool_calls at index {idx} expects replies {expected}, "
            f"got {answered}"
        )


@pytest.mark.asyncio
async def test_supervisor_history_answers_every_tool_call(monkeypatch):
    calls: list[list[dict]] = []

    async def fake_chat_completion(llm_config, messages, tools=None, temperature=None):
        calls.append([dict(m) for m in messages])
        return _transfer_response(call_id=f"call_{len(calls)}")

    # _supervisor_node_llm imports chat_completion locally from app.agents.llm_client.
    import app.agents.llm_client as agents_llm_client
    monkeypatch.setattr(agents_llm_client, "chat_completion", fake_chat_completion)

    orch = SupervisorOrchestrator(
        scan_id="scan_test",
        target="http://example.com",
        user_prompt="test",
        transfer_targets=["recon"],
        llm_config={"provider": "openai_compatible", "model": "deepseek-flash",
                    "api_key": "sk-test", "base_url": "https://api.deepseek.com/v1"},
    )

    state: dict = {"target": "http://example.com", "status": "running", "decisions": []}
    await orch._supervisor_node_llm(state)

    # The expert node appends its result as a user message before the supervisor
    # is called again — mirror that here.
    orch._supervisor_messages.append({
        "role": "user",
        "content": "[Result from recon] reachability check finished.",
    })

    await orch._supervisor_node_llm({**state, "decisions": orch.decisions})

    assert len(calls) == 2, "supervisor LLM should have been invoked twice"
    # First call: just [system, user] — nothing to answer yet.
    assert not any(m.get("tool_calls") for m in calls[0])
    # Second call: the round-tripped history must be valid for the API.
    _assert_tool_calls_are_answered(calls[1])
    answered_ids = [m["tool_call_id"] for m in calls[1] if m.get("role") == "tool"]
    assert "call_1" in answered_ids
