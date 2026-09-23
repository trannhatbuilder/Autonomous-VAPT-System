"""
Tests for app.trace.replay — ReplayTraceRecorder + JSONL trace engine.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import pytest

from app.trace import append_trace_event, read_trace, get_trace_stats, trace_path, ReplayTraceRecorder


@pytest.fixture
def trace_dir(tmp_path):
    """Per-test trace directory."""
    return tmp_path


class TestAppendReadTrace:
    def test_append_and_read(self, trace_dir):
        """append_trace_event writes JSONL, read_trace returns events."""
        append_trace_event("scan_001", "scan_start", scan_id="scan_001",
                          data={"target": "http://x.com"}, trace_dir=trace_dir)
        append_trace_event("scan_001", "turn_start", turn=0,
                          data={"state_dim": 337}, trace_dir=trace_dir)
        events = read_trace("scan_001", trace_dir=trace_dir)
        assert len(events) == 2
        assert events[0]["event"] == "scan_start"
        assert events[1]["event"] == "turn_start"
        assert events[1]["turn"] == 0

    def test_read_nonexistent_trace(self, trace_dir):
        """read_trace returns [] for non-existent trace."""
        events = read_trace("nonexistent", trace_dir=trace_dir)
        assert events == []

    def test_trace_path_sanitization(self, trace_dir):
        """trace_path sanitizes unsafe characters."""
        p = trace_path("scan_abc/evil<>:", trace_dir=trace_dir)
        assert "/" not in p.name
        assert "<" not in p.name
        assert p.suffix == ".jsonl"

    def test_long_value_truncation(self, trace_dir):
        """Long strings are truncated with SHA-256 hash + preview."""
        long_output = "A" * 5000
        append_trace_event("scan_001", "tool_executed", turn=0,
                          data={"output": long_output}, trace_dir=trace_dir)
        events = read_trace("scan_001", trace_dir=trace_dir)
        output = events[0]["data"]["output"]
        assert isinstance(output, dict)
        assert output["truncated"] is True
        assert "sha256" in output
        assert len(output["preview"]) <= 4000

    def test_event_has_timestamp(self, trace_dir):
        """Every event has an ISO 8601 timestamp."""
        append_trace_event("scan_001", "test", trace_dir=trace_dir)
        events = read_trace("scan_001", trace_dir=trace_dir)
        assert "ts" in events[0]
        assert "T" in events[0]["ts"]  # ISO 8601

    def test_limit_param(self, trace_dir):
        """read_trace respects limit param."""
        for i in range(10):
            append_trace_event("scan_001", "event", turn=i, trace_dir=trace_dir)
        events = read_trace("scan_001", limit=3, trace_dir=trace_dir)
        assert len(events) == 3


class TestGetTraceStats:
    def test_stats_for_empty_trace(self, trace_dir):
        """get_trace_stats returns zeros for non-existent trace."""
        stats = get_trace_stats("nonexistent", trace_dir=trace_dir)
        assert stats["event_count"] == 0
        assert stats["turn_count"] == 0
        assert stats["file_size_bytes"] == 0

    def test_stats_with_events(self, trace_dir):
        """get_trace_stats returns correct counts."""
        append_trace_event("scan_001", "scan_start", trace_dir=trace_dir)
        for i in range(5):
            append_trace_event("scan_001", "turn", turn=i, trace_dir=trace_dir)
        stats = get_trace_stats("scan_001", trace_dir=trace_dir)
        assert stats["event_count"] == 6
        assert stats["turn_count"] == 5  # max turn is 4, +1
        assert stats["file_size_bytes"] > 0
        assert stats["first_event_ts"] is not None
        assert stats["last_event_ts"] is not None


class TestReplayTraceRecorder:
    def test_recorder_tracks_turn_count(self, trace_dir):
        """ReplayTraceRecorder tracks turn_count from events."""
        recorder = ReplayTraceRecorder("scan_001", trace_dir=trace_dir)
        recorder.event("turn_start", turn=0)
        recorder.event("turn_start", turn=1)
        recorder.event("turn_start", turn=2)
        assert recorder._turn_count == 3

    def test_recorder_tracks_total_reward(self, trace_dir):
        """ReplayTraceRecorder accumulates reward."""
        recorder = ReplayTraceRecorder("scan_001", trace_dir=trace_dir)
        recorder.event("turn_end", turn=0, reward=1.0)
        recorder.event("turn_end", turn=1, reward=5.0)
        recorder.event("turn_end", turn=2, reward=-2.0)
        assert abs(recorder._total_reward - 4.0) < 1e-6

    def test_recorder_writes_jsonl_file(self, trace_dir):
        """Recorder events are written to JSONL file."""
        recorder = ReplayTraceRecorder("scan_001", trace_dir=trace_dir)
        recorder.event("scan_start", data={"target": "http://x.com"})
        recorder.event("turn_start", turn=0)
        recorder.event("turn_end", turn=0, reward=1.0)
        events = read_trace("scan_001", trace_dir=trace_dir)
        assert len(events) == 3

    def test_recorder_kg_edge_tracking(self, trace_dir):
        """record_kg_edge_update tracks edge outcomes."""
        recorder = ReplayTraceRecorder("scan_001", trace_dir=trace_dir)
        recorder.record_kg_edge_update("edge_1", 0.5, 0.6, True)
        recorder.record_kg_edge_update("edge_2", 0.5, 0.4, False)
        assert len(recorder._kg_edges_updated) == 2
        assert recorder._kg_edges_updated[0]["success"] is True
        assert recorder._kg_edges_updated[1]["success"] is False

    def test_finalize_writes_scan_complete_event(self, trace_dir):
        """finalize() writes a scan_complete trace event."""
        recorder = ReplayTraceRecorder("scan_001", trace_dir=trace_dir)
        recorder.event("turn_start", turn=0, reward=1.0)

        # Mock session for finalize (no real DB)
        class MockSession:
            async def execute(self, *args, **kwargs):
                class R:
                    def scalars(self):
                        class S:
                            def first(self): return None
                        return S()
                return R()
            def add(self, *a): pass
            async def flush(self): pass

        asyncio.run(recorder.finalize(MockSession(), scan_status="completed"))

        events = read_trace("scan_001", trace_dir=trace_dir)
        scan_complete = [e for e in events if e["event"] == "scan_complete"]
        assert len(scan_complete) == 1
        assert scan_complete[0]["data"]["scan_status"] == "completed"
        assert scan_complete[0]["data"]["turn_count"] == 1