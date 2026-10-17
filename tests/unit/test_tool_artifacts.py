"""
Unit tests — per-scan tool output artifacts (`app/agents/tool_artifacts.py`).

The artifact store is what makes the truncation notices truthful: the LLM
sees a capped result (`MAX_LLM_OUTPUT_CHARS`) and pruned history
(`PRUNE_TO_CHARS`), but the full output stays retrievable by execution_id
via the `fetch_tool_output` tool.
"""
from __future__ import annotations

import pytest

from app.agents import tool_artifacts as art


@pytest.fixture(autouse=True)
def _clean_store():
    """Isolate every test — the store is module-level process state."""
    art.clear_scan_artifacts("scan_a")
    art.clear_scan_artifacts("scan_b")
    yield
    art.clear_scan_artifacts("scan_a")
    art.clear_scan_artifacts("scan_b")


class TestRecordAndGet:
    def test_roundtrip(self):
        art.record_artifact("scan_a", "exec-1", "sqlmap", "full output here")
        found = art.get_artifact("scan_a", "exec-1")
        assert found is not None
        assert found["body"] == "full output here"
        assert found["tool"] == "sqlmap"
        assert found["total_chars"] == len("full output here")

    def test_unknown_id_is_none(self):
        assert art.get_artifact("scan_a", "nope") is None

    def test_scans_are_isolated(self):
        art.record_artifact("scan_a", "exec-1", "nmap", "a")
        assert art.get_artifact("scan_b", "exec-1") is None

    def test_empty_ids_are_ignored(self):
        art.record_artifact("", "exec-1", "nmap", "x")
        art.record_artifact("scan_a", "", "nmap", "x")
        assert art.list_artifacts("scan_a") == []

    def test_metadata_preserved(self):
        art.record_artifact(
            "scan_a", "exec-1", "nuclei", "body",
            spill_path="/tmp/spill.txt", command="nuclei -u http://t",
        )
        found = art.get_artifact("scan_a", "exec-1")
        assert found["spill_path"] == "/tmp/spill.txt"
        assert found["command"] == "nuclei -u http://t"

    def test_oversized_artifact_is_clipped_with_note(self):
        huge = "A" * (art.MAX_ARTIFACT_CHARS + 500)
        art.record_artifact("scan_a", "exec-big", "nmap", huge)
        found = art.get_artifact("scan_a", "exec-big")
        assert len(found["body"]) == art.MAX_ARTIFACT_CHARS
        assert found["total_chars"] == len(huge)
        assert "capped" in found["clip_note"]


class TestLatestLookup:
    def test_find_latest_returns_newest_for_tool(self):
        art.record_artifact("scan_a", "exec-1", "sqlmap", "first")
        art.record_artifact("scan_a", "exec-2", "sqlmap", "second")
        art.record_artifact("scan_a", "exec-3", "nmap", "other tool")
        latest = art.find_latest("scan_a", "sqlmap")
        assert latest is not None
        assert latest["execution_id"] == "exec-2"
        assert latest["body"] == "second"

    def test_find_latest_unknown_tool_is_none(self):
        assert art.find_latest("scan_a", "nikto") is None


class TestEviction:
    def test_fifo_cap(self):
        for i in range(art.MAX_ARTIFACTS_PER_SCAN + 3):
            art.record_artifact("scan_a", f"exec-{i}", "nmap", f"body-{i}")
        listed = art.list_artifacts("scan_a")
        assert len(listed) == art.MAX_ARTIFACTS_PER_SCAN
        # Oldest entries were dropped, newest kept.
        assert art.get_artifact("scan_a", "exec-0") is None
        newest = art.MAX_ARTIFACTS_PER_SCAN + 2
        assert art.get_artifact("scan_a", f"exec-{newest}") is not None

    def test_listing_is_ordered_and_bodyless(self):
        art.record_artifact("scan_a", "exec-1", "nmap", "b1")
        art.record_artifact("scan_a", "exec-2", "httpx", "b2")
        listed = art.list_artifacts("scan_a")
        assert [e["execution_id"] for e in listed] == ["exec-1", "exec-2"]
        assert all("body" not in e for e in listed)


class TestSpillFallback:
    def test_read_spill_reads_file(self, tmp_path):
        path = tmp_path / "output_x.txt"
        path.write_text("spilled content", encoding="utf-8")
        assert art.read_spill(str(path)) == "spilled content"

    def test_read_spill_missing_file_is_none(self, tmp_path):
        assert art.read_spill(str(tmp_path / "nope.txt")) is None
        assert art.read_spill(None) is None


class TestClear:
    def test_clear_drops_everything(self):
        art.record_artifact("scan_a", "exec-1", "nmap", "body")
        art.clear_scan_artifacts("scan_a")
        assert art.get_artifact("scan_a", "exec-1") is None
        assert art.list_artifacts("scan_a") == []
        assert art.find_latest("scan_a", "nmap") is None
