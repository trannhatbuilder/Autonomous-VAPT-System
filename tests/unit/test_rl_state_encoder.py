"""
Tests for app.rl.state_encoder — 337-dim state vector encoder.

Covers:
  - State vector shape + dtype (337, float32)
  - Section layout (8 scalars + 256 tech + 32 vec + 32 tool + 9 flags)
  - Hash bucketing deterministic (SHA-256, not PYTHONHASHSEED)
  - Multi-hot tech stack encoding
  - One-hot last-vector / last-tool encoding
  - Flag section correctness
  - from_session dict adapter
  - Padding/truncation defense
  - Empty session edge case
  - Action space size (7) + names
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set test env BEFORE importing app modules
os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import numpy as np
import pytest

from app.rl.state_encoder import (
    StateEncoder, PentestState, StateEncodingConfig, DEFAULT_STATE_CONFIG,
    STATE_DIM, ACTION_DIM, ACTION_SPACE,
    TECH_STACK_VOCAB, VECTOR_LABELS, VECTOR_VOCAB, TOOL_VOCAB,
    _hash_bucket, _one_hot, encode_session,
)


# ── Section 1: Constants ─────────────────────────────────────────────────

class TestConstants:
    def test_state_dim_337(self):
        """STATE_DIM must be exactly 337 (8+256+32+32+9)."""
        assert STATE_DIM == 337

    def test_action_dim_7(self):
        assert ACTION_DIM == 7

    def test_action_space_names(self):
        """All 7 actions match master plan §12 W16 action space."""
        assert ACTION_SPACE == [
            "execute_command_recon",
            "execute_command_exploit",
            "execute_command_fuzz",
            "select_skill",
            "consult_kg",
            "report_vulnerability",
            "pentest_complete",
        ]

    def test_vocab_sizes(self):
        assert TECH_STACK_VOCAB == 256
        assert VECTOR_VOCAB == 32
        assert TOOL_VOCAB == 32


# ── Section 2: Helpers ───────────────────────────────────────────────────

class TestHelpers:
    def test_hash_bucket_deterministic(self):
        """SHA-256 hash must be reproducible across runs (not PYTHONHASHSEED)."""
        h1 = _hash_bucket("nginx", 256)
        h2 = _hash_bucket("nginx", 256)
        assert h1 == h2
        assert 0 <= h1 < 256

    def test_hash_bucket_different_inputs(self):
        """Different strings should (usually) hash to different buckets."""
        h1 = _hash_bucket("nginx", 256)
        h2 = _hash_bucket("apache", 256)
        assert h1 != h2  # not guaranteed but extremely likely with SHA-256

    def test_hash_bucket_range(self):
        for s in ["", "a", "nginx", "php", "wordpress"]:
            h = _hash_bucket(s, 256)
            assert 0 <= h < 256

    def test_one_hot_in_range(self):
        v = _one_hot(5, 32)
        assert v.shape == (32,)
        assert v.dtype == np.float32
        assert v[5] == 1.0
        assert v.sum() == 1.0

    def test_one_hot_out_of_range(self):
        """Out-of-range index → zero vector (no signal)."""
        v_neg = _one_hot(-1, 32)
        v_big = _one_hot(100, 32)
        assert v_neg.sum() == 0.0
        assert v_big.sum() == 0.0


# ── Section 3: PentestState.to_vector ────────────────────────────────────

class TestPentestStateVector:
    def _make_state(self, **overrides):
        defaults = dict(
            turn=10, tdi=0.5, mode="llm",
            findings_count=5, verified_findings=3, fp_findings=1,
            commands_run=20, tokens_used=10000, token_budget_remaining=0.5,
            tech_stack=("nginx", "php", "mysql"),
            last_vector="SQLi", last_tool="recon",
            last_success=True, kg_available=True,
            planner_paths_live=10, planner_paths_pruned=5,
            recon_login=True, recon_api=False, recon_upload=True,
            recon_graphql=False, recon_websocket=True,
        )
        defaults.update(overrides)
        return PentestState(**defaults)

    def test_vector_shape_and_dtype(self):
        state = self._make_state()
        vec = state.to_vector()
        assert vec.shape == (337,)
        assert vec.dtype == np.float32

    def test_scalar_section_8_dims(self):
        """Section 1: 8 scalar normalized features."""
        state = self._make_state(turn=50, tdi=0.3, mode="broaden")
        vec = state.to_vector()
        scalars = vec[:8]
        assert scalars.shape == (8,)
        # All scalars in [0, 1]
        assert (scalars >= 0).all() and (scalars <= 1).all()
        # Turn normalized
        assert abs(scalars[0] - 0.5) < 1e-6  # 50/100
        # TDI
        assert abs(scalars[1] - 0.3) < 1e-6
        # Mode broaden=1.0
        assert scalars[2] == 1.0

    def test_tech_stack_multihot(self):
        """Section 2: 256-dim tech stack multi-hot — 3 techs → 3 set bits."""
        state = self._make_state(tech_stack=("nginx", "php", "mysql"))
        vec = state.to_vector()
        tech_vec = vec[8:8 + 256]
        assert int(tech_vec.sum()) == 3
        # Each set bit is 1.0
        assert set(tech_vec[tech_vec > 0].tolist()) == {1.0}

    def test_last_vector_onehot(self):
        """Section 3: 32-dim one-hot — exactly 1 set bit.

        `last_vector="SQLi"` used to be used here, and it passed because
        `_hash_bucket` gave the unknown label a private bucket. That was the
        defect, not the contract: an unknown label must encode as the
        documented "no signal" zero vector, which `test_unknown_is_all_zero`
        below now pins.
        """
        state = self._make_state(last_vector="exploit")
        vec = state.to_vector()
        vec_section = vec[8 + 256:8 + 256 + 32]
        assert int(vec_section.sum()) == 1

    def test_unknown_last_vector_is_all_zero(self):
        """An unknown label must not invent a feature."""
        state = self._make_state(last_vector="SQLi")
        vec = state.to_vector()
        vec_section = vec[8 + 256:8 + 256 + 32]
        assert int(vec_section.sum()) == 0

    def test_last_tool_onehot(self):
        """Section 4: 32-dim one-hot — exactly 1 set bit."""
        state = self._make_state(last_tool="recon")
        vec = state.to_vector()
        tool_section = vec[8 + 256 + 32:8 + 256 + 32 + 32]
        assert int(tool_section.sum()) == 1

    def test_flags_section_9_dims(self):
        """Section 5: 9 flags."""
        state = self._make_state()
        vec = state.to_vector()
        flags = vec[-9:]
        assert flags.shape == (9,)
        # First flag is last_success (True → 1.0)
        assert flags[0] == 1.0
        # Second flag is kg_available (True → 1.0)
        assert flags[1] == 1.0
        # recon_login (True → 1.0), recon_api (False → 0.0)
        assert flags[4] == 1.0  # recon_login
        assert flags[5] == 0.0  # recon_api

    def test_reproducibility(self):
        """Same state → same vector (deterministic)."""
        state = self._make_state()
        v1 = state.to_vector()
        v2 = state.to_vector()
        assert np.array_equal(v1, v2)


# ── Section 4: StateEncoder.from_session ─────────────────────────────────

class TestStateEncoderFromSession:
    def test_full_session(self):
        session = {
            "_turn": 5, "_commands_run": 12,
            "_tokens_used": 5000, "_token_budget": 128000,
            "_last_command": "nuclei -u http://target.com",
            "_last_vector": "XSS",
            "_last_turn_success": True,
            "_kg_consulted": True,
            "tdi_log": [{"tdi": 0.3, "mode": "exploit"}],
            "recon_signals": {
                "technologies": ["nginx", "php"],
                "has_login": True, "has_api": False, "has_upload": True,
                "has_graphql": False, "has_websocket": True,
            },
            "findings": [
                {"name": "SQLi", "location": "/login", "severity": "high",
                 "verified": True, "false_positive": False},
                {"name": "XSS", "location": "/search", "severity": "medium",
                 "verified": False, "false_positive": True},
            ],
        }
        state = StateEncoder.from_session(session)
        assert state.turn == 5
        assert state.tdi == 0.3
        assert state.mode == "exploit"
        assert state.findings_count == 2
        assert state.verified_findings == 1
        assert state.fp_findings == 1
        assert state.commands_run == 12
        assert state.last_success is True
        assert state.kg_available is True
        assert state.recon_login is True
        assert state.recon_api is False
        # `nuclei` is credited as execute_command_fuzz by
        # `app.rl.actions.TOOL_TO_ACTION`; the encoder's `last_tool` now
        # derives from that table instead of a second hand-written heuristic
        # that called it recon (state contradicting its own reward).
        assert state.last_tool == "fuzz"  # nuclei → fuzz (credit bucket)

    def test_empty_session(self):
        """Empty session dict → all defaults, no crash."""
        state = StateEncoder.from_session({})
        assert state.turn == 0
        assert state.tdi == 0.5  # default
        assert state.mode == "llm"  # default
        assert state.findings_count == 0
        assert state.token_budget_remaining >= 0.0

    def test_encode_session_one_liner(self):
        session = {"_turn": 3}
        v1 = encode_session(session)
        v2 = StateEncoder.from_session(session).to_vector()
        assert np.array_equal(v1, v2)

    def test_infer_tool_from_command(self):
        """_infer_tool_from_command derives from the credit table.

        Every expectation below must equal `TOOL_TO_ACTION[name]` minus the
        `execute_command_` prefix; `nuclei` sits in fuzz because that is where
        credit assignment puts it (it is a scanner, not an exploit).
        """
        assert StateEncoder._infer_tool_from_command("nuclei -u http://x") == "fuzz"
        assert StateEncoder._infer_tool_from_command("nmap -sV 10.0.0.1") == "recon"
        assert StateEncoder._infer_tool_from_command("sqlmap -u http://x --batch") == "exploit"
        assert StateEncoder._infer_tool_from_command("msfconsole -q") == "exploit"
        assert StateEncoder._infer_tool_from_command("ffuf -u http://x/FUZZ") == "fuzz"
        assert StateEncoder._infer_tool_from_command("gobuster dir -u http://x") == "fuzz"
        # No evidence of a shell bucket is encoded as NO SIGNAL, not as
        # "recon". The old default made a turn that only recorded a finding
        # (or turn 1, which ran nothing) claim reconnaissance had happened.
        assert StateEncoder._infer_tool_from_command("") == ""
        assert StateEncoder._infer_tool_from_command("record_vulnerability") == ""

# ══════════════════════════════════════════════════════════════════════════
# Round 18 (SB2) — `kg_consulted` is PER-TURN, and it is actually populated
# ══════════════════════════════════════════════════════════════════════════
# `_session_builder.py` documented two contradictory contracts for the same
# flag in the same file:
#
#   header:    "_kg_consulted: False -> Phase 2 set True after first KG
#               consult"                              ← latch (sticky)
#   param doc: "True if KG was queried this turn"      ← per-turn
#
# A latch makes the encoder's `kg_available` dimension constant from turn 2
# onward — a dead feature. Stricter still: NO production caller passed the
# parameter at all, so the dimension was a permanent 0.0 (a dead constant
# from turn 1). Fixed both: one contract, and the caller now supplies it.
APP_DIR = Path(__file__).resolve().parents[2] / "app"
BUILDER = APP_DIR / "rl" / "_session_builder.py"
AGENT = APP_DIR / "agents" / "react_agent.py"


def _to_vector_call_names() -> list[str]:
    """Names of the module-level helpers called from `PentestState.to_vector`.

    AST rather than a source substring scan: `state_encoder.py` documents the
    calls it *used* to make, and a substring guard matches its own
    documentation (this project has made that mistake twice already).
    """
    import ast as _ast

    tree = _ast.parse((APP_DIR / "rl" / "state_encoder.py").read_text(encoding="utf-8"))
    fn = next(
        node for node in _ast.walk(tree)
        if isinstance(node, _ast.FunctionDef) and node.name == "to_vector"
    )
    return [
        node.func.id
        for node in _ast.walk(fn)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    ]


def _builder_doc() -> str:
    import ast as _ast

    tree = _ast.parse(BUILDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "build_session":
            return _ast.get_docstring(node) or ""
    raise AssertionError("build_session() not found")


class TestKgConsultedIsPerTurn:
    """The flag must describe ONE turn, and must reach the encoder."""

    # ── behavioural ──────────────────────────────────────────────────
    def _session(self, kg: bool) -> dict:
        from app.rl._session_builder import build_session

        return build_session(
            scan_id="s", target="t", turn=1, commands_run=0, tokens_used=0,
            token_budget=1000, last_command="", last_vector="",
            last_turn_success=False, findings=[], kg_consulted=kg,
        )

    def test_flag_reaches_the_encoder_dimension(self):
        assert float(StateEncoder.from_session(self._session(True)).kg_available) == 1.0
        assert float(StateEncoder.from_session(self._session(False)).kg_available) == 0.0

    def test_flag_does_not_latch_across_sessions(self):
        """A consult on turn N must not colour the state on turn N+1."""
        from app.rl._session_builder import build_session

        consulted = build_session(
            scan_id="s", target="t", turn=2, commands_run=1, tokens_used=10,
            token_budget=1000, last_command="nmap", last_vector="recon",
            last_turn_success=False, findings=[], kg_consulted=True,
        )
        after = build_session(
            scan_id="s", target="t", turn=3, commands_run=2, tokens_used=20,
            token_budget=1000, last_command="nmap", last_vector="recon",
            last_turn_success=False, findings=[], kg_consulted=False,
        )
        assert consulted["_kg_consulted"] is True
        assert after["_kg_consulted"] is False, (
            "build_session() latched the KG flag — the encoder dimension is "
            "then constant from the first consult onward"
        )

    def test_default_is_false(self):
        from app.rl._session_builder import build_session

        session = build_session(
            scan_id="s", target="t", turn=1, commands_run=0, tokens_used=0,
            token_budget=1000, last_command="", last_vector="",
            last_turn_success=False, findings=[],
        )
        assert session["_kg_consulted"] is False

    # ── documentation contract ───────────────────────────────────────
    def test_builder_header_does_not_describe_a_latch(self):
        src = BUILDER.read_text(encoding="utf-8")
        header = src.split('"""')[1]
        assert "after first KG consult" not in header, (
            "the module header still describes a sticky/latch semantic, "
            "contradicting the parameter doc in the same file"
        )
        assert "per-turn" in header

    def test_builder_param_doc_states_per_turn(self):
        doc = _builder_doc()
        assert "turn that just finished" in doc or "this turn" in doc
        assert "latch" in doc, (
            "the docstring no longer warns that a latched value would make "
            "the encoder dimension constant"
        )

    def test_no_latch_key_was_added(self):
        """Only add a key the encoder reads — it reads exactly one."""
        src = BUILDER.read_text(encoding="utf-8")
        assert "_kg_ever_consulted" not in src
        assert "_kg_consulted_count" not in src
        encoder_src = (APP_DIR / "rl" / "state_encoder.py").read_text(encoding="utf-8")
        assert "_kg_ever_consulted" not in encoder_src

    # ── the wiring that was missing entirely ─────────────────────────
    def test_agent_passes_the_flag_to_every_session(self):
        """AST, not text: comments mention `build_session(` too."""
        import ast as _ast

        tree = _ast.parse(AGENT.read_text(encoding="utf-8"))
        calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Name)
            and n.func.id == "build_session"
        ]
        assert len(calls) == 3, f"expected 3 call sites, found {len(calls)}"
        missing = [
            n.lineno for n in calls
            if not any(k.arg == "kg_consulted" for k in n.keywords)
        ]
        assert not missing, (
            f"build_session() at line(s) {missing} omit kg_consulted= — the "
            "parameter is dead there and the encoder dimension collapses to "
            "a constant 0.0"
        )

    def test_agent_resets_the_flag_per_turn(self):
        src = AGENT.read_text(encoding="utf-8")
        assert "last_turn_kg_consulted = False" in src, (
            "the per-turn KG flag is never reset — it silently becomes a "
            "scan-lifetime latch"
        )

    def test_agent_sets_the_flag_on_both_consult_paths(self):
        """Manual `consult_kg` AND the recon-triggered auto-consult."""
        src = AGENT.read_text(encoding="utf-8")
        assert src.count("last_turn_kg_consulted = True") == 2, (
            "expected exactly two setters: the manual consult_kg tool and "
            "the auto-consult after the first recon output"
        )

    def test_both_setters_are_after_a_kg_consult_call(self):
        """Guard the setters against drifting off the consult."""
        import ast as _ast

        tree = _ast.parse(AGENT.read_text(encoding="utf-8"))
        setters = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Assign):
                if any(
                    isinstance(t, _ast.Name) and t.id == "last_turn_kg_consulted"
                    for t in node.targets
                ) and isinstance(node.value, _ast.Constant) and node.value.value is True:
                    setters.append(node.lineno)
        assert len(setters) == 2

        src_lines = AGENT.read_text(encoding="utf-8").splitlines()
        consult_lines = [
            i + 1 for i, line in enumerate(src_lines)
            if "harness.kg_consult(" in line
        ]
        for line in setters:
            # The setter must follow a kg_consult() within the same block.
            assert any(0 <= line - c <= 6 for c in consult_lines), (
                f"last_turn_kg_consulted = True at line {line} is not adjacent "
                f"to a harness.kg_consult() call ({consult_lines})"
            )


# ══════════════════════════════════════════════════════════════════════════
# Round 19 (SB3) — `last_vector` is validated, and its vocabulary is derived
# ══════════════════════════════════════════════════════════════════════════
# The builder documented {"recon","exploit","fuzz","report","complete",""}
# and then `str()`-passed anything through. Three of the six tokens the real
# producer emits were ALREADY outside that set (`consult_kg`,
# `report_vulnerability`, `pentest_complete`) — a docstring is not a
# validator, so nothing noticed.
#
# The report's stated failure mode ("unknown bucket -> all-zero vector") is
# factually wrong: `_hash_bucket()` maps EVERY string into [0, 32), so an
# unknown label gets its own bucket rather than a zero vector. That is worse
# in one way (a typo impersonates a distinct attack vector) and unreachable
# in another (`_one_hot(-1)`'s zero vector cannot be produced from here).
# Both facts are pinned below so the wrong model does not come back.
VECTOR_BUILDER = Path(__file__).resolve().parents[2] / "app" / "rl" / "_session_builder.py"


@pytest.fixture()
def _clean_vector_warnings():
    """Warn-once state is module-level; isolate every test from its neighbours."""
    from app.rl import _session_builder as sb

    sb._WARNED_VECTORS.clear()
    yield sb._WARNED_VECTORS
    sb._WARNED_VECTORS.clear()


def _build(vector, **overrides) -> dict:
    from app.rl._session_builder import build_session

    kwargs = dict(
        scan_id="s", target="t", turn=1, commands_run=0, tokens_used=0,
        token_budget=1000, last_command="", last_vector=vector,
        last_turn_success=False, findings=[],
    )
    kwargs.update(overrides)
    return build_session(**kwargs)


class TestLastVectorIsValidated:
    def test_unknown_label_folds_to_empty(self, _clean_vector_warnings):
        assert _build("sqlmap")["_last_vector"] == ""

    def test_unknown_label_warns_once_per_value(self, _clean_vector_warnings, caplog):
        caplog.set_level("WARNING", logger="app.rl._session_builder")
        for _ in range(5):
            _build("sqlmap")
        _build("verify")
        warned = [r.message for r in caplog.records if "unknown last_vector" in r.message]
        assert len(warned) == 2, (
            "warn-once is broken — a single mis-mapping would log on every "
            "one of a scan's ~100 turns"
        )
        assert any("sqlmap" in w for w in warned)
        assert any("verify" in w for w in warned)

    def test_case_and_whitespace_are_folded(self, _clean_vector_warnings):
        assert _build("Recon")["_last_vector"] == "recon"
        assert _build("  EXPLOIT ")["_last_vector"] == "exploit"

    def test_non_string_input_does_not_raise(self, _clean_vector_warnings):
        """A bad label must not abort a running scan mid-episode."""
        for value in (None, 0, 17, b"recon", ["recon"]):
            assert _build(value)["_last_vector"] in VECTOR_LABELS

    def test_does_not_raise_on_unknown(self, _clean_vector_warnings):
        assert _build("total_nonsense")["_last_vector"] == ""  # no exception

    def test_unknown_labels_are_not_scattered_across_buckets(self, _clean_vector_warnings):
        """The point of folding: one bucket for "not a vector", not N."""
        buckets = {
            _hash_bucket(_build(label)["_last_vector"], VECTOR_VOCAB)
            for label in ("sqlmap", "verify", "scan", "Recon2", "zzz")
        }
        assert len(buckets) == 1
        assert buckets == {_hash_bucket("", VECTOR_VOCAB)}

    def test_canonical_labels_stay_distinguishable(self, _clean_vector_warnings):
        """Normalization must not collapse the labels that DO carry meaning."""
        buckets = [_hash_bucket(label, VECTOR_VOCAB) for label in VECTOR_LABELS]
        assert len(set(buckets)) == len(VECTOR_LABELS), (
            f"two VECTOR_LABELS share a hash bucket: "
            f"{dict(zip(VECTOR_LABELS, buckets))}"
        )


class TestVectorVocabularyIsDerived:
    def test_labels_match_action_space(self):
        expected = tuple(
            a.replace("execute_command_", "") for a in ACTION_SPACE
        ) + ("",)
        assert VECTOR_LABELS == expected
        assert len(VECTOR_LABELS) == ACTION_DIM + 1

    def test_every_token_the_producer_can_emit_is_legal(self):
        """The live-drift guard: this is what the docstring failed to be."""
        from app.agents.react_agent import _TOOL_TO_ACTION, _infer_vector_from_tools

        emitted = {_infer_vector_from_tools([tool]) for tool in _TOOL_TO_ACTION}
        illegal = emitted - set(VECTOR_LABELS)
        assert not illegal, (
            f"the agent produces last_vector values the builder rejects and "
            f"folds to '': {sorted(illegal)}"
        )

    def test_every_action_maps_to_a_legal_label(self):
        for action in ACTION_SPACE:
            assert action.replace("execute_command_", "") in VECTOR_LABELS

    def test_builder_derives_instead_of_restating(self):
        """A hand-maintained copy in the builder is how the drift started."""
        src = VECTOR_BUILDER.read_text(encoding="utf-8")
        assert "from app.rl.state_encoder import VECTOR_LABELS" in src
        assert '"recon", "exploit", "fuzz", "report", "complete"' not in src
        assert "'recon', 'exploit', 'fuzz', 'report', 'complete'" not in src

    def test_encoder_does_not_pretend_unknown_is_a_zero_vector(self):
        """Superseded by SE4 — the OPPOSITE is now the contract.

        This used to assert the source still contained "does NOT yield a zero
        vector", i.e. it pinned the hash behaviour as correct. Hashing a
        closed set bought two defects (aliasing risk, and a typo inventing a
        private feature) to solve a problem a tuple index does not have.
        Sections 3/4 are direct indices now, so an unknown label DOES yield
        the zero vector and this guard is inverted to say so.

        AST, not a substring scan: the explanatory comment above these lines
        quotes the old `_hash_bucket(self.last_vector, ...)` call, and a
        substring guard matches its own documentation (a mistake already made
        twice in this project).
        """
        calls = _to_vector_call_names()
        assert calls.count("_label_index") == 2, (
            "Sections 3 and 4 must both be direct indices: "
            f"_label_index calls in to_vector = {calls.count('_label_index')}"
        )

    def test_hash_bucket_is_only_used_for_the_open_set(self):
        """Section 2 alone needs hashing — the tech stack is unbounded."""
        calls = _to_vector_call_names()
        assert calls.count("_hash_bucket") == 1, (
            "_hash_bucket should be called exactly once in to_vector — for "
            "the tech multi-hot. Anything else is a closed set being hashed "
            f"again: {calls}"
        )

    def test_hash_bucket_never_returns_out_of_range(self):
        """Section 2 invariant: the tech bucket index is always in range."""
        for label in ["", "recon", "sqlmap", "\u0000", "\u4e2d\u6587", "x" * 500]:
            idx = _hash_bucket(label, TECH_STACK_VOCAB)
            assert 0 <= idx < TECH_STACK_VOCAB

    def test_no_label_ever_encodes_to_all_zeros(self):
        vec = StateEncoder.from_session(_build("") ) .to_vector()
        section = vec[8 + TECH_STACK_VOCAB: 8 + TECH_STACK_VOCAB + VECTOR_VOCAB]
        assert not np.array_equal(section, np.zeros(VECTOR_VOCAB, dtype=np.float32)), (
            "the vector section IS all-zero for '': _one_hot(-1) is reachable "
            "after all and the builder's fallback is genuine 'no signal'"
        )


# ══════════════════════════════════════════════════════════════════════════
# Round 20 (SB4) — `egats_planner` is plain data, not "Anything"
# ══════════════════════════════════════════════════════════════════════════
# The builder took `egats_planner: Any | None` and forwarded it verbatim into
# the session dict. A live `EGATS` instance is mutable and has no JSON form,
# so any consumer that serialized the session would raise
# `TypeError: Object of type EGATS is not JSON serializable`.
#
# Two things the report got wrong, pinned here so the wrong fix cannot land:
#   1. Nothing actually serialized the session, so the "time bomb" had no
#      fuse — but the `Any` was real and the field had ZERO callers, i.e. the
#      two planner dimensions were permanently 0 (same class as SB2's
#      `kg_consulted`).
#   2. Changing only the type hint would have been a WORSE bug: the encoder
#      introspected the planner with `planner.stats()` inside a bare
#      `except Exception: pass`, so a dict silently produced 0/0 forever.
IMPORTED_PLANNER = {"total_paths": 4, "alive": 3, "pruned": 1}


@pytest.fixture()
def _clean_planner_warnings():
    from app.rl import _session_builder as sb

    sb._WARNED_PLANNERS.clear()
    yield sb._WARNED_PLANNERS
    sb._WARNED_PLANNERS.clear()


class TestEgatsPlannerIsSerializable:
    def test_session_with_planner_snapshot_survives_json_dumps(self):
        import json

        session = _build("recon", egats_planner=IMPORTED_PLANNER)
        assert json.loads(json.dumps(session))["egats_planner"] == IMPORTED_PLANNER

    def test_session_without_planner_is_json_serializable(self):
        import json

        json.dumps(_build("recon"))  # None must not break dumps

    def test_live_planner_object_is_rejected(self, _clean_planner_warnings):
        """The builder is a serialization boundary: no live objects."""
        from app.rl.egats import EGATS

        session = _build("recon", egats_planner=EGATS())
        assert session["egats_planner"] is None
        assert _clean_planner_warnings == {"EGATS"}

    def test_live_planner_rejection_warns_once(self, _clean_planner_warnings, caplog):
        from app.rl.egats import EGATS

        caplog.set_level("WARNING", logger="app.rl._session_builder")
        for _ in range(4):
            _build("recon", egats_planner=EGATS())
        hits = [r for r in caplog.records if "egats_planner" in r.message]
        assert len(hits) == 1, "per-call logging would fire on every turn"

    def test_snapshot_is_copied_not_aliased(self):
        """A caller mutating its dict must not rewrite a built session."""
        source = {"alive": 2, "pruned": 0}
        session = _build("recon", egats_planner=source)
        source["alive"] = 99
        assert session["egats_planner"]["alive"] == 2

    def test_encoder_reads_counts_from_the_snapshot(self):
        state = StateEncoder.from_session(_build("recon", egats_planner=IMPORTED_PLANNER))
        assert (state.planner_paths_live, state.planner_paths_pruned) == (3, 1)

    def test_encoder_caps_counts_in_the_flag_dims(self):
        vec = StateEncoder.from_session(
            _build("recon", egats_planner={"alive": 500, "pruned": 500})
        ).to_vector()
        flags = vec[8 + TECH_STACK_VOCAB + VECTOR_VOCAB + TOOL_VOCAB:]
        assert flags.dtype == np.float32
        assert float(flags.max()) <= 1.0

    def test_missing_or_bad_counts_fall_back_to_zero(self):
        for snapshot in (None, {}, {"alive": "x"}, {"alive": -5}, {"pruned": None}):
            state = StateEncoder.from_session(
                _build("recon", egats_planner=snapshot)
            )
            assert state.planner_paths_live >= 0
            assert state.planner_paths_pruned >= 0
        assert StateEncoder.from_session(
            _build("recon", egats_planner={"alive": "x"})
        ).planner_paths_live == 0

    def test_egats_to_dict_is_json_serializable(self):
        import json

        from app.rl.egats import EGATS

        planner = EGATS()
        planner.seed_from_kg(
            type("FakeKG", (), {"get_attack_paths": staticmethod(
                lambda *a, **k: [{"nodes": ["a", "b"], "edges": [], "score": 1.0}]
            )})(),
            ["a"],
        )
        snapshot = planner.to_dict()
        json.dumps(snapshot)  # must not raise
        assert snapshot["alive"] + snapshot["pruned"] == snapshot["total_paths"] == 1
        assert not any(isinstance(v, EGATS) for v in snapshot.values())

    def test_harness_snapshot_is_empty_before_seeding(self):
        import json

        from app.rl.egats import EGATS
        from app.rl.harness import HarnessBridge

        # `object.__new__`: HarnessBridge.__init__ spawns a training worker and
        # needs a running event loop. This test only exercises the accessor.
        bridge = object.__new__(HarnessBridge)
        bridge._egats_enabled = True
        bridge._egats = None
        assert bridge.egats_snapshot() == {}
        json.dumps(bridge.egats_snapshot())

        bridge._egats = EGATS()
        snapshot = bridge.egats_snapshot()
        assert snapshot["alive"] == 0 and snapshot["pruned"] == 0
        json.dumps(snapshot)


class TestPlannerContractGuards:
    def test_builder_annotation_is_not_any(self):
        """AST, not text: a comment mentioning `Any` must not pass the guard."""
        import ast

        tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_session"
        )
        arg = next(a for a in fn.args.kwonlyargs if a.arg == "egats_planner")
        annotation = ast.unparse(arg.annotation)
        assert "Any" not in annotation, annotation
        assert "EgatsPlannerSnapshot" in annotation, annotation

    def test_encoder_does_not_introspect_a_live_planner(self):
        """AST guard — docstrings/comments may name `.stats()`, code may not."""
        import ast

        tree = ast.parse((APP_DIR / "rl" / "state_encoder.py").read_text(encoding="utf-8"))
        offending = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "stats"
            and isinstance(node.value, ast.Name)
            and node.value.id == "planner"
        ]
        assert not offending, (
            "the encoder must read the plain-data snapshot, not walk a live "
            "planner (a dict has no .stats(), and the old bare except turned "
            "that AttributeError into a silent 0/0)"
        )

    def test_encoder_has_no_swallowing_except_around_planner(self):
        src = (APP_DIR / "rl" / "state_encoder.py").read_text(encoding="utf-8")
        assert "except Exception:\n                pass" not in src

    def test_all_build_session_calls_pass_a_planner_snapshot(self):
        """AST — `src.count("build_session(")` also matches comments/docstrings."""
        import ast

        tree = ast.parse(AGENT.read_text(encoding="utf-8"))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "build_session"
        ]
        assert len(calls) == 3, f"expected 3 production call sites, got {len(calls)}"
        for call in calls:
            passed = {kw.arg: kw.value for kw in call.keywords}
            assert "egats_planner" in passed, (
                f"build_session at line {call.lineno} omits egats_planner — the "
                f"two planner dims stay permanently 0"
            )
            assert "egats_snapshot" in ast.unparse(passed["egats_planner"]), ast.unparse(
                passed["egats_planner"]
            )

    def test_builder_doc_warns_against_passing_the_live_planner(self):
        text = BUILDER.read_text(encoding="utf-8")
        assert "EGATS.to_dict()" in text
        assert "NOT the live" in text


# ══════════════════════════════════════════════════════════════════════════
# Round 21 (SB5) — the session dict is a detached SNAPSHOT, and stateless
# ══════════════════════════════════════════════════════════════════════════
# `list(findings)` copies the list but shares the finding dicts, so a verifier
# that flipped `verified` after the call would rewrite a state a consumer had
# already encoded. Same aliasing on `recon_signals` (passed straight through,
# not copied at all) and on the entries of `tdi_log`.
#
# The report's stated trigger ("create_task fire-and-forget verifier") does
# not exist in react_agent — but the invariant was still false, and a snapshot
# that only *claims* to be one is a trap for the next consumer.
def _finding(**overrides) -> dict:
    from app.evidence.finding_state import finding_to_state

    base = finding_to_state(
        {"title": "SQLi", "location": "http://t/a?id=1", "severity": "high"},
        {"status": "recorded", "verification": "confirmed"},
    )
    base.update(overrides)
    return base


class TestSessionIsADetachedSnapshot:
    def test_finding_dicts_are_copied_not_shared(self):
        finding = _finding()
        session = _build("recon", findings=[finding])
        assert session["findings"][0] is not finding
        assert session["findings"][0] == finding

    def test_mutating_a_finding_after_build_cannot_change_the_session(self):
        """The exact failure: a late verifier flips the verdict on us."""
        finding = _finding()
        session = _build("recon", findings=[finding])
        before = StateEncoder.from_session(session).to_vector()

        finding["verified"] = False
        finding["false_positive"] = True
        finding["verification"] = "plausible"

        assert session["findings"][0]["verified"] is True
        assert np.array_equal(StateEncoder.from_session(session).to_vector(), before), (
            "an already-encoded state changed when the caller mutated a "
            "finding — the session aliased live state"
        )

    def test_appending_to_the_callers_list_cannot_change_the_session(self):
        finding = _finding()
        findings = [finding]
        session = _build("recon", findings=findings)
        findings.append(_finding(title="other"))
        findings.clear()
        assert len(session["findings"]) == 1

    def test_recon_signals_are_copied_at_both_levels(self):
        recon = {"technologies": ["nginx"], "has_api": True}
        session = _build("recon", recon_signals=recon)
        recon["technologies"].append("django")
        recon["has_api"] = False
        assert session["recon_signals"] == {"technologies": ["nginx"], "has_api": True}

    def test_tdi_log_entries_are_copied(self):
        tdi = [{"tdi": 0.4, "mode": "llm"}]
        session = _build("recon", tdi_log=tdi)
        tdi[0]["tdi"] = 0.95
        assert session["tdi_log"][0]["tdi"] == 0.4

    def test_planner_snapshot_nested_paths_are_copied(self):
        snapshot = {"alive": 1, "pruned": 0, "paths": [{"nodes": ["a", "b"], "attempts": 2}]}
        session = _build("recon", egats_planner=snapshot)
        snapshot["paths"][0]["attempts"] = 99
        assert session["egats_planner"]["paths"][0]["attempts"] == 2

    def test_omitted_collections_default_to_fresh_containers(self):
        session = _build("recon")
        assert session["findings"] == []
        assert session["tdi_log"] == []
        assert session["recon_signals"] == {}
        session["findings"].append({"x": 1})
        assert _build("recon")["findings"] == [], "default containers are shared"

    def test_snapshot_value_terminates_on_a_cycle(self):
        from app.rl._session_builder import _snapshot_value

        cyclic: dict = {}
        cyclic["self"] = cyclic
        assert isinstance(_snapshot_value(cyclic), dict)  # no RecursionError


class TestDepthTwoIsSufficient:
    def test_finding_state_values_are_all_immutable_scalars(self):
        """Proof that a depth-2 copy is a COMPLETE copy, not a best effort.

        If a future key holds a list/dict, `_snapshot_value(depth=2)` starts
        sharing it and the "snapshot" claim becomes false again.
        """
        from app.evidence.finding_state import FINDING_STATE_KEYS

        finding = _finding(false_positive_at="2026-10-17T00:00:00Z")
        assert set(finding) == set(FINDING_STATE_KEYS)
        mutable = {k: type(v).__name__ for k, v in finding.items() if isinstance(v, (list, dict, set))}
        assert not mutable, f"finding keys hold mutable values: {mutable}"

    def test_builder_does_not_shallow_copy_the_findings_list(self):
        """AST: the value bound to 'findings' must not be a bare `list(...)`."""
        import ast

        tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_session"
        )
        returned = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
        pairs = {
            k.value: v for k, v in zip(returned.value.keys, returned.value.values)
        }
        findings_expr = ast.unparse(pairs["findings"])
        assert "list(findings)" not in findings_expr, (
            "list() shares the inner finding dicts — use _snapshot_value()"
        )
        assert "_snapshot_value" in findings_expr, findings_expr


class TestStatelessContractIsDocumented:
    def test_module_docstring_warns_against_persisting_state(self):
        """The R2 lesson must be readable where the next author will look."""
        from app.rl._session_builder import __doc__ as module_doc

        text = module_doc or ""
        assert "STATELESS" in text
        assert "new_credit_state()" in text
        for key in ("_seen_finding_keys", "_credited_verified_ids", "_penalised_fp_ids"):
            assert key in text, f"module docstring must name {key}"
        assert "re-initialised every turn" in text

    def test_returns_section_states_both_properties(self):
        from app.rl._session_builder import build_session

        doc = build_session.__doc__ or ""
        assert "detached copy" in doc
        assert "no scanned-lifetime state" in doc


# ══════════════════════════════════════════════════════════════════════════
# Round 22 (SB6/SB7) — numeric boundary coercion + a feature-status header
# that is finally true (and `recon_signals`, which was not wired at all)
# ══════════════════════════════════════════════════════════════════════════
# SB6 asked for `int(x or 0)` on three counters and `max(int(budget or 0), 1)`.
# The `int(None)` claim was real for the expression but unreachable from any
# production call site; the div-by-zero was already guarded in the encoder;
# and the `max(..., 1)` floor would have reported "100% of the budget remains"
# for an unknown budget. The coercion itself was worth adding — every caller
# wraps `build_session()` in a swallowing try/except, so a bad counter drops
# the whole RL step silently.
#
# SB7's "Phase-3 status" complaint was accurate for `tdi_log` and INVERTED for
# `recon_signals`, which the header called "populated in Phase 2" while no
# caller passed it — 256 of 337 dims were zero on every turn.
@pytest.fixture
def _clean_count_warnings():
    from app.rl import _session_builder as builder_module

    builder_module._WARNED_COUNTS.clear()
    yield
    builder_module._WARNED_COUNTS.clear()


def _session_for(**overrides) -> dict:
    from app.rl._session_builder import build_session

    base = dict(
        scan_id="scan-1", target="http://t", turn=1, commands_run=0,
        tokens_used=0, token_budget=500_000, last_command="", last_vector="",
        last_turn_success=False, findings=[],
    )
    base.update(overrides)
    return build_session(**base)


class TestNumericBoundaryNeverRaises:
    """`_as_count` degrades to 0 + one warning; it never propagates a crash."""

    @pytest.mark.parametrize("bad", [None, "abc", float("inf"), float("nan"), {}, [], object()])
    def test_uncoercible_turn_becomes_zero(self, bad, _clean_count_warnings):
        session = _session_for(turn=bad)
        assert session["_turn"] == 0
        assert isinstance(session["_turn"], int)

    @pytest.mark.parametrize(
        "value,expected",
        [(7, 7), ("8", 8), (3.9, 3), (True, 1), (False, 0)],
    )
    def test_coercible_values_keep_their_old_behaviour(self, value, expected):
        """Coercion must not change what already worked."""
        assert _session_for(turn=value)["_turn"] == expected

    def test_all_four_counters_are_coerced(self, _clean_count_warnings):
        session = _session_for(
            turn=None, commands_run=None, tokens_used=None, token_budget=None
        )
        for key in ("_turn", "_commands_run", "_tokens_used", "_token_budget"):
            assert session[key] == 0, key

    def test_the_encoder_still_accepts_the_degraded_session(self, _clean_count_warnings):
        """The point of not raising: a bad counter must not kill the turn."""
        state = StateEncoder.from_session(_session_for(turn=None, tokens_used=None))
        assert state.turn == 0
        assert state.tokens_used == 0

    def test_warns_once_per_field_not_once_per_call(self, _clean_count_warnings, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                _session_for(turn=None, commands_run=None)
        hits = [r for r in caplog.records if "is not an integer" in r.getMessage()]
        assert len(hits) == 2, f"expected one warning per field, got {len(hits)}"

    def test_overflowerror_is_handled(self, _clean_count_warnings):
        """`int(float('inf'))` raises OverflowError, not ValueError."""
        assert _session_for(tokens_used=float("inf"))["_tokens_used"] == 0


class TestTokenBudgetIsNotFlooredToOne:
    """The report's `max(int(token_budget or 0), 1)` would have been a lie."""

    def test_zero_budget_stays_zero(self):
        assert _session_for(token_budget=0)["_token_budget"] == 0

    def test_unknown_budget_reports_zero_remaining_not_full(self):
        """1.0 would mean "the whole budget is left" — the opposite of unknown."""
        state = StateEncoder.from_session(_session_for(token_budget=0, tokens_used=0))
        assert state.token_budget_remaining == 0.0

    def test_builder_does_not_clamp_the_budget(self):
        """AST: no `max(...)` around the budget — the encoder owns that case."""
        import ast

        tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_session"
        )
        returned = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
        pairs = {k.value: v for k, v in zip(returned.value.keys, returned.value.values)}
        expr = ast.unparse(pairs["_token_budget"])
        assert "max(" not in expr, expr
        assert "_as_count" in expr, expr

    def test_encoder_guards_the_division_itself(self):
        """Why the builder does not need to: the guard already exists."""
        src = (APP_DIR / "rl" / "state_encoder.py").read_text(encoding="utf-8")
        assert "if token_budget <= 0:" in src


class TestReconSignalsAreWired:
    """256 of the 337 dims were dead: no caller ever passed recon_signals."""

    def test_harness_accessor_is_empty_before_recon(self):
        from app.rl.harness import HarnessBridge

        bridge = object.__new__(HarnessBridge)  # no event loop needed
        bridge._tech_stack = []
        assert bridge.recon_signals() == {}

    def test_harness_accessor_returns_a_copy(self):
        from app.rl.harness import HarnessBridge

        bridge = object.__new__(HarnessBridge)
        bridge._tech_stack = ["nginx", "php"]
        signals = bridge.recon_signals()
        assert signals == {"technologies": ["nginx", "php"]}
        signals["technologies"].append("django")
        assert bridge._tech_stack == ["nginx", "php"]

    def test_accessor_output_lights_up_the_tech_block(self):
        """End-to-end: the wiring is what makes these dims non-zero."""
        import numpy as np
        from app.rl.harness import HarnessBridge

        bridge = object.__new__(HarnessBridge)
        bridge._tech_stack = ["nginx", "php"]

        off = StateEncoder.from_session(_session_for()).to_vector()
        on = StateEncoder.from_session(
            _session_for(recon_signals=bridge.recon_signals())
        ).to_vector()
        assert np.count_nonzero(off[8:264]) == 0
        assert np.count_nonzero(on[8:264]) == 2

    def test_no_build_session_call_is_a_decode_footgun(self):
        """AST — s and s' must encode the same features, or the delta is noise."""
        import ast

        tree = ast.parse(AGENT.read_text(encoding="utf-8"))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "build_session"
        ]
        assert len(calls) == 3, f"expected 3 production call sites, got {len(calls)}"
        for call in calls:
            passed = {kw.arg: kw.value for kw in call.keywords}
            assert "recon_signals" in passed, (
                f"build_session at line {call.lineno} omits recon_signals — "
                f"256 of 337 dims stay permanently 0"
            )
            assert "recon_signals" in ast.unparse(passed["recon_signals"])


class TestFeatureStatusHeaderIsHonest:
    """A status comment that lies is worse than no comment."""

    def _header(self) -> str:
        from app.rl import _session_builder as builder_module

        return (builder_module.__doc__ or "").split("The encoder tolerates")[0]

    def test_tdi_log_is_not_described_as_a_future_stub(self):
        header = self._header()
        assert "Phase 3 populated" not in header
        assert "ACTIVE (Phase 3)" in header

    def test_recon_signals_is_documented_as_wired(self):
        header = self._header()
        assert "Phase 2 populated" not in header
        assert "WIRED" in header
        assert "HarnessBridge.recon_signals()" in header

    def test_egats_planner_is_documented_as_active(self):
        assert "ACTIVE (Phase 4)" in self._header()

    def test_header_states_the_cost_of_an_unwired_field(self):
        assert "256 of the 337" in self._header()
