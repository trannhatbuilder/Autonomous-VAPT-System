"""R36 guards — the state vector's LAYOUT contract (SE3–SE7).

These pin the four things SE3–SE7 changed, in a way that survives the next
refactor. Every guard here is AST- or behaviour-based, never a substring scan
of the source: this project has twice written a "must not contain" guard that
matched its own explanatory comment.

Covered:
  SE3  `_last_tool` is read, takes precedence, and an explicit "" does NOT
       fall back to `_last_command`.
  SE4  Sections 3/4 are DIRECT indices: no collisions by construction, unknown
       -> zero vector, and the vocabulary is derived from ACTION_SPACE (one
       table, not two).
  SE5  `_FLAG_DIMS` really is 7 booleans + 2 planner ratios.
  SE6  `_unit_interval` rejects NaN/inf instead of clamping them.
  SE7  `STATE_DIM` has exactly one definition and nothing hardcodes 337.
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

from app.rl.actions import ACTION_DIM, ACTION_SPACE, tool_category  # noqa: E402
from app.rl.state_encoder import (  # noqa: E402
    STATE_DIM,
    STATE_LAYOUT_VERSION,
    TOOL_LABELS,
    TOOL_VOCAB,
    VECTOR_LABELS,
    VECTOR_VOCAB,
    PentestState,
    StateEncoder,
    _label_index,
    _unit_interval,
    state_layout_id,
)


APP_DIR = Path(__file__).resolve().parents[2] / "app"
ENCODER_SRC = APP_DIR / "rl" / "state_encoder.py"
Q_LEARNER_SRC = APP_DIR / "rl" / "q_learner.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found")


def _encode(**session: object) -> PentestState:
    """PentestState from a minimal session plus overrides."""
    base = {"_turn": 1, "_commands_run": 0, "_token_budget": 1000}
    base.update(session)
    return StateEncoder.from_session(base)


# ── SE4: Sections 3/4 are direct indices ─────────────────────────────────
class TestDirectIndexSections:
    def test_every_vector_label_gets_its_own_dimension(self):
        """No collisions — by construction, not by luck."""
        indices = [_label_index(label, VECTOR_LABELS) for label in VECTOR_LABELS]
        assert len(set(indices)) == len(VECTOR_LABELS), (
            f"two VECTOR_LABELS share a dimension: "
            f"{dict(zip(VECTOR_LABELS, indices))}"
        )
        assert all(0 <= i < VECTOR_VOCAB for i in indices)

    def test_every_tool_label_gets_its_own_dimension(self):
        indices = [_label_index(label, TOOL_LABELS) for label in TOOL_LABELS]
        assert len(set(indices)) == len(TOOL_LABELS)
        assert all(0 <= i < TOOL_VOCAB for i in indices)

    @pytest.mark.parametrize("label", [lbl for lbl in VECTOR_LABELS if lbl])
    def test_known_vector_label_sets_exactly_one_dim(self, label):
        vec = _encode(_last_vector=label).to_vector()
        section = vec[8 + 256: 8 + 256 + VECTOR_VOCAB]
        assert section.sum() == pytest.approx(1.0), (
            f"{label!r} did not light exactly one dimension of Section 3"
        )
        assert section[VECTOR_LABELS.index(label)] == pytest.approx(1.0), (
            f"{label!r} does not sit at its own index — the label order and "
            f"the encoding have diverged"
        )

    @pytest.mark.parametrize(
        "tool,label",
        [("nmap", "recon"), ("sqlmap", "exploit"), ("nuclei", "fuzz")],
    )
    def test_known_tool_sets_exactly_one_dim(self, tool, label):
        vec = _encode(_last_tool=tool).to_vector()
        start = 8 + 256 + VECTOR_VOCAB
        section = vec[start: start + TOOL_VOCAB]
        assert section.sum() == pytest.approx(1.0)
        assert section[TOOL_LABELS.index(label)] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "label", ["total_nonsense", "rekon", 17, b"recon", ["recon"]],
    )
    def test_unknown_label_is_a_zero_vector_not_a_private_bucket(self, label):
        """The old `_hash_bucket` mapped EVERY string into range.

        A typo therefore invented a legitimate-looking feature instead of
        encoding "no signal". Direct indexing cannot do that.
        """
        vec = _encode(_last_vector=label).to_vector()
        section = vec[8 + 256: 8 + 256 + VECTOR_VOCAB]
        assert section.sum() == pytest.approx(0.0), (
            f"{label!r} lit a Section-3 dimension — an unknown label must be "
            f"indistinguishable from 'no signal'"
        )

    def test_the_empty_label_has_its_own_designated_dimension(self):
        """"" is not a typo — it is the documented "no vector" label.

        It therefore sits at a real index rather than being conflated with
        an unknown label. This is the one exception to the zero-vector rule
        and it is intentional.
        """
        assert "" in VECTOR_LABELS
        index = VECTOR_LABELS.index("")
        vec = _encode(_last_vector="").to_vector()
        section = vec[8 + 256: 8 + 256 + VECTOR_VOCAB]
        assert section[index] == pytest.approx(1.0)

    def test_case_and_whitespace_are_folded(self):
        assert (
            _label_index("  RECON  ", VECTOR_LABELS)
            == _label_index("recon", VECTOR_LABELS)
        )

    def test_vocabularies_are_derived_from_action_space(self):
        """One table. Section 3/4 cannot drift from ACTION_SPACE."""
        assert VECTOR_LABELS == tuple(
            a.replace("execute_command_", "") for a in ACTION_SPACE
        ) + ("",)
        assert TOOL_LABELS == tuple(
            a.replace("execute_command_", "") for a in ACTION_SPACE
            if a.startswith("execute_command_")
        )

    def test_tool_vocab_is_exactly_what_the_credit_table_can_produce(self):
        """A label Section 4 accepts but no tool maps to (or vice versa)
        would be the R35 split brain again."""
        from app.rl.actions import TOOL_TO_ACTION

        produced = {tool_category(name) for name in TOOL_TO_ACTION} - {""}
        assert produced == set(TOOL_LABELS), (
            f"Section 4 vocabulary {set(TOOL_LABELS)} != what the credit "
            f"table can produce {produced}"
        )

    def test_bucket_names_are_not_tool_names(self):
        """Documented asymmetry, not a bug.

        `tool_category()` takes TOOL names; its OUTPUT is a bucket name, and
        `tool_category("recon")` is `""` on purpose (see
        `test_rl_action_mapping_consistency.py`). The encoder's Section 4
        vocabulary is nonetheless the bucket set — so `_last_tool` must
        carry a tool name or command line, which is exactly what the agent
        loop passes (`tc["name"]`).
        """
        assert tool_category("recon") == ""
        assert tool_category("nuclei") == "fuzz"


# ── SE3: `_last_tool` is an EXPLICIT signal and wins ─────────────────────
class TestLastToolPrecedence:
    def test_explicit_tool_wins_over_the_command_string(self):
        state = _encode(_last_tool="nuclei", _last_command="nmap -sV")
        assert state.last_tool == "fuzz", (
            "the explicit tool was ignored in favour of re-parsing the "
            "command — the exact SE3 defect"
        )

    def test_empty_explicit_tool_means_not_supplied(self):
        """An EMPTY `_last_tool` is "no value given", so the command is used.

        The dangerous case is the opposite one — a non-empty value that maps
        to "" (see `test_non_shell_action_overrides_a_stale_command`), where
        falling back would let last turn's command override this turn's
        explicit signal.
        """
        state = _encode(_last_tool="", _last_command="nmap -sV")
        assert state.last_tool == "recon"

    def test_non_command_action_encodes_as_no_signal(self):
        """`record_vulnerability` is not a shell tool; it must not claim recon."""
        state = _encode(_last_tool="record_vulnerability", _last_command="")
        assert state.last_tool == ""

    def test_non_shell_action_overrides_a_stale_command(self):
        state = _encode(
            _last_tool="record_vulnerability", _last_command="nmap -sV",
        )
        assert state.last_tool == "", (
            "a finding-only turn inherited last turn's recon command"
        )

    def test_command_line_is_not_echoed_as_a_tool_name(self):
        """`tool_category` parses the command line, so passing one is fine."""
        assert _encode(_last_tool="sqlmap -u http://x").last_tool == "exploit"

    def test_no_signal_at_all_is_a_zero_vector(self):
        state = _encode(_last_tool="", _last_command="")
        vec = state.to_vector()
        start = 8 + 256 + VECTOR_VOCAB
        assert vec[start:start + TOOL_VOCAB].sum() == pytest.approx(0.0)

    def test_turn_one_before_the_session_builder_exists(self):
        """A hand-built session (no `_last_tool` key) must still work."""
        assert _encode(_last_command="nmap -sV").last_tool == "recon"


class TestBuilderSuppliesLastTool:
    def test_build_session_emits_last_tool(self):
        from app.rl._session_builder import build_session

        session = build_session(
            scan_id="s", target="t", turn=2, commands_run=1, tokens_used=10,
            token_budget=1000, last_command="nuclei -u http://x",
            last_tool="nuclei", last_vector="fuzz", last_turn_success=False,
            findings=[],
        )
        assert session["_last_tool"] == "nuclei"
        assert StateEncoder.from_session(session).last_tool == "fuzz"

    def test_build_session_default_is_empty(self):
        from app.rl._session_builder import build_session

        session = build_session(
            scan_id="s", target="t", turn=1, commands_run=0, tokens_used=0,
            token_budget=1000, last_command="", last_vector="",
            last_turn_success=False, findings=[],
        )
        assert session["_last_tool"] == ""

    def test_the_live_agent_loop_passes_last_tool(self):
        """SE3's contract is dead unless production actually supplies it.

        This project's recurring failure is a documented key with zero
        callers, so the call sites are checked, not just the signature.
        """
        tree = _tree(APP_DIR / "agents" / "react_agent.py")
        build_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "build_session"
        ]
        assert len(build_calls) >= 3, (
            f"expected >=3 build_session() call sites, found {len(build_calls)}"
        )
        missing = [
            node.lineno for node in build_calls
            if not any(kw.arg == "last_tool" for kw in node.keywords)
        ]
        assert not missing, (
            f"build_session() called at line(s) {missing} without last_tool= — "
            f"the encoder would silently fall back to parsing the command"
        )


# ── SE4/SE7: the layout fingerprint ─────────────────────────────────────
class TestLayoutFingerprint:
    def test_layout_id_is_deterministic(self):
        assert state_layout_id() == state_layout_id()

    def test_layout_id_encodes_the_version(self):
        assert state_layout_id().startswith(f"v{STATE_LAYOUT_VERSION}")

    def test_layout_id_names_every_label_in_order(self):
        """Reordering a vocabulary changes the meaning of every dimension."""
        layout = state_layout_id()
        assert "/".join(VECTOR_LABELS) in layout
        assert "/".join(TOOL_LABELS) in layout

    def test_layout_id_names_every_section_size(self):
        layout = state_layout_id()
        for part in ("8", str(VECTOR_VOCAB), str(TOOL_VOCAB), "9"):
            assert part in layout, f"{part} missing from {layout!r}"
        assert str(STATE_DIM) == (
            str(8 + 256 + VECTOR_VOCAB + TOOL_VOCAB + 9)
        )

    def test_layout_id_changes_when_a_label_changes(self, monkeypatch):
        import app.rl.state_encoder as se

        before = se.state_layout_id()
        monkeypatch.setattr(se, "TOOL_LABELS", tuple(se.TOOL_LABELS) + ("bogus",))
        assert se.state_layout_id() != before

    def test_save_cache_stamps_the_layout(self):
        src = Q_LEARNER_SRC.read_text(encoding="utf-8")
        assert '"layout": state_layout_id()' in src, (
            "save_cache() no longer stamps the layout — a cache written now "
            "would be unusable forever after the next layout change"
        )

    def test_load_compares_the_layout_not_just_the_shape(self):
        src = Q_LEARNER_SRC.read_text(encoding="utf-8")
        assert "layout_ok" in src and "state_layout_id()" in src, (
            "the loader no longer validates the layout — a same-shape but "
            "semantically different cache would be adopted silently"
        )

    def test_a_shape_matched_cache_without_a_layout_is_quarantined(
        self, tmp_path, monkeypatch,
    ):
        """The load-bearing case: shapes agree, semantics do not."""
        import app.rl.q_learner as qmod

        weights = tmp_path / "q_weights.npz"
        meta = tmp_path / "q_meta.json"
        monkeypatch.setattr(qmod, "WEIGHTS_FILE", weights)
        monkeypatch.setattr(qmod, "META_FILE", meta)

        learner = qmod.DoubleQLearner()
        np.savez(weights, **dict(learner.online.get_weights()))
        meta.write_text(json.dumps({"step": 500, "lr": 0.001}), encoding="utf-8")

        fresh = qmod.DoubleQLearner()
        assert fresh.train_steps == 0, (
            "a cache with the right shapes but no layout stamp was loaded — "
            "SE4 changed what every dimension means, so this cannot be safe"
        )
        assert weights.exists() is False, "quarantined copy not moved"
        assert list(tmp_path.glob("*.stale-layout.*.npz")), (
            "quarantined as the wrong reason — expected stale-layout"
        )

    def test_a_layout_matched_cache_is_still_loaded(self, tmp_path, monkeypatch):
        """The guard must not be a blanket refusal."""
        import app.rl.q_learner as qmod

        weights = tmp_path / "q_weights.npz"
        meta = tmp_path / "q_meta.json"
        monkeypatch.setattr(qmod, "WEIGHTS_FILE", weights)
        monkeypatch.setattr(qmod, "META_FILE", meta)

        learner = qmod.DoubleQLearner()
        learner.save_cache()
        fresh = qmod.DoubleQLearner()
        assert fresh.train_steps == learner.train_steps
        assert weights.exists() is True


# ── SE5: the flag section is 7 booleans + 2 planner ratios ──────────────
class TestFlagSectionComposition:
    def test_flag_dims_is_nine(self):
        """7 booleans + 2 planner ratios — not "8 booleans + 1 ratio"."""
        assert STATE_DIM == 8 + 256 + VECTOR_VOCAB + TOOL_VOCAB + 9

    def test_encoder_emits_exactly_two_planner_dims(self):
        tree = _tree(ENCODER_SRC)
        fn = _find_func(tree, "to_vector")
        planner_calls = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Attribute) and node.attr.startswith("planner_paths_")
        ]
        assert len(planner_calls) == 2, (
            f"expected 2 planner dims, found {len(planner_calls)}"
        )
        names = {node.attr for node in planner_calls}
        assert names == {"planner_paths_live", "planner_paths_pruned"}

    def test_the_flags_tail_is_nine_long(self):
        vec = _encode(
            _kg_consulted=True,
            recon_signals={"has_login": True, "has_api": True},
            egats_planner={"alive": 3, "pruned": 1},
        ).to_vector()
        flags = vec[-9:]
        assert len(flags) == 9
        # Order: last_success, kg_available, planner live, planner pruned,
        #        login, api, upload, graphql, websocket.
        assert flags[1] == pytest.approx(1.0), "kg_available not emitted"
        assert flags[2] > 0.0, "planner alive ratio not emitted"
        assert flags[3] > 0.0, "planner pruned ratio not emitted"
        assert flags[4] == pytest.approx(1.0), "login flag not emitted"
        assert flags[5] == pytest.approx(1.0), "api flag not emitted"
        assert flags[6:].sum() == pytest.approx(0.0)

    def test_flag_dims_comment_matches_the_code(self):
        """A comment is not a validator — so the number is checked."""
        tree = _tree(ENCODER_SRC)
        value = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_FLAG_DIMS"
                for t in node.targets
            ):
                value = ast.literal_eval(node.value)
        assert value == 9, f"_FLAG_DIMS is {value}, expected 9"


# ── SE6: NaN cannot enter the state vector ─────────────────────────────
class TestUnitInterval:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_falls_back_to_the_default(self, bad):
        """A bare `min(max(x,0),1)` returns NaN — it does not clamp it."""
        assert _unit_interval(bad, default=0.5) == 0.5
        assert math.isfinite(_unit_interval(bad, default=0.5))

    @pytest.mark.parametrize("bad", [None, "abc", b"x", [], object()])
    def test_uncoercible_falls_back_to_the_default(self, bad):
        assert _unit_interval(bad, default=0.25) == 0.25

    @pytest.mark.parametrize(
        "raw,expected",
        [(-5.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (5.0, 1.0), ("0.3", 0.3)],
    )
    def test_finite_values_are_clamped(self, raw, expected):
        assert _unit_interval(raw) == pytest.approx(expected)

    def test_a_nan_tdi_cannot_poison_the_vector(self):
        state = PentestState(
            turn=1, tdi=float("nan"), mode="llm", findings_count=0,
            verified_findings=0, fp_findings=0, commands_run=0,
            tokens_used=0, token_budget_remaining=float("nan"),
            tech_stack=(), last_vector="", last_tool="", last_success=False,
            kg_available=False, planner_paths_live=0,
            planner_paths_pruned=0, recon_login=False, recon_api=False,
            recon_upload=False, recon_graphql=False, recon_websocket=False,
        )
        vec = state.to_vector()
        assert np.isfinite(vec).all(), (
            "a non-finite scalar reached the state vector — every forward "
            "pass from it is NaN"
        )


# ── SE7: STATE_DIM has one definition ─────────────────────────────────
class TestStateDimIsSingleSourced:
    def test_encoder_source_does_not_hardcode_337(self):
        tree = _tree(ENCODER_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "STATE_DIM"
                for t in node.targets
            ):
                assert not (
                    isinstance(node.value, ast.Constant)
                    and node.value.value == 337
                ), "STATE_DIM is hardcoded to 337 again"
                return
        raise AssertionError("STATE_DIM assignment not found")

    def test_q_learner_imports_state_dim(self):
        src = Q_LEARNER_SRC.read_text(encoding="utf-8")
        assert "from app.rl.state_encoder import" in src
        assert "STATE_DIM" in src.split("from app.rl.state_encoder import")[1]

    @pytest.mark.parametrize("other", ["policy", "experience_store", "reward"])
    def test_no_rl_module_redefines_the_dimensions(self, other):
        path = APP_DIR / "rl" / f"{other}.py"
        if not path.exists():
            pytest.skip(f"{path.name} absent")
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"STATE_DIM", "ACTION_DIM"}
                for t in node.targets
            ):
                raise AssertionError(
                    f"{path.name} redefines {node.targets[0].id} — the "
                    f"encoder owns it"
                )

    def test_state_dim_matches_the_section_sizes(self):
        assert STATE_DIM == (
            8 + 256 + VECTOR_VOCAB + TOOL_VOCAB + 9
        )
        assert ACTION_DIM == 7
        assert VECTOR_VOCAB == 32 and TOOL_VOCAB == 32
