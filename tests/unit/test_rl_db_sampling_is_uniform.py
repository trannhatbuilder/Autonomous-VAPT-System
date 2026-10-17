"""Guards: `sample_from_db` must sample uniformly, and say so.

Background — the bug these guard against
----------------------------------------
`ExperienceStore.sample_from_db()` documented itself as:

    "Sample uniformly (or by scan_id) from Postgres for offline training.
     ...
     is_weight is set to 1.0 (uniform sampling — no IS correction)."

while executing:

    stmt.order_by(RLTransition.priority.desc()).limit(batch_size)

Two things were wrong at once:

1. **Not uniform, and not even varying.** `LIMIT n` over a deterministic
   ORDER BY returns *the same n rows* on every call. Repeated offline
   training therefore overfit one fixed slice of `vapt_rl_transitions`.
2. **Not "the high-TD-error rows" either** — which is the part a reader
   naturally assumes. `ExperienceStore.persist_priorities()` is a
   documented no-op stub (`return 0`) deferred to W17+, so TD errors never
   reach the DB. The stored `priority` is whatever was written at INSERT
   time: `SumTree._max_priority` (a running in-memory maximum that every
   later row inherits) or `abs(rl_reward) * PER_PRIORITY_MULTIPLIER` for
   self-correction rejections. `ORDER BY priority DESC` was therefore
   selecting a priority-coincident accident, not a meaningful bias.

Because `is_weight` was hard-coded to 1.0 and `PentestPolicy.train_from_db`
passes no `is_weights`, the (nonexistent) bias was also uncorrected.

`training_loop.train_from_scan()`'s docstring was *honest* about the old
behaviour ("ordered by priority DESC") — so the codebase held two
contradicting docstrings, and the one sitting next to the SQL was the liar.
Hence the docstring-consistency guards below: the class of defect is
"prose and SQL disagree", not "one ORDER BY clause is wrong".
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
import types
from typing import Any

import numpy as np

from app.rl.experience_store import ExperienceStore
from app.rl import training_loop
from app.rl.policy import PentestPolicy

STORE_SRC = inspect.getsource(ExperienceStore)
SAMPLE_DB_SRC = textwrap.dedent(inspect.getsource(ExperienceStore.sample_from_db))
SAMPLE_DB_DOC = textwrap.dedent(
    inspect.getdoc(ExperienceStore.sample_from_db) or ""
)
TRAIN_FROM_SCAN_DOC = textwrap.dedent(
    inspect.getdoc(training_loop.train_from_scan) or ""
)


# ── source-level: the SQL ─────────────────────────────────────────────


class TestNoPriorityOrdering:
    def test_sample_from_db_does_not_order_by_priority(self):
        assert "priority.desc()" not in SAMPLE_DB_SRC, (
            "sample_from_db orders by priority again — it is then neither "
            "uniform (same rows every run) nor IS-corrected"
        )

    def test_sample_from_db_orders_by_random(self):
        assert "func.random()" in SAMPLE_DB_SRC, (
            "sample_from_db no longer randomises the ordering; whichever "
            "ordering replaced it must be justified in the docstring"
        )

    def test_func_is_imported_where_it_is_used(self):
        """`func.random()` with no `func` import is a NameError at runtime."""
        assert re.search(
            r"from sqlalchemy import[^\n]*\bfunc\b", SAMPLE_DB_SRC
        ), "func.random() is used but `func` is not imported inside the function"

    def test_order_by_argument_is_a_random_call(self):
        """AST, not text: proves the ordering *expression* is random()."""
        tree = ast.parse(SAMPLE_DB_SRC)
        order_bys: list[ast.Call] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "order_by"
            ):
                order_bys.extend(
                    a for a in node.args if isinstance(a, ast.Call)
                )
        assert order_bys, "no order_by(...) call expression found"
        assert any(
            isinstance(c.func, ast.Attribute) and c.func.attr == "random"
            for c in order_bys
        ), f"order_by argument is not random(): {[ast.dump(c) for c in order_bys]}"


# ── docstring / SQL agreement (the defect CLASS) ──────────────────────


class TestDocstringAgreesWithSql:
    def test_uniform_claim_is_backed_by_random_ordering(self):
        """If the docstring says uniform, the SQL must randomise.

        This is the guard that would have caught the original bug: the
        docstring said "uniform" and the SQL did not.
        """
        claims_uniform = "uniform" in SAMPLE_DB_DOC.lower()
        assert claims_uniform, "sample_from_db stopped claiming uniform sampling"
        assert "func.random()" in SAMPLE_DB_SRC, (
            f"docstring claims uniform sampling but the SQL does not randomise:\n"
            f"{SAMPLE_DB_SRC}"
        )

    def test_docstring_explains_the_importance_weight(self):
        """`is_weight = 1.0` is only honest for uniform sampling — say why."""
        assert "is_weight" in SAMPLE_DB_DOC
        assert "IS" in SAMPLE_DB_DOC or "importance" in SAMPLE_DB_DOC.lower()

    def test_is_weight_literal_is_one(self):
        assert re.search(
            r'"is_weight":\s*1\.0', SAMPLE_DB_SRC
        ), "is_weight is no longer 1.0 — uniform sampling no longer holds"

    def test_training_loop_docstring_does_not_claim_priority_ordering(self):
        """The docstring must now describe what the SQL actually does.

        Note what this does *not* do: it does not forbid the words
        "priority DESC". Explaining the bug requires quoting it, and a guard
        that bans the substring forces you to delete the explanation to get
        green — which is exactly how the original contradiction survived.
        The positive property (docstring says uniform) plus the AST guard
        below (code does not sort by priority) is the honest pair.
        """
        assert "uniform" in TRAIN_FROM_SCAN_DOC.lower(), (
            "training_loop.train_from_scan should state that sampling is uniform"
        )

    def test_uniform_claim_in_training_loop_is_backed_by_sql(self):
        assert "uniform" in TRAIN_FROM_SCAN_DOC.lower()
        assert "func.random()" in SAMPLE_DB_SRC, (
            "train_from_scan claims uniform sampling but sample_from_db does "
            "not randomise its ordering"
        )

    def test_sample_db_docstring_explains_why_uniform(self):
        lowered = SAMPLE_DB_DOC.lower()
        assert "uniform" in lowered
        # The docstring must carry the reason, not just the adjective: the
        # stored priority column is not TD-derived, so "priority DESC" was
        # never selecting what a reader would assume.
        assert "persist_priorities" in SAMPLE_DB_DOC or "td error" in lowered

    def test_no_module_orders_by_priority_in_code(self):
        """AST guard over the DB path: no `order_by(<...>.priority.desc())`.

        Deliberately semantic rather than textual — prose that quotes the old
        bug is documentation, not a defect.
        """
        for mod in (ExperienceStore, training_loop, PentestPolicy):
            tree = ast.parse(inspect.getsource(mod))
            offenders = []
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "order_by"
                ):
                    continue
                for arg in node.args:
                    if "priority" in ast.dump(arg) and "desc" in ast.dump(arg):
                        offenders.append(ast.dump(arg)[:120])
            assert not offenders, (
                f"{mod.__name__} orders a DB query by priority DESC: {offenders}"
            )


# ── behavioural: the actual statement, and the returned batch ─────────


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return list(self._rows)


class _CapturingSession:
    """Stands in for AsyncSession; records the statement it is handed.

    No DB required — `sample_from_db(session=...)` never opens its own.
    """

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.rows = rows or []
        self.statements: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.statements.append(stmt)
        return _FakeResult(self.rows)


def _row(turn: int = 0, priority: float = 1.0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        scan_id="scan-x",
        turn=turn,
        state_vector=[0.0] * 4,
        action=1,
        reward=0.5,
        next_state_vector=[1.0] * 4,
        done=False,
        priority=priority,
        action_name="execute_command_recon",
        reward_breakdown={"total": 0.5},
    )


class TestCompiledStatement:
    async def test_statement_is_random_ordered(self):
        sess = _CapturingSession([_row()])
        store = ExperienceStore()
        await store.sample_from_db(8, session=sess)

        assert sess.statements, "no statement executed"
        compiled = str(sess.statements[0].compile(
            compile_kwargs={"literal_binds": True}
        )).lower()
        assert "random()" in compiled, f"compiled SQL is not random-ordered:\n{compiled}"
        assert "priority desc" not in compiled, (
            f"compiled SQL still sorts by priority:\n{compiled}"
        )

    async def test_scan_id_still_filters(self):
        sess = _CapturingSession([])
        store = ExperienceStore()
        await store.sample_from_db(8, scan_id="scan-x", session=sess)
        compiled = str(sess.statements[0].compile(
            compile_kwargs={"literal_binds": True}
        )).lower()
        assert "scan_id" in compiled
        assert "random()" in compiled


class TestReturnedBatchContract:
    async def test_is_weight_is_one_for_every_sample(self):
        rows = [_row(turn=i, priority=float(10 - i)) for i in range(5)]
        sess = _CapturingSession(rows)
        store = ExperienceStore()
        batch = await store.sample_from_db(5, session=sess)

        assert len(batch) == 5
        assert all(b["is_weight"] == 1.0 for b in batch), (
            "uniform sampling must carry is_weight == 1.0; anything else is "
            "an IS correction that was never computed"
        )

    async def test_priority_does_not_influence_the_returned_order(self):
        """Rows are returned in DB order regardless of their priority column.

        With `ORDER BY priority DESC` the highest-priority row always came
        first; now the store must not re-sort client-side either.
        """
        rows = [_row(turn=0, priority=1.0), _row(turn=1, priority=99.0)]
        sess = _CapturingSession(rows)
        store = ExperienceStore()
        batch = await store.sample_from_db(2, session=sess)
        assert [b["turn"] for b in batch] == [0, 1], (
            "sample_from_db re-sorted by priority client-side"
        )

    async def test_vectors_are_float32_arrays_of_the_right_dim(self):
        sess = _CapturingSession([_row()])
        store = ExperienceStore()
        batch = await store.sample_from_db(1, session=sess)
        assert batch[0]["state"].dtype == np.float32
        assert batch[0]["next_state"].dtype == np.float32
        assert batch[0]["state"].shape == (4,)

    async def test_empty_result_is_an_empty_list(self):
        sess = _CapturingSession([])
        store = ExperienceStore()
        assert await store.sample_from_db(4, session=sess) == []


# ── the dropped `session` argument ────────────────────────────────────


class TestSessionIsForwarded:
    def test_sample_from_db_accepts_session(self):
        params = inspect.signature(ExperienceStore.sample_from_db).parameters
        assert "session" in params

    def test_train_from_db_accepts_and_forwards_session(self):
        params = inspect.signature(PentestPolicy.train_from_db).parameters
        assert "session" in params, (
            "train_from_db cannot receive the session that "
            "training_loop.train_from_scan has always passed"
        )
        src = inspect.getsource(PentestPolicy.train_from_db)
        assert "session=session" in src, "train_from_db drops the session"

    def test_train_from_scan_forwards_its_session(self):
        src = inspect.getsource(training_loop.train_from_scan)
        params = inspect.signature(training_loop.train_from_scan).parameters
        assert "session" in params
        assert "session=session" in src, (
            "train_from_scan accepts `session` and silently ignores it — a "
            "caller with uncommitted rows reads an empty table and gets no error"
        )

    async def test_caller_owned_session_is_not_closed(self):
        """sample_from_db must not take ownership of a session it was given."""
        closed = {"v": False}

        class _Owned(_CapturingSession):
            async def __aexit__(self, *exc: object) -> None:  # pragma: no cover
                closed["v"] = True

        sess = _Owned([_row()])
        store = ExperienceStore()
        batch = await store.sample_from_db(1, session=sess)
        assert len(batch) == 1
        assert closed["v"] is False, (
            "sample_from_db entered `async with` on a caller-owned session"
        )

    async def test_no_session_opens_its_own(self, monkeypatch):
        """Without a session, the store must still work on its own."""
        import contextlib

        captured: list[Any] = []

        class _CM:
            def __init__(self) -> None:
                self.inner = _CapturingSession([_row(turn=7)])

            async def __aenter__(self) -> _CapturingSession:
                captured.append(self.inner)
                return self.inner

            async def __aexit__(self, *exc: object) -> None:
                return None

        import app.db.session as db_session

        monkeypatch.setattr(db_session, "async_session", lambda: _CM())
        store = ExperienceStore()
        batch = await store.sample_from_db(1)
        assert [b["turn"] for b in batch] == [7]
        assert captured, "no session was opened when none was supplied"
        assert "random()" in str(
            captured[0].statements[0].compile(
                compile_kwargs={"literal_binds": True}
            )
        ).lower()
