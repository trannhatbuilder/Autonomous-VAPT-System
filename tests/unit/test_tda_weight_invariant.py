"""Round 9 (design note 1) — the TDI weight-sum invariant.

``MODEL_WEIGHTS_SUM`` existed but nothing enforced it: retuning one of
``W_H``/``W_E``/``W_C``/``W_S`` shifted TDI's range away from [0, 1] while
``TDI_BROADEN = 0.60`` kept being compared against it — silently. The guard
now lives at module import, so a bad weights edit fails at load instead of
40 minutes into a 100-turn scan.

The enforcement test does not just re-do the arithmetic in the test file
(that would pass even if the guard were deleted). It re-executes the REAL
module source with one weight tampered, which is the only way to prove the
``raise`` is actually wired.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

import app.rl.tda as tda

TDA_FILE = Path(tda.__file__).resolve()
_TAMPERED_NAME = "app.rl._tda_tampered_for_tests"


def _exec_tampered_source(**replacements: tuple[str, str]) -> dict:
    """Execute tda.py's real source with literal weights swapped out.

    Raises whatever the module raises at import time — which is the point.

    The namespace is registered in ``sys.modules`` for the duration of the
    exec because ``@dataclass`` resolves ``from __future__ import
    annotations`` field types through ``sys.modules[cls.__module__]``; an
    unregistered module name would fail with an AttributeError that has
    nothing to do with the weight guard.
    """
    source = TDA_FILE.read_text(encoding="utf-8")
    for original, replacement in replacements.values():
        assert original in source, f"could not find {original!r} in {TDA_FILE}"
        source = source.replace(original, replacement, 1)

    module = types.ModuleType(_TAMPERED_NAME)
    module.__file__ = str(TDA_FILE)
    previous = sys.modules.get(_TAMPERED_NAME)
    sys.modules[_TAMPERED_NAME] = module
    try:
        exec(compile(source, str(TDA_FILE), "exec"), module.__dict__)  # noqa: S102
    finally:
        if previous is None:
            sys.modules.pop(_TAMPERED_NAME, None)
        else:
            sys.modules[_TAMPERED_NAME] = previous
    return module.__dict__


class TestWeightsSumToOne:
    def test_shipped_weights_sum_to_one(self):
        assert tda.MODEL_WEIGHTS_SUM == pytest.approx(1.0)

    def test_sum_is_computed_from_the_parts(self):
        assert tda.MODEL_WEIGHTS_SUM == pytest.approx(
            tda.W_H + tda.W_E + tda.W_C + tda.W_S
        )

    def test_broaden_threshold_is_inside_the_unit_range(self):
        assert 0.0 < tda.TDI_BROADEN < 1.0


class TestGuardActuallyFires:
    def test_tampered_weight_raises_at_import(self):
        with pytest.raises(RuntimeError, match=r"must sum to 1\.0"):
            _exec_tampered_source(W_H=("W_H: float = 0.30", "W_H: float = 0.50"))

    def test_the_message_names_the_broken_sum(self):
        with pytest.raises(RuntimeError, match=r"got 1\.2"):
            _exec_tampered_source(W_H=("W_H: float = 0.30", "W_H: float = 0.50"))

    def test_a_compensating_edit_still_imports(self):
        """Rebalancing two weights so the sum holds must NOT trip the guard."""
        namespace = _exec_tampered_source(
            W_H=("W_H: float = 0.30", "W_H: float = 0.40"),
            W_S=("W_S: float = 0.20", "W_S: float = 0.10"),
        )
        assert namespace["MODEL_WEIGHTS_SUM"] == pytest.approx(1.0)

    def test_float_drift_does_not_trip_the_guard(self):
        """The comparison uses a 1e-9 tolerance, not ==.

        0.1 + 0.2 != 0.3 in binary float; a sum built from thirds must not
        explode at import.
        """
        namespace = _exec_tampered_source(
            W_H=("W_H: float = 0.30", "W_H: float = 0.1"),
            W_E=("W_E: float = 0.30", "W_E: float = 0.2"),
            W_C=("W_C: float = 0.20", "W_C: float = 0.3"),
            W_S=("W_S: float = 0.20", "W_S: float = 0.4"),
        )
        assert namespace["MODEL_WEIGHTS_SUM"] == pytest.approx(1.0)


class TestGuardIsStructurallyPresent:
    """Source guard: the runtime tests above would be skipped if the
    `if ...: raise RuntimeError` block were deleted, because there would be
    nothing left to trip. Assert the AST shape so a deletion is caught."""

    def test_module_level_if_raising_on_the_sum(self):
        tree = ast.parse(TDA_FILE.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            mentioned = {
                n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
            }
            if "MODEL_WEIGHTS_SUM" not in mentioned:
                continue
            raises = [
                n for n in ast.walk(node) if isinstance(n, ast.Raise)
            ]
            if not raises:
                continue
            exc = raises[0].exc
            name = (
                exc.func.id
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name)
                else None
            )
            assert name == "RuntimeError", (
                f"the weight-sum guard raises {name!r}, expected RuntimeError"
            )
            return
        raise AssertionError(
            "no module-level `if ... MODEL_WEIGHTS_SUM ...: raise RuntimeError` "
            "found in app/rl/tda.py — the invariant is unenforced again"
        )
