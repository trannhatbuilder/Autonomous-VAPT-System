"""Shared test configuration.

R27 / PL4 — isolate the RL artifact directory.

`HarnessBridge` persists real files under `settings.rl_dir` (`data/rl/`):
`q_weights.npz`, `q_meta.json` and — since R27 — `policy_state.json`, which
carries ε / temperature / the episode counter. Any test that constructs a
`HarnessBridge` and calls `end_scan()` writes those files.

That was harmless while only weights were written (they are opaque), but
`policy_state.json` is *behavioural*: the suite ended up leaving
`{"epsilon": 0.05, "episode": 9, ...}` in the repo, so a real scan started
with 9 episodes of fabricated history and ε pinned near its floor. Test
runs were silently rewriting production RL state.

This autouse fixture redirects the three paths into a per-test tmp dir so
the repo copy is never touched. Tests that need their own layout still
override the paths themselves — this only changes the default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_rl_artifacts(tmp_path, monkeypatch):
    """Point Q weights / meta / policy state at a throwaway directory."""
    import app.rl.q_learner as qmod

    monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
    monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")
    monkeypatch.setattr(qmod, "META_DIR", tmp_path, raising=False)
    yield
