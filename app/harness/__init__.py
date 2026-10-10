"""
VAPT-AI Harness Package — Phase 3 (slimmed down).

Previously contained EvidenceAuditor, Verifier, PoCValidator, ConfidenceScorer,
CVSSValidator, HarnessBridge, etc. — all removed in Phase 3 overhaul.

Now only contains:
    - ansi.py: ANSI escape code stripper (used by tool_bridge + events)

The single ReAct loop (app/agents/react_agent.py) auto-verifies findings with
evidence — no second LLM auditor needed (CyberStrikeAI pattern).
"""
from app.harness.ansi import strip_ansi

__all__ = ["strip_ansi"]