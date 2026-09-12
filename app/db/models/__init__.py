"""VAPT-AI SQLAlchemy 2.0 models — W1-C + W1-D complete.

Importing this package registers ALL model classes with Base.metadata so
Alembic autogenerate (W1-E) detects them.

Tables created (W1-C, 18 tables):
    vapt_users                     — single-user auth (admin)
    vapt_refresh_tokens            — JWT refresh tokens
    vapt_scans                     — scan lifecycle
    vapt_consent_forms             — D19 consent + scope declaration
    vapt_assets                    — discovered assets
    vapt_pentest_facts             — blackboard shared state
    vapt_findings                  — verified vulnerability findings
    vapt_evidence                  — 5-layer evidence chain (D23 custody)
    vapt_poc_results               — PoC results
    vapt_c2_sessions               — unified C2 sessions (D28)
    vapt_c2_tasks                  — C2 tasks (L1-L5, HITL for L3+)
    vapt_hitl_approvals            — HITL approval gate (D25)
    vapt_attack_chains             — attack path graph edges
    vapt_audit_log                 — append-only HMAC-sealed audit log (D23)
    vapt_quarantine                — quarantined dangerous output
    vapt_retest_requests           — "I've Fixed This" retest flow
    vapt_wstg_methodology_catalog  — OWASP WSTG v4.2 catalog
    vapt_skill_executions          — skill invocation log

Tables created (W1-D, 6 tables):
    vapt_rl_checkpoints            — RL model checkpoints (Dueling Double DQN weights)
    vapt_rl_transitions            — RL experience tuples (PER SumTree source)
    vapt_kg_nodes                  — Knowledge Graph nodes (6 types)
    vapt_kg_edges                  — Knowledge Graph edges (5 types + outcome probs)
    vapt_kg_scan_snapshots         — per-scan KG fork snapshots
    vapt_replay_traces             — replay trace metadata (JSONL on disk)

Total: 24 VAPT-AI tables.

Auxiliary tables (deferred to W2+):
    approval_policies              — W7 (HITL auto-approval rules)
    disclosure_windows             — W22 (ISO 29147 disclosure tracking)
    system_health_reports          — W22 (zero-finding scan reports)
    compliance_mappings            — W20 (PCI-DSS / OWASP ASVS mapping)
    c2_listeners                   — W14-W15 (C2 listener configs)
    c2_payloads                    — W14-W15 (generated beacon payloads)

Naming convention:
    All VAPT-AI tables prefixed with `vapt_` to avoid collision with legacy
    EVVO tables (users, scans, findings, ...) which are still in use by 482
    tests and will be migrated in later weeks.
"""
# W1-C — Pentest lifecycle (18 tables)
from app.db.models.user import RefreshToken, User
from app.db.models.scan import Asset, ConsentForm, Scan
from app.db.models.pentest import Evidence, Finding, PentestFact, PoCResult
from app.db.models.c2 import C2Session, C2Task
from app.db.models.hitl import HITLApproval
from app.db.models.audit import AuditLog, Quarantine
from app.db.models.attackchain import AttackChain, RetestRequest
from app.db.models.methodology import SkillExecution, WSTGMethodologyCatalog

# W1-D — RL + KG + Replay (6 tables)
from app.db.models.kg import KGEdge, KGNode, KGScanSnapshot
from app.db.models.replay import ReplayTrace
from app.db.models.rl import RLCheckpoint, RLTransition

__all__ = [
    # User
    "User",
    "RefreshToken",
    # Scan
    "Scan",
    "ConsentForm",
    "Asset",
    # Pentest
    "PentestFact",
    "Finding",
    "Evidence",
    "PoCResult",
    # C2
    "C2Session",
    "C2Task",
    # HITL
    "HITLApproval",
    # Audit
    "AuditLog",
    "Quarantine",
    # Attack chain
    "AttackChain",
    "RetestRequest",
    # Methodology
    "WSTGMethodologyCatalog",
    "SkillExecution",
    # RL
    "RLCheckpoint",
    "RLTransition",
    # KG
    "KGNode",
    "KGEdge",
    "KGScanSnapshot",
    # Replay
    "ReplayTrace",
]