"""baseline schema snapshot

Mirrors `routes/deps/database.py:init_db()` plus the legacy SQL files in
`migrations/`. Everything is `CREATE TABLE IF NOT EXISTS` so this migration is
safe to apply on databases that were previously bootstrapped by `init_db()`.

Revision ID: 0001
Revises:
Create Date: 2026-04-21
"""
from __future__ import annotations

from pathlib import Path

from alembic import op


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


_LEGACY_FILES = (
    "001_shield_engine_phase1.sql",
    "002_schema_init.sql",
    "003_greybox_scan.sql",
    "004_compliance_attack_graph.sql",
)

_CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    plan TEXT NOT NULL DEFAULT 'free',
    is_active BOOLEAN NOT NULL DEFAULT true,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    target_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    progress INT DEFAULT 0,
    scan_type TEXT DEFAULT 'standard',
    preview_json JSONB,
    result_json JSONB,
    error TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    risk_score INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Shield-Engine Core Tables
CREATE TABLE IF NOT EXISTS agent_states (
    agent_name    VARCHAR(255) PRIMARY KEY,
    state_data    JSONB NOT NULL DEFAULT '{}',
    status        VARCHAR(50) NOT NULL DEFAULT 'running',
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shield_metrics (
    id            SERIAL PRIMARY KEY,
    scan_id       VARCHAR(255) NOT NULL,
    metric_type   VARCHAR(100) NOT NULL,
    metric_value  FLOAT NOT NULL DEFAULT 0,
    metadata      JSONB DEFAULT '{}',
    recorded_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shield_metrics_scan ON shield_metrics(scan_id);
CREATE INDEX IF NOT EXISTS idx_shield_metrics_type ON shield_metrics(metric_type);

CREATE TABLE IF NOT EXISTS scan_feedback (
    id            SERIAL PRIMARY KEY,
    scan_id       VARCHAR(255) NOT NULL,
    finding_hash  VARCHAR(128) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,
    comment       TEXT DEFAULT '',
    finding_data  JSONB,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(scan_id, finding_hash)
);
CREATE INDEX IF NOT EXISTS idx_scan_feedback_type ON scan_feedback(feedback_type);

CREATE TABLE IF NOT EXISTS detection_patterns (
    pattern_hash  VARCHAR(64) PRIMARY KEY,
    pattern_name  VARCHAR(255) NOT NULL,
    confidence    FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    true_positives  INT NOT NULL DEFAULT 0,
    false_positives INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm_models (
    id            SERIAL PRIMARY KEY,
    provider      VARCHAR(50) NOT NULL,
    model_name    VARCHAR(255) NOT NULL,
    display_name  VARCHAR(255),
    is_active     BOOLEAN NOT NULL DEFAULT false,
    api_key_enc   TEXT,
    config        JSONB DEFAULT '{}',
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_models_active ON llm_models(provider, model_name);

-- Session management table (Fixes the 'relation sessions does not exist' error)
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    fingerprint_hash TEXT,
    user_agent TEXT,
    refresh_token_hash TEXT,
    ip_address TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- Engagement management
CREATE TABLE IF NOT EXISTS engagements (
    id VARCHAR(64) PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    client_name VARCHAR(255),
    type VARCHAR(50) NOT NULL DEFAULT 'internal',
    scope_rules TEXT,
    roe TEXT,
    start_date DATE,
    end_date DATE,
    status VARCHAR(50) NOT NULL DEFAULT 'planning',
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_engagements_created_by ON engagements(created_by);
CREATE INDEX IF NOT EXISTS idx_engagements_status ON engagements(status);

-- Add engagement_id to scans table
ALTER TABLE scans ADD COLUMN IF NOT EXISTS engagement_id VARCHAR(64);
CREATE INDEX IF NOT EXISTS idx_scans_engagement ON scans(engagement_id);

-- Findings database
CREATE TABLE IF NOT EXISTS findings (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(64) NOT NULL,
    engagement_id VARCHAR(64),
    name VARCHAR(512) NOT NULL,
    description TEXT,
    vuln_type VARCHAR(128),
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    cvss_score FLOAT,
    location TEXT,
    poc TEXT,
    verification_command TEXT,
    verification_output TEXT,
    verification_method VARCHAR(64),
    verifier_token VARCHAR(64),
    verification_confidence FLOAT,
    remediation TEXT,
    cwe_id VARCHAR(32),
    cve_id VARCHAR(32),
    tool_used VARCHAR(128),
    verified BOOLEAN NOT NULL DEFAULT false,
    false_positive BOOLEAN NOT NULL DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id);
CREATE INDEX IF NOT EXISTS idx_findings_engagement ON findings(engagement_id);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_findings_vuln_type ON findings(vuln_type);
CREATE INDEX IF NOT EXISTS idx_findings_cve ON findings(cve_id);
CREATE INDEX IF NOT EXISTS idx_findings_tool ON findings(tool_used);
CREATE INDEX IF NOT EXISTS idx_findings_verified ON findings(verified);

-- AI Pentest Attack Graphs
CREATE TABLE IF NOT EXISTS attack_graphs (
    scan_id VARCHAR(64) PRIMARY KEY,
    graph_json JSONB NOT NULL DEFAULT '{}',
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scan_quota_plans (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    scans_per_month INT NOT NULL DEFAULT 0,
    price_usd FLOAT NOT NULL DEFAULT 0,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Findings summary view
CREATE OR REPLACE VIEW findings_summary AS
SELECT
    engagement_id,
    COUNT(*) AS total_findings,
    COUNT(*) FILTER (WHERE severity = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE severity = 'high') AS high_count,
    COUNT(*) FILTER (WHERE severity = 'medium') AS medium_count,
    COUNT(*) FILTER (WHERE severity = 'low') AS low_count,
    COUNT(*) FILTER (WHERE verified = true) AS verified_count,
    COUNT(*) FILTER (WHERE false_positive = true) AS fp_count,
    MAX(created_at) AS last_finding_at
FROM findings
GROUP BY engagement_id;
"""



def _legacy_sql_dir() -> Path:
    # alembic/versions/0001_*.py -> ../../migrations
    return Path(__file__).resolve().parents[2] / "migrations"


def upgrade() -> None:
    op.execute(_CORE_SCHEMA)


def downgrade() -> None:
    # Baseline snapshot — no destructive downgrade.
    raise RuntimeError("Cannot downgrade past baseline 0001")
