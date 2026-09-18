---
name: disclosure-workflow
description: ISO 29147 disclosure timeline management
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["disclosure", "iso-29147", "timeline"]
  triggers: ["disclosure", "iso 29147", "timeline"]
allowed-tools: ""
mapped-agents: ["reporting-remediation"]
---

# disclosure-workflow

## Overview

ISO 29147 vulnerability disclosure timeline management — per-severity windows (Critical 7d, High 30d, Medium 90d, Low 180d) + notification emails + retest flow.

## When to Use

At finding creation + on retest. Loaded by reporting-remediation agent.

**Mapped agents**: reporting-remediation

## Methodology

- 1. On finding creation: auto-start disclosure timer per severity
- 2. Severity windows: Critical 7d, High 30d, Medium 90d, Low 180d
- 3. Notify user X days before window expires (email)
- 4. On retest request: re-run PoC, update finding status
- 5. Status flow: pending → scheduled → running → verified / still_vulnerable / fixed
- 6. After fix verified: close finding + log audit entry

## Tool Recipes

```bash
POST /api/findings/{id}/retest (W22 — retest endpoint)
```
```bash
app/core/disclosure.py (W22 — disclosure module)
```

## Output Standards

Each finding: disclosed_at, expires_at, notified (bool), retest_status. Email notifications sent via aiosmtplib. ISO 29147 compliance.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/disclosure-workflow` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
