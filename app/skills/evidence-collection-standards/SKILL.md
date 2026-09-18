---
name: evidence-collection-standards
description: Evidence chain of custody — HMAC seals, 5-layer chain
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["evidence", "custody", "hmac", "chain"]
  triggers: ["evidence", "custody", "hmac", "tamper"]
allowed-tools: ""
mapped-agents: ["reporting-remediation"]
---

# evidence-collection-standards

## Overview

Evidence chain of custody — 5-layer chain (Detection, Validation, Exploitation, Post-Exploitation, Audit) + HMAC-SHA256 tamper seals per evidence row.

## When to Use

Whenever evidence is captured. Loaded by reporting-remediation agent (also used implicitly by destructive agents).

**Mapped agents**: reporting-remediation

## Methodology

- 1. Capture evidence at each layer (detection → validation → exploitation → post-exploit → audit)
- 2. Each evidence row gets HMAC-SHA256 tamper seal (custody_seal field)
- 3. Calculate evidence_hash (SHA-256 of raw_output)
- 4. Custody verifier runs on every Evidence read
- 5. PII redaction BEFORE storage (app/pii/redactor.py)
- 6. Quarantine dangerous output (raw shell output, command output)

## Tool Recipes

```bash
Evidence.add_evidence(finding_id, layer='detection', raw_output=redacted_output, tool_used='nmap')
```
```bash
CustodyVerifier.verify_evidence(evidence_id)
```
```bash
CustodyVerifier.verify_finding_chain(finding_id)
```

## Output Standards

Evidence row format: finding_id, layer (one of 5), raw_output (PII redacted), tool_used, captured_at, custody_seal (HMAC), evidence_hash (SHA-256). Tamper detection on every read.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/evidence-collection-standards` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
