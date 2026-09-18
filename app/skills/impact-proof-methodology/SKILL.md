---
name: impact-proof-methodology
description: Business impact proof without data exfiltration
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["impact", "exfiltration", "business"]
  triggers: ["impact", "exfiltration", "business"]
allowed-tools: "metasploit"
mapped-agents: ["impact-exfiltration", "reporting-remediation"]
---

# impact-proof-methodology

## Overview

Business impact proof methodology — demonstrates accessibility of sensitive data WITHOUT actual exfiltration. Emphasizes PII redaction via app/pii/redactor.py.

## When to Use

To prove business impact of exploited vulnerabilities. Loaded by impact-exfiltration + reporting-remediation agents.

**Mapped agents**: impact-exfiltration, reporting-remediation

## Methodology

- 1. Identify high-value targets (databases, file servers, admin panels)
- 2. Demonstrate accessibility (count records, NOT dump them)
- 3. NEVER actually exfiltrate data — only prove accessibility
- 4. Redact all PII via app/pii/redactor.py before storing evidence
- 5. Capture: count of accessible records, sample (1 record, PII redacted)
- 6. Map to MITRE ATT&CK TA0009 (Collection), TA0010 (Exfiltration) — but with NO exfiltration

## Tool Recipes

```bash
metasploit: post/windows/gather/credentials/credential_collector
```
```bash
sqlmap --count (instead of --dump)
```
```bash
app/pii/redactor.py for scrubbing output
```

## Output Standards

Impact proof: accessible resource count, sample (redacted), accessibility method, MITRE ATT&CK technique. NO actual data exfiltration. PII redaction is MANDATORY.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/impact-proof-methodology` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
