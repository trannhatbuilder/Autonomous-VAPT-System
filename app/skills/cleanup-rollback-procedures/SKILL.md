---
name: cleanup-rollback-procedures
description: Post-scan cleanup script + verification checklist
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["cleanup", "rollback", "post-scan"]
  triggers: ["cleanup", "rollback", "post-scan"]
allowed-tools: ""
mapped-agents: ["cleanup-rollback"]
---

# cleanup-rollback-procedures

## Overview

Post-scan cleanup procedures — runs scripts/cleanup_scan.sh + verifies no persistence left behind + produces cleanup verification report.

## When to Use

At scan completion (always). Loaded by cleanup-rollback agent.

**Mapped agents**: cleanup-rollback

## Methodology

- 1. Run scripts/cleanup_scan.sh (removes /tmp/sqlmap-*, ~/.msf6/loot/, ~/.msf6/logs/)
- 2. Verify no persistence left (check cron, systemd, registry Run keys)
- 3. Remove any uploaded webshells
- 4. Remove any beacon payloads (c2_payloads/ dir)
- 5. Document cleanup actions in audit log
- 6. Produce cleanup verification report for evidence chain

## Tool Recipes

```bash
scripts/cleanup_scan.sh scan_<id>
```
```bash
crontab -l | grep -v 'beacon\|reverse' | crontab -
```
```bash
reg delete 'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' /v Update /f
```

## Output Standards

Cleanup report: list of removed artifacts, list of checked persistence mechanisms (all clean), audit log entries, timestamp. Each cleanup action recorded as PentestFact (cleanup_action).

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/cleanup-rollback-procedures` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
