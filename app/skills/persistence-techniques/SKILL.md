---
name: persistence-techniques
description: Persistence mechanisms + cleanup procedures
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["persistence", "scheduled-task", "service", "registry"]
  triggers: ["persistence", "backdoor", "scheduled task"]
allowed-tools: "metasploit"
mapped-agents: ["persistence-maintenance", "cleanup-rollback"]
---

# persistence-techniques

## Overview

Persistence mechanisms: Linux (cron, systemd service, SSH keys, bashrc) + Windows (scheduled task, service, registry Run key, WMI subscription). Maps to MITRE ATT&CK TA0003 (Persistence).

## When to Use

After obtaining persistent access is desired (HITL approved). Loaded by persistence-maintenance + cleanup-rollback agents.

**Mapped agents**: persistence-maintenance, cleanup-rollback

## Methodology

- 1. Choose persistence mechanism (HITL approval required)
- 2. Linux: cron job, systemd service, SSH authorized_keys, bashrc
- 3. Windows: scheduled task, service, Run registry key, WMI subscription
- 4. Document persistence details (path, trigger, cleanup command)
- 5. Test persistence (reboot target, verify access survives)
- 6. ALWAYS create cleanup procedure alongside installation
- 7. Persist cleanup procedure in PentestFact (cleanup_action)

## Tool Recipes

```bash
metasploit: post/windows/manage/persistence_exe (with REXEC, REGRUN, SCHTASK options)
```
```bash
echo '* * * * * /tmp/beacon' | crontab -
```
```bash
reg add 'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' /v Update /t REG_SZ /d 'C:\Users\Public\beacon.exe'
```

## Output Standards

Each persistence: mechanism, path/registry key, trigger (boot/login/scheduled), cleanup command, MITRE ATT&CK technique (e.g. T1053 Scheduled Task). Cleanup MUST be documented before installation.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/persistence-techniques` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
