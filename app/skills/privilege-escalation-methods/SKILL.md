---
name: privilege-escalation-methods
description: Linux + Windows privesc methodology — GTFOBins + winPEAS workflow
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["privesc", "linux", "windows", "gtfobins"]
  triggers: ["privilege escalation", "privesc", "root", "admin"]
allowed-tools: "linpeas,winpeas,mimikatz"
mapped-agents: ["privilege-escalation"]
---

# privilege-escalation-methods

## Overview

Privilege escalation methodology — Linux (SUID/GTFOBins/cron/capabilities) + Windows (service paths/registry/SeImpersonate). Maps to MITRE ATT&CK TA0004 (Privilege Escalation).

## When to Use

After initial shell as low-priv user. Loaded by privilege-escalation agent.

**Mapped agents**: privilege-escalation

## Methodology

- 1. Run linpeas (Linux) or winpeas (Windows) — automated enum
- 2. Identify SUID binaries → check GTFOBins for exploitation
- 3. Check sudo permissions (sudo -l)
- 4. Cron job abuse (cron.d, /etc/crontab, user crons)
- 5. Linux capabilities (getcap -r / 2>/dev/null)
- 6. Windows: unquoted service paths, writable service binPath
- 7. Windows: SeImpersonate abuse (JuicyPotato, PrintSpoofer)
- 8. Capture before/after uid evidence (id; whoami /priv)

## Tool Recipes

```bash
linpeas.sh -a 2>/dev/null | tee linpeas.txt
```
```bash
winpeas.exe (or winPEASx64.exe)
```
```bash
find / -perm -4000 -type f 2>/dev/null  # SUID
```
```bash
getcap -r / 2>/dev/null  # Linux capabilities
```

## Output Standards

Each privesc path: technique (GTFOBins link or MITRE technique ID), original uid, escalated uid, command output, MITRE ATT&CK T1068 (Exploitation for Privilege Escalation).

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/privilege-escalation-methods` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
