---
name: redteam-opsec
description: Red team OPSEC — low-noise testing, blue-team evasion considerations
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["opsec", "redteam", "evasion", "low-noise"]
  triggers: ["opsec", "evasion", "low noise", "blue team"]
allowed-tools: ""
mapped-agents: ["opsec-evasion", "persistence-maintenance"]
---

# redteam-opsec

## Overview

Operational Security (OPSEC) considerations for red team engagements. Advisory only — does NOT provide WAF bypass or AV evasion techniques (out of scope for VAPT-AI MVP). Covers: timing, payload encoding, log cleanup, blue-team detection surface.

## When to Use

Before + during active exploitation. Loaded by opsec-evasion agent as advisory context.

**Mapped agents**: opsec-evasion, persistence-maintenance

## Methodology

- 1. Slow scan timing (nmap -T2 to evade IDS)
- 2. Rotate User-Agent + randomize delays between requests
- 3. Avoid peak business hours for noisy scans
- 4. Use proxychains + Tor for anonymous enumeration
- 5. Document every action for cleanup (cleanup-rollback skill)
- 6. NEVER attempt to disable EDR/AV — out of scope

## Tool Recipes

```bash
nmap -T2 --max-rate 50 -sS -p 1-1000 target
```
```bash
ffuf -u 'http://target/FUZZ' -w wordlist.txt -rate 10
```
```bash
sqlmap --delay=5 --random-agent --time-sec=15
```

## Output Standards

OPSEC recommendations documented as advisory notes. NO actual bypass techniques. Map to MITRE ATT&CK T1070 (Indicator Removal) only for cleanup procedures.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/redteam-opsec` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
