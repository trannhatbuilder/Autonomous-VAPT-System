---
name: unlimited-attack-scope
description: Full kill-chain playbook — exploit → privesc → lateral → impact
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["kill-chain", "full-scope", "lateral"]
  triggers: ["kill chain", "full scope", "lateral movement"]
allowed-tools: "sqlmap,metasploit,mimikatz,netexec,impacket,responder"
mapped-agents: ["penetration", "lateral-movement"]
---

# unlimited-attack-scope

## Overview

Full kill-chain playbook following MITRE ATT&CK Framework: Initial Access → Execution → Persistence → Privilege Escalation → Defense Evasion → Credential Access → Discovery → Lateral Movement → Collection → Exfiltration → Impact.

## When to Use

For plan_execute mode (full kill-chain scan). Loaded by penetration + lateral-movement agents.

**Mapped agents**: penetration, lateral-movement

## Methodology

- 1. Initial access (exploit web vuln via web-attack-methods)
- 2. Execution (run shell command on target)
- 3. Privilege escalation (use post-exploitation skill)
- 4. Credential access (mimikatz, /etc/shadow dump)
- 5. Discovery (internal network scan with masscan)
- 6. Lateral movement (pass-the-hash, Kerberos tickets via impacket)
- 7. Collection (identify high-value targets — DC, file servers)
- 8. Impact proof (use impact-proof-methodology skill)
- 9. Cleanup (use cleanup-rollback-procedures skill)

## Tool Recipes

```bash
metasploit: use exploit/multi/handler; set PAYLOAD windows/x64/meterpreter/reverse_tcp
```
```bash
impacket-smbexec user:password@target
```
```bash
netexec smb 192.168.1.0/24 -u user -p password --shares
```

## Output Standards

Each kill-chain step: MITRE ATT&CK technique ID, evidence (command + output), affected systems, chain link to next step.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/unlimited-attack-scope` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
