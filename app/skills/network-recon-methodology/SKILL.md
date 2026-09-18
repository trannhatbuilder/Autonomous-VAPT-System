---
name: network-recon-methodology
description: Network-level recon — SMB, Kerberos, AD enumeration
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["network", "smb", "kerberos", "ad"]
  triggers: ["network scan", "smb", "kerberos", "active directory"]
allowed-tools: "nmap,masscan,rustscan,netexec,impacket,enum4linux-ng,responder"
mapped-agents: ["recon", "intel-collection"]
---

# network-recon-methodology

## Overview

Network-level recon methodology — SMB enumeration (shares, sessions), Kerberos (user enum via GetUserSPNs.py), Active Directory (BloodHound graph). For internal network pentest.

## When to Use

When target is an internal network (not just web app). Loaded by recon + intel-collection agents.

**Mapped agents**: recon, intel-collection

## Methodology

- 1. Host discovery (nmap -sn for live hosts)
- 2. Port scan (nmap -sS -p 445,88,389,636 for AD)
- 3. SMB enum (netexec smb --shares, enum4linux-ng)
- 4. Kerberos user enum (GetUserSPNs.py -usersfile users.txt)
- 5. AD graph (bloodhound-python -u user -p pass -d domain.local -c All)
- 6. Identify high-value targets (DC, SQL servers, admin workstations)

## Tool Recipes

```bash
nmap -sn 192.168.1.0/24 -oA hosts
```
```bash
netexec smb 192.168.1.0/24 --shares -u '' -p ''
```
```bash
bloodhound-python -u user -p pass -d domain.local -c All -ns 192.168.1.1
```
```bash
impacket-GetUserSPNs.py domain.local/user:pass -request
```

## Output Standards

Map findings to MITRE ATT&CK T1018 (Remote System Discovery), T1087 (Account Discovery), T1069 (Permission Groups Discovery). Persist AD graph for lateral-movement agent.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/network-recon-methodology` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
