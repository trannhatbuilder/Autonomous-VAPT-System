---
name: attack-surface-recon
description: Recon methodology — nmap workflow, subdomain enumeration, port/service mapping
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["recon", "nmap", "subdomain", "ports", "services"]
  triggers: ["recon", "port scan", "subdomain", "fingerprint"]
allowed-tools: "nmap,httpx,whatweb,masscan,rustscan,subfinder,amass,dnsenum,fierce,gau,waybackurls"
mapped-agents: ["recon", "intel-collection", "attack-surface-enumeration"]
---

# attack-surface-recon

## Overview

Reconnaissance methodology following PTES (Penetration Testing Execution Standard) Intelligence Gathering phase. Covers: passive OSINT, active port scanning, service fingerprinting, subdomain enumeration.

## When to Use

Always — first phase of every scan. Loaded by recon, intel-collection, attack-surface-enumeration agents.

**Mapped agents**: recon, intel-collection, attack-surface-enumeration

## Methodology

- 1. Passive OSINT (theHarvester for emails, subfinder/amass for subdomains)
- 2. DNS recon (dnsenum + fierce for zone transfers + DNS records)
- 3. Historical URLs (gau + waybackurls for past endpoints)
- 4. Active port scan (nmap -sS -sV --top-ports 1000)
- 5. Service fingerprint (httpx for HTTP tech, whatweb for stack)
- 6. Persist findings to PentestFact blackboard (target/<host>)

## Tool Recipes

```bash
theharvester -d example.com -b all -f recon.html
```
```bash
subfinder -d example.com -all -recursive -o subdomains.txt
```
```bash
nmap -sS -sV --top-ports 1000 -oA nmap_full 192.168.1.0/24
```
```bash
httpx -l hosts.txt -title -tech-detect -status-code
```

## Output Standards

For each discovered asset: IP/hostname, open ports, service+version, tech stack. Map to MITRE ATT&CK T1595 (Active Scanning) + T1592 (Gather Victim Host Info).

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/attack-surface-recon` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
