---
name: auth-bypass-methods
description: Auth bypass techniques — JWT, OAuth, session manipulation
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["auth", "jwt", "oauth", "session"]
  triggers: ["authentication", "jwt", "oauth", "session"]
allowed-tools: "nuclei,sqlmap,ffuf"
mapped-agents: ["penetration", "vulnerability-triage"]
---

# auth-bypass-methods

## Overview

Authentication bypass methods: JWT manipulation (alg:none, key confusion), OAuth flow abuse, session fixation/prediction, brute force, credential stuffing.

## When to Use

When target has login/auth endpoints. Loaded by penetration + vulnerability-triage agents.

**Mapped agents**: penetration, vulnerability-triage

## Methodology

- 1. Identify auth mechanism (cookie, JWT, basic auth)
- 2. JWT analysis: decode header/payload, test alg:none, brute key
- 3. Session analysis: predict session IDs, test fixation
- 4. Brute force (use ffuf with username/password wordlists)
- 5. OAuth: check redirect_uri, state CSRF, scope escalation
- 6. MFA bypass: replay OTP, test race conditions

## Tool Recipes

```bash
jwt_tool '<JWT_TOKEN>' -C -d /usr/share/wordlists/rockyou.txt
```
```bash
ffuf -u 'http://target/login' -X POST -d 'user=FUZZ&pass=admin' -w users.txt
```
```bash
nuclei -u http://target -t exposures/files/jwt-secret-key.yaml
```

## Output Standards

Each finding: auth mechanism, bypass technique, MITRE ATT&CK T1078 (Valid Accounts), evidence (request/response), remediation.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/auth-bypass-methods` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
