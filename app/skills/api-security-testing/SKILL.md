---
name: api-security-testing
description: REST/GraphQL API pentest methodology — OWASP API Top 10
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["api", "rest", "graphql", "owasp-api"]
  triggers: ["api", "rest", "graphql", "openapi"]
allowed-tools: "nuclei,nikto,sqlmap,ffuf,arjun"
mapped-agents: ["vulnerability-triage", "penetration"]
---

# api-security-testing

## Overview

API security testing per OWASP API Security Top 10 (2023): BOLA, Broken Authentication, Excessive Data Exposure, Lack of Rate Limiting, Broken Function Level Authorization, Mass Assignment, SSRF, Improper Inventory, Unsafe API Consumption.

## When to Use

When target exposes an API (REST or GraphQL). Loaded by vulnerability-triage + penetration agents.

**Mapped agents**: vulnerability-triage, penetration

## Methodology

- 1. Discover API endpoints (OpenAPI spec at /openapi.json, /swagger.json, /docs)
- 2. Map parameters (use arjun for hidden params)
- 3. Test BOLA (replace object IDs in URL/body)
- 4. Test broken auth (missing JWT, expired token, weak signatures)
- 5. Test rate limiting (send 100 requests/sec, observe response)
- 6. Test GraphQL (introspection query, batch attacks)
- 7. Test SSRF (URL parameters pointing to internal services)

## Tool Recipes

```bash
ffuf -u 'http://target/api/FUZZ' -w api_endpoints.txt -mc 200,201,401,403
```
```bash
arjun -u http://target/api/v1/users
```
```bash
nuclei -u http://target -t exposures/apis/
```
```bash
graphw00f -t http://target/graphql
```

## Output Standards

Each finding: OWASP API Top 10 category (e.g. API1:2023-BOLA), HTTP method, endpoint, evidence (request/response), remediation.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/api-security-testing` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
