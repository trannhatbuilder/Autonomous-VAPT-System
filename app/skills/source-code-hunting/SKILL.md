---
name: source-code-hunting
description: Source code review for vulnerabilities (SAST) — pattern matching + taint analysis
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["sast", "code-review", "static-analysis"]
  triggers: ["source code", "code review", "sast"]
allowed-tools: ""
mapped-agents: ["vulnerability-triage", "attack-surface-enumeration"]
---

# source-code-hunting

## Overview

Static Application Security Testing (SAST) methodology. Reviews source code for common vulnerability patterns: SQLi (string concat), XSS (unescaped output), hardcoded secrets, insecure deserialization, path traversal.

## When to Use

When source code is available (white-box or grey-box pentest). Loaded by vulnerability-triage + attack-surface-enumeration agents.

**Mapped agents**: vulnerability-triage, attack-surface-enumeration

## Methodology

- 1. Identify tech stack (Python/PHP/Node/Java) — use whatweb output
- 2. Search for common sink patterns: eval, exec, system, unescape, pickle.loads
- 3. Trace user input to sinks (taint analysis)
- 4. Check for hardcoded secrets (grep -r 'password=' 'api_key=')
- 5. Identify outdated dependencies (CVE scanning via component-vuln-intel skill)
- 6. Verify findings dynamically (manual PoC + tool validation)

## Tool Recipes

```bash
grep -rn 'eval\|exec\|system' --include='*.py' .
```
```bash
grep -rn 'mysqli_query\|mysql_query' --include='*.php' .
```
```bash
grep -rn 'innerHTML\|dangerouslySetInnerHTML' --include='*.js' .
```
```bash
grep -rE '(password|api_key|secret)\s*=' .
```

## Output Standards

Each finding: source file + line, vulnerable pattern, taint path (source → sink), CWE ID, suggested fix. Use MITRE ATT&CK T1027 (Obfuscated Files) for anti-pattern detection.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/source-code-hunting` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
