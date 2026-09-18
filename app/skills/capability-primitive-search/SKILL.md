---
name: capability-primitive-search
description: Capability primitive search — Wappalyzer-style tech stack + capability discovery
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["wappalyzer", "capabilities", "tech-stack"]
  triggers: ["capabilities", "tech stack", "wappalyzer"]
allowed-tools: "whatweb,httpx,wpscan"
mapped-agents: ["attack-surface-enumeration", "penetration"]
---

# capability-primitive-search

## Overview

Capability primitive search identifies the underlying technology stack + capabilities of a target. Used to narrow down attack surface to known-vulnerable tech.

## When to Use

After recon identifies HTTP endpoints. Loaded by attack-surface-enumeration + penetration agents.

**Mapped agents**: attack-surface-enumeration, penetration

## Methodology

- 1. Use whatweb for fingerprinting (server, framework, CMS, language)
- 2. Probe common endpoints (/admin, /api, /robots.txt, /sitemap.xml)
- 3. Identify authentication mechanism (cookie, JWT, basic auth)
- 4. Identify session management (cookie names, JWT claims)
- 5. Map capabilities to attack paths (e.g. WordPress → wpscan)
- 6. Persist findings to PentestFact blackboard (infra/ category)

## Tool Recipes

```bash
whatweb -a 3 -v http://target
```
```bash
httpx -title -tech-detect -status-code -follow-redirects
```
```bash
wpscan --url http://target --enumerate ap,at,u --random-user-agent
```

## Output Standards

Each capability: technology name, version, confidence, detected endpoints, suggested attack paths.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/capability-primitive-search` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
