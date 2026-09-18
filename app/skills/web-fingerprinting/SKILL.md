---
name: web-fingerprinting
description: Web tech fingerprinting — whatweb + wappalyzer-style stack detection
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["fingerprint", "wappalyzer", "whatweb", "tech-detect"]
  triggers: ["fingerprint", "tech stack", "whatweb"]
allowed-tools: "whatweb,httpx,wpscan,nuclei"
mapped-agents: ["attack-surface-enumeration", "recon"]
---

# web-fingerprinting

## Overview

Web technology fingerprinting via HTTP headers + HTML patterns. Identifies: web server (Apache/Nginx), framework (Django/Express/Laravel), CMS (WordPress/Drupal), JS libraries (jQuery/React).

## When to Use

After HTTP endpoint discovery. Loaded by attack-surface-enumeration + recon agents.

**Mapped agents**: attack-surface-enumeration, recon

## Methodology

- 1. Fetch headers (curl -sI) — look for Server, X-Powered-By, X-Generator
- 2. Analyze HTML <head> (meta generator, script src, link href)
- 3. Probe common paths (/wp-admin, /admin, /server-status)
- 4. Use whatweb -a 3 for aggressive detection
- 5. Use nuclei -t technologies/ for additional fingerprinting
- 6. Cross-reference with component-vuln-intel skill for CVE matching

## Tool Recipes

```bash
whatweb -a 3 -v http://target
```
```bash
curl -sI http://target | grep -iE 'server|x-powered|x-generator'
```
```bash
nuclei -u http://target -t technologies/ -severity info,low
```

## Output Standards

Each detected tech: name, version, confidence (%), source (header/HTML/path), detected endpoints. Persist to PentestFact blackboard as 'infra/' category.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/web-fingerprinting` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
