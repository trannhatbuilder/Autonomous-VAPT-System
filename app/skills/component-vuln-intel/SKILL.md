---
name: component-vuln-intel
description: Component fingerprinting + CVE matching (NVD lookup workflow)
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["cve", "nvd", "components", "dependencies"]
  triggers: ["cve", "component", "dependency", "version"]
allowed-tools: "nuclei,nikto,whatweb,httpx"
mapped-agents: ["vulnerability-triage", "intel-collection"]
---

# component-vuln-intel

## Overview

Third-party component vulnerability intelligence. Identifies software versions → matches against NVD CVE database → produces prioritized finding list.

## When to Use

After recon phase identifies tech stack. Loaded by vulnerability-triage + intel-collection agents.

**Mapped agents**: vulnerability-triage, intel-collection

## Methodology

- 1. Fingerprint components (whatweb for web stack, nuclei -t technologies)
- 2. For each component+version: query NVD API (app/core/cve_client.py)
- 3. Filter CVEs by CVSS severity (>=7.0 high, >=9.0 critical)
- 4. Verify CVE applies (some require specific configs)
- 5. For verified CVEs: capture PoC + remediation (upgrade version)
- 6. Persist to blackboard as candidate_finding

## Tool Recipes

```bash
whatweb -a 3 http://target
```
```bash
nuclei -u http://target -t technologies/ -severity medium,high,critical
```
```bash
curl 'https://services.nvd.nist.gov/rest/json/cves/2.0?cpeName=cpe:2.3:a:apache:http_server:2.4.41'
```

## Output Standards

Each CVE finding: CVE ID, CVSS v3.1 vector + score, affected version, patched version, exploitability (PoC available?), remediation (upgrade to X).

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/component-vuln-intel` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
