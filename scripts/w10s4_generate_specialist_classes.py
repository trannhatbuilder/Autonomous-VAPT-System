"""
W10-S4: Generate 13 sub-agent specialist classes.

Each class inherits from app.agents.base.BaseAgent and overrides _decide_next()
with a deterministic stub behavior per agent role.

W11+ will replace _decide_next with real LiteLLM calls.

Output:
    app/agents/specialists/__init__.py            (re-exports all 13 classes)
    app/agents/specialists/<agent_name>_agent.py  (13 files, one per sub-agent)

Run:
    python /home/z/my-project/vapt-ai/scripts/w10s4_generate_specialist_classes.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIALISTS_DIR = PROJECT_ROOT / "app" / "agents" / "specialists"
SPECIALISTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 13 sub-agent stub behaviors ----------

# Each entry: (agent_name, role_summary, stub_observation)
SUB_AGENTS = [
    (
        "engagement-planning",
        "Defines engagement scope, ROE, and success criteria; produces test blueprints",
        "Engagement plan (W10 stub): scope=http://example.com, success_criteria='discover exploitable vulns', "
        "ROE='non-destructive, HITL for destructive ops'. Blueprint: 1) recon 2) vuln-triage 3) penetration (HITL).",
    ),
    (
        "intel-collection",
        "OSINT + passive intelligence: emails, subdomains, DNS, historical URLs",
        "Intel collection (W10 stub): theHarvester found 3 emails; subfinder found 5 subdomains "
        "(api.example.com, dev.example.com, ...); waybackurls returned 47 historical URLs.",
    ),
    (
        "recon",
        "Active port scanning + service fingerprinting",
        "Recon (W10 stub): nmap found ports 22 (ssh OpenSSH 8.2p1), 80 (http Apache 2.4.41), "
        "443 (https OpenSSL 1.1.1f); httpx confirmed HTTP tech stack.",
    ),
    (
        "attack-surface-enumeration",
        "Maps attack surface: subdomains, dirs, files, tech stacks, entry points",
        "Attack surface (W10 stub): katana crawled 23 URLs; feroxbuster found /admin, /api/v1, "
        "/backup.zip; whatweb identified Apache 2.4.41 + PHP 7.4 + MySQL.",
    ),
    (
        "vulnerability-triage",
        "Vulnerability candidate detection + severity ranking (evidence-centric, NOT weaponized)",
        "Vuln triage (W10 stub): nuclei detected 3 findings (1 medium: CVE-2021-41773 path traversal; "
        "2 low: missing HSTS, server version disclosure). Recommend penetration agent for validation.",
    ),
    (
        "penetration",
        "Active exploitation to validate findings (HITL required for destructive ops)",
        "Penetration (W10 stub): sqlmap confirmed SQLi on /api/users?id=1 (boolean-blind). "
        "Recommend --os-shell (requires HITL approval) for further exploitation.",
    ),
    (
        "privilege-escalation",
        "Privilege escalation path enumeration (HITL required)",
        "Privilege escalation (W10 stub): linpeas found SUID /usr/bin/find (GTFOBins: privesc). "
        "winpeas found unquoted service path 'C:\\Program Files\\VulnSvc\\service.exe'.",
    ),
    (
        "lateral-movement",
        "Internal network discovery + credential/session exploitation (HITL required)",
        "Lateral movement (W10 stub): netexec smb 192.168.1.0/24 found 4 hosts, 1 Pwn3d! "
        "(192.168.1.10 with creds user:Pass123). impacket-smbexec available for RCE.",
    ),
    (
        "persistence-maintenance",
        "Persistence mechanism evaluation (HITL required)",
        "Persistence (W10 stub): evaluated 3 mechanisms: 1) cron job, 2) systemd service, "
        "3) SSH authorized_keys. All require HITL approval. Recommend documenting cleanup steps.",
    ),
    (
        "impact-exfiltration",
        "Business impact + data accessibility proof (HITL required, NO actual exfiltration)",
        "Impact (W10 stub): metasploit post/windows/gather/credentials found 2 hashes + 1 cleartext. "
        "Accessibility proven. NO exfiltration performed. PII redacted from evidence.",
    ),
    (
        "opsec-evasion",
        "Advisory: low-noise testing strategies (no WAF bypass / evasion exploits)",
        "OPSEC advisory (W10 stub): recommend 1) slow nmap timing (-T2), 2) rotate User-Agent, "
        "3) avoid peak hours. Do NOT provide WAF bypass techniques — out of scope.",
    ),
    (
        "cleanup-rollback",
        "Runs cleanup_scan.sh + verifies no persistence left behind",
        "Cleanup (W10 stub): scripts/cleanup_scan.sh removed /tmp/sqlmap-*, ~/.msf6/loot/, "
        "~/.msf6/logs/, nuclei output dirs. Verified no cron jobs, no systemd services, "
        "no SSH authorized_keys modifications. Audit log entry created.",
    ),
    (
        "reporting-remediation",
        "Aggregates blackboard facts into structured report",
        "Reporting (W10 stub): aggregated 3 findings (1 critical SQLi, 1 medium path traversal, "
        "1 low info disclosure). Report sections: Executive Summary, Findings, Remediation, "
        "Disclosure Window (ISO 29147). Recommend upgrade Apache, patch CVE-2021-41773, add HSTS.",
    ),
]


# ---------- Template ----------

TEMPLATE = '''\
"""
{class_name} — {agent_name} specialist.

{description}

Ported from CyberStrikeAI agents/{agent_name}.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/{agent_name}.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: {safety_class}
HITL required: {hitl_required}

Tool allowlist: {tool_allowlist_repr}

Usage:
    from app.agents.specialists.{module_name} import {class_name}

    agent = {class_name}(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class {class_name}(BaseAgent):
    """{description}

    Safety class: {safety_class}
    HITL required: {hitl_required}
    """

    AGENT_NAME = "{agent_name}"

    async def _decide_next(
        self,
        turn: int,
    ) -> tuple[str, str | None, dict[str, Any], str]:
        """W10 stub: return deterministic canned response.

        W11+ will replace with LiteLLM call using self.system_prompt +
        self.task_description + blackboard context.

        Returns:
            (thought, tool_name, tool_args, observation)
        """
        task_summary = self.task_description[:80] if self.task_description else "(no task description)"
        thought = "{role_summary} on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "{stub_observation_escaped}"

        return thought, tool_name, tool_args, observation
'''


# ---------- Main ----------

def to_module_name(agent_name: str) -> str:
    """e.g. 'attack-surface-enumeration' → 'attack_surface_enumeration_agent'"""
    return agent_name.replace("-", "_") + "_agent"


def to_class_name(agent_name: str) -> str:
    """e.g. 'attack-surface-enumeration' → 'AttackSurfaceEnumerationAgent'"""
    parts = agent_name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts) + "Agent"


def render_template(
    agent_name: str,
    role_summary: str,
    stub_observation: str,
    safety_class: str,
    hitl_required: bool,
    tool_allowlist: list[str],
) -> str:
    class_name = to_class_name(agent_name)
    module_name = to_module_name(agent_name)

    # Escape backslashes + double quotes for safe embedding in Python string literal
    safe_obs = stub_observation.replace("\\", "\\\\").replace('"', '\\"')
    # Escape any other characters that could break Python string parsing (newlines)
    safe_obs = safe_obs.replace("\n", "\\n")

    return TEMPLATE.format(
        class_name=class_name,
        module_name=module_name,
        agent_name=agent_name,
        description=role_summary,
        role_summary=role_summary,
        stub_observation_escaped=safe_obs,
        safety_class=safety_class,
        hitl_required=str(hitl_required),
        tool_allowlist_repr=repr(tool_allowlist),
    )


def main() -> int:
    # Load registry to get safety_class + tool_allowlist per agent
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.agents.registry import agent_registry

    generated = []
    for agent_name, role_summary, stub_observation in SUB_AGENTS:
        meta = agent_registry.require_agent(agent_name)
        content = render_template(
            agent_name=agent_name,
            role_summary=role_summary,
            stub_observation=stub_observation,
            safety_class=meta.safety_class,
            hitl_required=meta.is_destructive,
            tool_allowlist=list(meta.tool_allowlist),
        )
        module_name = to_module_name(agent_name)
        out_path = SPECIALISTS_DIR / f"{module_name}.py"
        out_path.write_text(content, encoding="utf-8")
        generated.append((agent_name, class_name_for(agent_name), len(content), out_path))

    # Generate __init__.py with re-exports
    init_content = generate_init_py()
    init_path = SPECIALISTS_DIR / "__init__.py"
    init_path.write_text(init_content, encoding="utf-8")

    print(f"Generated {len(generated)} specialist classes + __init__.py:")
    for agent_name, cls_name, size, path in generated:
        rel = path.relative_to(PROJECT_ROOT)
        print(f"  {agent_name:35s}  {cls_name:40s}  {size:5d} bytes  {rel}")
    print(f"  {'__init__.py':35s}  {'':40s}  {len(init_content):5d} bytes  {init_path.relative_to(PROJECT_ROOT)}")
    return 0


def class_name_for(agent_name: str) -> str:
    return to_class_name(agent_name)


def generate_init_py() -> str:
    imports = []
    exports = []
    for agent_name, _, _ in SUB_AGENTS:
        module_name = to_module_name(agent_name)
        class_name = to_class_name(agent_name)
        imports.append(f"from app.agents.specialists.{module_name} import {class_name}")
        exports.append(f'    "{class_name}",')

    return (
        '"""\n'
        "VAPT-AI Specialist Agents — W10-S4.\n"
        "\n"
        "13 sub-agent classes (one per non-orchestrator agent in app/agents/*.md).\n"
        "Each class inherits from app.agents.base.BaseAgent.\n"
        "\n"
        "Generated by scripts/w10s4_generate_specialist_classes.py.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "\n"
        + "\n".join(imports)
        + "\n\n\n"
        + "__all__ = [\n"
        + "\n".join(exports)
        + "\n]\n"
    )


if __name__ == "__main__":
    sys.exit(main())