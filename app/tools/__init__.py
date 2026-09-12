"""VAPT-AI tool wrappers package — MCP tool definitions for security CLI tools.

Each tool is defined as a YAML file in this directory (e.g. nmap.yaml, nuclei.yaml).
The loader (loader.py) reads all YAMLs at startup and registers them as MCP tools.

W2-A: 10 core tools (nmap, nuclei, sqlmap, gobuster, feroxbuster, subfinder,
      httpx, whatweb, nikto, dalfox)
W5: 20+ additional tools (Metasploit, hashcat, john, impacket, etc.)
"""
