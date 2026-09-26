#!/usr/bin/env bash
# ============================================================
# VAPT-AI Tool Availability Diagnostic
# ============================================================
#
# Verifies that all 32 security tools are installed AND executable by
# the systemd `nhat` user (User=nhat in vapt-ai-app.service).
#
# Run from your VAPT-AI home:
#     cd ~/VAPT-AI
#     bash scripts/diagnose_tools.sh
#
# The output tells you:
#   ✅ Found at <path> (executable)
#   ⚠  Found at <path> but NOT executable by nhat (Permission denied)
#   ❌ Not found in any PATH
#
# For each missing/broken tool, the script prints the install command.
# ============================================================

set -uo pipefail

# Simulate the systemd PATH (after our Phase D-2 fix)
# This matches the Environment=PATH= line in vapt-ai-app.service
SYSTEMD_PATH="/home/nhat/go/bin:/root/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/opt/go/bin:/opt/nmap/bin:/opt/feroxbuster:/opt/sqlmap"

# Also include venv bin (since uvicorn's child processes inherit this)
VENV_BIN="$HOME/VAPT-AI/venv/bin"
FULL_PATH="$VENV_BIN:$SYSTEMD_PATH"

# Tools to check — must match app/tools/*.yaml
TOOLS=(
  amass dalfox dnsenum feroxbuster ffuf fierce fscan gau gobuster hashcat
  httpx hydra impacket-smbexec john katana linpeas masscan metasploit
  mimikatz netexec nikto nmap nuclei responder rustscan sqlmap subfinder
  theharvester waybackurls whatweb winpeas wpscan
)

# Install hints for missing tools
declare -A INSTALL_HINTS=(
  [amass]="go install -v github.com/owasp-amass/amass/v4/...@master"
  [dalfox]="go install github.com/hahwul/dalfox@latest"
  [dnsenum]="apt install dnsenum"
  [feroxbuster]="cargo install feroxbuster OR download from https://github.com/epi052/feroxbuster/releases"
  [ffuf]="go install github.com/ffuf/ffuf/v2@latest"
  [fierce]="apt install fierce OR gem install fierce"
  [fscan]="go install github.com/shadow1ng/fscan@latest"
  [gau]="go install github.com/lc/gau/v2/cmd/gau@latest"
  [gobuster]="apt install gobuster OR go install github.com/OJ/gobuster@latest"
  [hashcat]="apt install hashcat"
  [httpx]="go install github.com/projectdiscovery/httpx/cmd/httpx@latest"
  [hydra]="apt install hydra"
  [impacket-smbexec]="pipx install impacket"
  [john]="apt install john"
  [katana]="go install github.com/projectdiscovery/katana/cmd/katana@latest"
  [linpeas]="curl -L https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh -o /usr/local/bin/linpeas && chmod +x /usr/local/bin/linpeas"
  [masscan]="apt install masscan"
  [metasploit]="apt install metasploit-framework"
  [mimikatz]="Download from https://github.com/gentilkiwi/mimikatz/releases"
  [netexec]="pipx install netexec"
  [nikto]="apt install nikto OR git clone https://github.com/sullo/nikto.git"
  [nmap]="apt install nmap"
  [nuclei]="go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest"
  [responder]="git clone https://github.com/lgandx/Responder.git /opt/Responder && ln -s /opt/Responder/Responder.py /usr/local/bin/responder"
  [rustscan]="cargo install rustscan OR docker pull rustscan/rustscan:latest"
  [sqlmap]="apt install sqlmap OR git clone https://github.com/sqlmapproject/sqlmap.git /opt/sqlmap && ln -s /opt/sqlmap/sqlmap.py /usr/local/bin/sqlmap"
  [subfinder]="go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest"
  [theharvester]="apt install theharvester OR pipx install theharvester"
  [waybackurls]="go install github.com/tomnomnom/waybackurls@latest"
  [whatweb]="apt install whatweb OR gem install whatweb"
  [winpeas]="curl -L https://github.com/peass-ng/PEASS-ng/releases/latest/download/winPEASx64.exe -o /usr/local/bin/winpeas.exe"
  [wpscan]="gem install wpscan"
)

# Print header
echo "============================================================"
echo "VAPT-AI Tool Diagnostic"
echo "============================================================"
echo "Checked PATH (simulates systemd service):"
echo "  $FULL_PATH"
echo ""
echo "Running as user: $(whoami) (systemd User=nhat)"
echo ""

# Counters
FOUND_OK=0
FOUND_NOEXEC=0
MISSING=0

# Check each tool
for tool in "${TOOLS[@]}"; do
  # Search PATH for the binary
  found_path=""
  for dir in $(echo "$FULL_PATH" | tr ':' ' '); do
    if [ -x "$dir/$tool" ]; then
      found_path="$dir/$tool"
      break
    elif [ -f "$dir/$tool" ]; then
      # File exists but not executable — record and continue searching
      found_path="$dir/$tool (NOT EXECUTABLE)"
    fi
  done

  if [ -z "$found_path" ]; then
    echo "❌ $tool — not found in PATH"
    if [ -n "${INSTALL_HINTS[$tool]:-}" ]; then
      echo "   Install: ${INSTALL_HINTS[$tool]}"
    fi
    MISSING=$((MISSING + 1))
  elif [[ "$found_path" == *"NOT EXECUTABLE"* ]]; then
    echo "⚠  $tool — file exists but NOT executable: $found_path"
    echo "   Fix: sudo chmod 755 $found_path"
    FOUND_NOEXEC=$((FOUND_NOEXEC + 1))
  else
    # Verify it actually runs (catches "Permission denied" on exec attempt)
    # Use timeout to avoid hangs
    actual_path="$found_path"
    if timeout 5 "$actual_path" --version >/dev/null 2>&1 || \
       timeout 5 "$actual_path" -V >/dev/null 2>&1 || \
       timeout 5 "$actual_path" -h >/dev/null 2>&1; then
      echo "✅ $tool — at $actual_path"
      FOUND_OK=$((FOUND_OK + 1))
    else
      rc=$?
      if [ $rc -eq 126 ]; then
        echo "⚠  $tool — found at $actual_path but Permission denied on execute"
        echo "   Fix: sudo chmod 755 $actual_path OR sudo chown nhat:nhat $actual_path"
        FOUND_NOEXEC=$((FOUND_NOEXEC + 1))
      elif [ $rc -eq 127 ]; then
        echo "❌ $tool — command not found (file missing in PATH)"
        if [ -n "${INSTALL_HINTS[$tool]:-}" ]; then
          echo "   Install: ${INSTALL_HINTS[$tool]}"
        fi
        MISSING=$((MISSING + 1))
      else
        # Tool found + executable but errored on --version (some tools don't support -V)
        echo "✅ $tool — at $actual_path (exit code $rc on --version probe — OK if tool doesn't support --version)"
        FOUND_OK=$((FOUND_OK + 1))
      fi
    fi
  fi
done

# Summary
echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "  ✅ Working:        $FOUND_OK / ${#TOOLS[@]}"
echo "  ⚠  Not executable: $FOUND_NOEXEC"
echo "  ❌ Missing:        $MISSING"
echo ""

if [ $FOUND_NOEXEC -gt 0 ]; then
  echo "Permission-denied fixes (run as root or via sudo):"
  echo "  sudo chown -R nhat:nhat ~/go/bin/"
  echo "  sudo chmod -R 755 ~/go/bin/"
  echo "  # If tools are in /root/go/bin/ (installed via sudo), move them:"
  echo "  sudo cp /root/go/bin/{httpx,nuclei,subfinder,katana,dalfox,gau,waybackurls} /usr/local/bin/"
  echo "  sudo chmod 755 /usr/local/bin/{httpx,nuclei,subfinder,katana,dalfox,gau,waybackurls}"
fi

if [ $MISSING -gt 0 ]; then
  echo ""
  echo "Install all missing Go tools at once:"
  echo "  go install github.com/projectdiscovery/httpx/cmd/httpx@latest \\"
  echo "         github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest \\"
  echo "         github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest \\"
  echo "         github.com/projectdiscovery/katana/cmd/katana@latest \\"
  echo "         github.com/hahwul/dalfox@latest \\"
  echo "         github.com/lc/gau/v2/cmd/gau@latest \\"
  echo "         github.com/tomnomnom/waybackurls@latest \\"
  echo "         github.com/ffuf/ffuf/v2@latest"
  echo ""
  echo "Or run the bundled installer:"
  echo "  bash scripts/install_tools.sh"
fi

# Suggest restart if any tools were fixed
echo ""
echo "After fixing tool permissions, restart the backend to pick up PATH changes:"
echo "  sudo systemctl restart vapt-ai-app"