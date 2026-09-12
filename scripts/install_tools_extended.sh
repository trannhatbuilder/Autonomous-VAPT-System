#!/usr/bin/env bash
# ============================================================
# VAPT-AI v3.4 — Extended Security Tools Installer (W5-A)
# Disk-Quota-Safe Edition (Ubuntu-Patched, Dual-Python)
# ============================================================
#
# v3.4 CHANGES:
#   - [0] Creates a SECOND venv (venv314) with Python 3.14 for
#         tools whose new releases require newer Python.
#   - [3] netexec: installed into venv314 (PyPI rejects 3.12).
#         Wrapper /usr/local/bin/netexec + nxc.
#   - [10] mimikatz: release only ships mimikatz_trunk.zip
#         (mimikatz.exe standalone does NOT exist -> 404).
#         Now downloads zip, extracts, wraps with wine.
#   - [13] fscan: broader asset pattern + fallback build from
#         source with Go.
#   - [21] theHarvester: latest requires Python >=3.14 ->
#         installed into venv314 from GitHub. Wrapper created.
#   - Verification updated for new command names.
#
# venv isolation:
#   ~/VAPT-AI/venv     (Python 3.12) - existing tools, untouched
#   ~/VAPT-AI/venv314  (Python 3.14) - netexec, theHarvester
#
# ============================================================

set -uo pipefail

# ============================================================
# Colors
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# Logging helpers
# ============================================================

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

die() {
    error "$*"
    exit 1
}

check_cmd() {
    command -v "$1" >/dev/null 2>&1
}

cleanup_tmp() {
    if [[ -d "${TMPDIR:-}" && "${TMPDIR:-}" == "$HOME/tmp" ]]; then
        find "$TMPDIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null || true
    fi
}

on_error() {
    local exit_code=$?
    echo ""
    warn "A command failed (exit code ${exit_code}) while processing: ${CURRENT_TOOL:-startup}"
    warn "Continuing with the next step..."
    warn "Temporary files are kept at: ${TMPDIR:-unknown}"
    return 0
}

CURRENT_TOOL="startup"
trap on_error ERR

# ============================================================
# Storage-safe environment
# ============================================================

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"

export XDG_CACHE_HOME="$HOME/.cache"

export GOPATH="$HOME/go"
export GOCACHE="$HOME/.cache/go-build"
export GOMODCACHE="$HOME/go/pkg/mod"

export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"

export GEM_HOME="$HOME/.gem"
export GEM_PATH="$GEM_HOME"

export RUSTC_WRAPPER=""

export PATH="/usr/local/go/bin:$HOME/go/bin:$CARGO_HOME/bin:$GEM_HOME/bin:$PATH"

# ============================================================
# Create storage directories
# ============================================================

mkdir -p \
    "$TMPDIR" \
    "$GOPATH/bin" \
    "$GOCACHE" \
    "$GOMODCACHE" \
    "$CARGO_HOME/bin" \
    "$RUSTUP_HOME" \
    "$GEM_HOME" \
    "$XDG_CACHE_HOME" \
    "$HOME/VAPT-AI/tools"

chmod 700 "$TMPDIR"

cleanup_tmp

# ============================================================
# Header
# ============================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}VAPT-AI v3.4 — Extended Tools Installer (W5-A)${NC}"
echo -e "${CYAN}Disk-Quota-Safe Edition (Dual-Python)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

info "Temporary storage:"
echo "  TMPDIR       = $TMPDIR"
echo "  TMP          = $TMP"
echo "  TEMP         = $TEMP"
echo ""

info "Build/cache storage:"
echo "  GOPATH       = $GOPATH"
echo "  GOCACHE      = $GOCACHE"
echo "  GOMODCACHE   = $GOMODCACHE"
echo "  CARGO_HOME   = $CARGO_HOME"
echo "  RUSTUP_HOME  = $RUSTUP_HOME"
echo "  GEM_HOME     = $GEM_HOME"
echo ""

# ============================================================
# OS / architecture
# ============================================================

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    info "Operating system: ${PRETTY_NAME:-Unknown}"
else
    warn "/etc/os-release not found."
fi

ARCH="$(uname -m)"

case "$ARCH" in
    x86_64)
        GO_ARCH="amd64"
        RUST_TARGET="x86_64-unknown-linux-gnu"
        ;;
    aarch64|arm64)
        GO_ARCH="arm64"
        RUST_TARGET="aarch64-unknown-linux-gnu"
        ;;
    *)
        die "Unsupported architecture: $ARCH"
        ;;
esac

info "Architecture: $ARCH"
echo ""

# ============================================================
# Python environment (main venv - 3.12)
# ============================================================

PYTHON_BIN=""

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$HOME/VAPT-AI/venv/bin/python" ]]; then
    PYTHON_BIN="$HOME/VAPT-AI/venv/bin/python"
elif check_cmd python3; then
    PYTHON_BIN="$(command -v python3)"
elif check_cmd python; then
    PYTHON_BIN="$(command -v python)"
else
    die "Python 3 was not found."
fi

info "Python (main venv): $("$PYTHON_BIN" --version 2>&1)"
info "Python path: $PYTHON_BIN"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "Active venv: $VIRTUAL_ENV"
elif [[ "$PYTHON_BIN" == "$HOME/VAPT-AI/venv/bin/python" ]]; then
    info "Using VAPT-AI venv: $HOME/VAPT-AI/venv"
else
    warn "No virtual environment is active."
    warn "Python packages will use: $PYTHON_BIN"
fi

VENV_BIN_DIR=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    VENV_BIN_DIR="${VIRTUAL_ENV}/bin"
elif [[ "$PYTHON_BIN" == "$HOME/VAPT-AI/venv/bin/python" ]]; then
    VENV_BIN_DIR="$HOME/VAPT-AI/venv/bin"
fi

if [[ -n "$VENV_BIN_DIR" && ":$PATH:" != *":$VENV_BIN_DIR:"* ]]; then
    export PATH="$VENV_BIN_DIR:$PATH"
    info "Added venv bin to PATH: $VENV_BIN_DIR"
fi

echo ""

# ============================================================
# Secondary venv (venv314) for tools requiring Python >= 3.13
# ============================================================

CURRENT_TOOL="venv314-setup"
info "[0/22] Secondary venv with newer Python (venv314)"

VENV314_DIR="$HOME/VAPT-AI/venv314"
PY314_BIN=""

for cand in python3.14 python3.13; do
    if check_cmd "$cand"; then
        PY314_BIN="$(command -v "$cand")"
        break
    fi
done

# Fallback: system python3 if it is >= 3.14
if [[ -z "$PY314_BIN" ]] && check_cmd python3; then
    SYS_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$(printf '%s\n' "3.14" "$SYS_VER" | sort -V | head -1)" == "3.14" ]]; then
        PY314_BIN="$(command -v python3)"
    fi
fi

if [[ -z "$PY314_BIN" ]]; then
    warn "No Python >= 3.14 interpreter found (python3.14 / python3.13)."
    warn "netexec and theHarvester (latest) need newer Python."
    warn "Install python3.14 then re-run this script to add them."
else
    PY314_NAME="$("$PY314_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
    info "Found interpreter: $PY314_BIN ($PY314_NAME)"

    # Ensure venv support (ensurepip) for this interpreter
    if ! "$PY314_BIN" -c "import ensurepip" >/dev/null 2>&1; then
        info "Installing ${PY314_NAME}-venv package..."
        install_apt_if_available "${PY314_NAME}-venv" 2>/dev/null || true
    fi

    if [[ -x "$VENV314_DIR/bin/python" ]]; then
        info "venv314 already exists: $VENV314_DIR"
    else
        info "Creating venv314..."
        "$PY314_BIN" -m venv "$VENV314_DIR" \
            || warn "Could not create venv314."
    fi

    if [[ -x "$VENV314_DIR/bin/python" ]]; then
        "$VENV314_DIR/bin/python" -m pip install --upgrade pip --no-cache-dir \
            >/dev/null 2>&1 || true
        info "venv314 Python: $("$VENV314_DIR/bin/python" --version 2>&1)"
    fi
fi

echo ""

# ============================================================
# Disk / inode pre-flight
# ============================================================

info "===== Storage Pre-flight ====="
echo ""

info "Root filesystem:"
df -h /
echo ""

info "Home filesystem:"
df -h "$HOME"
echo ""

info "Inode usage:"
df -ih /
echo ""

HOME_AVAILABLE_KB="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$((4 * 1024 * 1024))  # 4 GiB

if [[ -n "$HOME_AVAILABLE_KB" && "$HOME_AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
    warn "Less than 4 GiB free on the filesystem containing $HOME."
    warn "The extended set contains large packages (especially Metasploit)."
    warn "Installation is stopped before expensive downloads/builds."
    warn "Free disk space and run this script again."
    exit 1
fi

# ============================================================
# Package helpers
# ============================================================

install_apt() {
    local pkg="$1"

    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null \
        | grep -q "install ok installed"; then
        info "$pkg already installed (apt)"
        return 0
    fi

    info "Installing $pkg (apt)..."
    sudo apt install -y "$pkg"
    info "$pkg installed."
}

install_apt_if_available() {
    local pkg="$1"

    if apt-cache show "$pkg" >/dev/null 2>&1; then
        install_apt "$pkg"
        return 0
    fi

    return 1
}

install_go_tool() {
    local pkg_path="$1"
    local bin_name="$2"

    if check_cmd "$bin_name" || [[ -x "$HOME/go/bin/$bin_name" ]]; then
        info "$bin_name already installed (Go)"
        return 0
    fi

    info "Installing $bin_name via Go..."
    info "  go install ${pkg_path}@latest"

    (
        export TMPDIR="$TMPDIR"
        export TMP="$TMP"
        export TEMP="$TEMP"
        export GOPATH="$GOPATH"
        export GOCACHE="$GOCACHE"
        export GOMODCACHE="$GOMODCACHE"
        export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"

        go install "${pkg_path}@latest"
    )

    if [[ -x "$HOME/go/bin/$bin_name" ]]; then
        info "$bin_name installed: $HOME/go/bin/$bin_name"
    elif check_cmd "$bin_name"; then
        info "$bin_name installed."
    else
        warn "$bin_name installation failed."
        return 1
    fi
}

install_pip() {
    local pkg="$1"

    if "$PYTHON_BIN" -m pip show "$pkg" >/dev/null 2>&1; then
        info "$pkg already installed (pip)"
        return 0
    fi

    info "Installing $pkg with pip..."

    TMPDIR="$TMPDIR" \
    TMP="$TMP" \
    TEMP="$TEMP" \
    PIP_NO_CACHE_DIR=1 \
    "$PYTHON_BIN" -m pip install --no-cache-dir "$pkg"

    info "$pkg installed."
}

# Install into venv314 (newer Python)
install_pip314() {
    local pkg="$1"

    if [[ ! -x "${VENV314_DIR:-}/bin/python" ]]; then
        warn "venv314 is not available; cannot install $pkg (needs newer Python)."
        return 1
    fi

    if "$VENV314_DIR/bin/python" -m pip show "${pkg##*/}" >/dev/null 2>&1; then
        info "${pkg##*/} already installed (venv314)"
        return 0
    fi

    info "Installing into venv314: $pkg"

    TMPDIR="$TMPDIR" \
    TMP="$TMP" \
    TEMP="$TEMP" \
    PIP_NO_CACHE_DIR=1 \
    "$VENV314_DIR/bin/python" -m pip install --no-cache-dir "$pkg"
}

download_to_usr_local() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [[ -f "$dest" ]]; then
        info "$name already exists at $dest"
        return 0
    fi

    local tmp_file="$TMPDIR/$(basename "$dest").download"

    info "Downloading $name..."
    info "  URL: $url"

    rm -f "$tmp_file"

    curl -fL --retry 3 --retry-delay 2 \
        "$url" \
        -o "$tmp_file"

    sudo install -m 0755 "$tmp_file" "$dest"
    rm -f "$tmp_file"

    info "$name installed at $dest"
}

download_github_release_asset() {
    local api_url="$1"
    local asset_pattern="$2"
    local dest="$3"
    local name="$4"

    if [[ -f "$dest" ]]; then
        info "$name already exists at $dest"
        return 0
    fi

    local json_file="$TMPDIR/${name}.release.json"
    local asset_url=""

    info "Resolving latest $name release..."
    curl -fsSL --retry 3 --retry-delay 2 \
        "$api_url" \
        -o "$json_file" || {
        warn "GitHub API request failed for $name (possibly rate-limited)."
        rm -f "$json_file"
        return 1
    }

    asset_url="$(
        "$PYTHON_BIN" - "$json_file" "$asset_pattern" <<'PY'
import json
import re
import sys

json_file = sys.argv[1]
pattern = sys.argv[2]

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for asset in data.get("assets", []):
    name = asset.get("name", "")
    url = asset.get("browser_download_url", "")
    if re.search(pattern, name, re.IGNORECASE):
        print(url)
        break
PY
    )"

    if [[ -z "$asset_url" ]]; then
        rm -f "$json_file"
        return 1
    fi

    download_to_usr_local "$asset_url" "$dest" "$name"
    rm -f "$json_file"
}

# ============================================================
# Apt update
# ============================================================

info "===== Updating APT ====="
sudo apt update -qq

# ============================================================
# Build dependencies / runtimes
# ============================================================

info "===== Installing prerequisites ====="

sudo apt install -y -qq \
    ca-certificates \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libpcap-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    python3-dev \
    python3-pip \
    ruby \
    ruby-dev \
    cargo \
    rustc \
    wine64 \
    unzip \
    tar \
    gzip

info "Prerequisites installed."
echo ""

# ============================================================
# Go
# ============================================================

info "===== Go ====="

GO_VERSION="1.25.0"

if check_cmd go; then
    info "Go already installed: $(go version)"
else
    GO_TARBALL="go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
    GO_INSTALL_DIR="$TMPDIR/go-install"

    mkdir -p "$GO_INSTALL_DIR"
    cd "$GO_INSTALL_DIR"

    info "Installing Go ${GO_VERSION}..."
    curl -fL --retry 3 --retry-delay 2 \
        "https://go.dev/dl/${GO_TARBALL}" \
        -o "$GO_TARBALL"

    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$GO_TARBALL"
    rm -f "$GO_TARBALL"

    if ! grep -q '/usr/local/go/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="/usr/local/go/bin:$HOME/go/bin:$HOME/.cargo/bin:$HOME/.gem/bin:$PATH"' >> "$HOME/.bashrc"
    fi

    export PATH="/usr/local/go/bin:$HOME/go/bin:$CARGO_HOME/bin:$GEM_HOME/bin:$PATH"

    if ! check_cmd go; then
        warn "Go installation failed."
        warn "Go-based tools will be reported as missing."
    else
        info "Go installed: $(go version)"
    fi

fi

export PATH="/usr/local/go/bin:$HOME/go/bin:$CARGO_HOME/bin:$GEM_HOME/bin:$PATH"

if ! check_cmd go; then
    warn "Go command is not available. Go-based tools will be reported as missing."
fi
echo ""

# ============================================================
# 1. Metasploit Framework (official Rapid7 installer)
# ============================================================

CURRENT_TOOL="metasploit-framework"
info "[1/22] metasploit-framework"

if check_cmd msfconsole; then
    info "metasploit-framework already installed"
else
    info "Installing metasploit-framework via official Rapid7 installer..."
    curl -fL --retry 3 --retry-delay 2 \
        https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb \
        -o "$TMPDIR/msfinstall" \
    && chmod 755 "$TMPDIR/msfinstall" \
    && sudo "$TMPDIR/msfinstall" \
    || warn "metasploit-framework installer failed. Retry manually later."
fi

# ============================================================
# 2. Impacket (+ Kali-style impacket-* aliases)
# ============================================================

CURRENT_TOOL="impacket"
info "[2/22] impacket"
install_pip impacket

if [[ -n "$VENV_BIN_DIR" && -x "$VENV_BIN_DIR/smbexec.py" ]]; then
    info "Creating impacket-* aliases in $VENV_BIN_DIR ..."
    for f in "$VENV_BIN_DIR/"*.py; do
        [[ -e "$f" ]] || continue
        base="$(basename "$f" .py)"
        ln -sf "$f" "$VENV_BIN_DIR/impacket-${base}"
    done
    hash -r
    info "impacket-smbexec and other impacket-* commands are now available."
else
    warn "Could not create impacket aliases (venv bin not found or smbexec.py missing)."
fi

# ============================================================
# 3. NetExec — venv314 (new releases reject Python 3.12)
# ============================================================

CURRENT_TOOL="netexec"
info "[3/22] netexec"

if check_cmd netexec || [[ -f /usr/local/bin/netexec ]]; then
    info "netexec already installed"
else
    info "Recent NetExec requires newer Python than the main venv (3.12)."
    if ! install_pip314 netexec; then
        warn "PyPI netexec failed in venv314; trying GitHub source..."
        install_pip314 "git+https://github.com/Pennyw0rth/NetExec" \
            || warn "netexec installation failed."
    fi

    if [[ -x "$VENV314_DIR/bin/netexec" ]]; then
        sudo tee /usr/local/bin/netexec >/dev/null <<'EOF'
#!/usr/bin/env bash
exec "$HOME/VAPT-AI/venv314/bin/netexec" "$@"
EOF
        sudo chmod 0755 /usr/local/bin/netexec
        info "Wrapper created: /usr/local/bin/netexec"
    fi

    if [[ -x "$VENV314_DIR/bin/nxc" ]]; then
        sudo tee /usr/local/bin/nxc >/dev/null <<'EOF'
#!/usr/bin/env bash
exec "$HOME/VAPT-AI/venv314/bin/nxc" "$@"
EOF
        sudo chmod 0755 /usr/local/bin/nxc
        info "Wrapper created: /usr/local/bin/nxc"
    fi
fi

# ============================================================
# 4. Responder (official GitHub — PyPI 'responder' is fake)
# ============================================================

CURRENT_TOOL="responder"
info "[4/22] responder"

if check_cmd responder; then
    info "responder already installed"
else
    info "Installing Responder from official GitHub (lgandx/Responder)..."
    if [[ ! -d "$HOME/VAPT-AI/tools/Responder" ]]; then
        git clone --depth 1 https://github.com/lgandx/Responder.git "$HOME/VAPT-AI/tools/Responder"
    else
        info "Responder repo already cloned, updating..."
        git -C "$HOME/VAPT-AI/tools/Responder" pull --ff-only 2>/dev/null || true
    fi

    echo '#!/usr/bin/env bash'            | sudo tee /usr/local/bin/responder >/dev/null
    echo "exec ${PYTHON_BIN} \"$HOME/VAPT-AI/tools/Responder/Responder.py\" \"\$@\"" \
                                         | sudo tee -a /usr/local/bin/responder >/dev/null
    sudo chmod 0755 /usr/local/bin/responder

    if check_cmd responder; then
        info "responder installed (wrapper at /usr/local/bin/responder)"
    else
        warn "responder wrapper installation failed."
    fi
fi

# ============================================================
# 5. Hashcat
# ============================================================

CURRENT_TOOL="hashcat"
info "[5/22] hashcat"
install_apt hashcat

# ============================================================
# 6. John the Ripper
# ============================================================

CURRENT_TOOL="john"
info "[6/22] john"
install_apt john

# ============================================================
# 7. Hydra
# ============================================================

CURRENT_TOOL="hydra"
info "[7/22] hydra"
install_apt hydra

# ============================================================
# 8. LinPEAS
# ============================================================

CURRENT_TOOL="linpeas"
info "[8/22] linpeas"

download_to_usr_local \
    "https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh" \
    "/usr/local/bin/linpeas" \
    "linpeas"

# ============================================================
# 9. WinPEAS
# ============================================================

CURRENT_TOOL="winpeas"
info "[9/22] winpeas"

download_to_usr_local \
    "https://github.com/peass-ng/PEASS-ng/releases/latest/download/winPEASx64.exe" \
    "/usr/local/bin/winpeas.exe" \
    "winpeas"

# ============================================================
# 10. Mimikatz
#     PATCH: releases only ship mimikatz_trunk.zip. A standalone
#     mimikatz.exe asset does NOT exist (previous URL was 404).
# ============================================================

CURRENT_TOOL="mimikatz"
info "[10/22] mimikatz"

MIMIKATZ_DIR="$HOME/VAPT-AI/tools/mimikatz"

if [[ -f /usr/local/bin/mimikatz || -f "$MIMIKATZ_DIR/x64/mimikatz.exe" ]]; then
    info "mimikatz already installed"
else
    info "Downloading mimikatz_trunk.zip..."
    ZIP_FILE="$TMPDIR/mimikatz_trunk.zip"
    if curl -fL --retry 3 --retry-delay 2 \
        "https://github.com/gentilkiwi/mimikatz/releases/latest/download/mimikatz_trunk.zip" \
        -o "$ZIP_FILE"; then
        mkdir -p "$MIMIKATZ_DIR"
        unzip -o -q "$ZIP_FILE" -d "$MIMIKATZ_DIR"
        rm -f "$ZIP_FILE"
    else
        warn "mimikatz download failed."
    fi
fi

if [[ -f "$MIMIKATZ_DIR/x64/mimikatz.exe" ]]; then
    sudo tee /usr/local/bin/mimikatz >/dev/null <<'EOF'
#!/usr/bin/env bash
exec wine "$HOME/VAPT-AI/tools/mimikatz/x64/mimikatz.exe" "$@"
EOF
    sudo chmod 0755 /usr/local/bin/mimikatz
    info "mimikatz wrapper created (runs via wine)."
else
    warn "mimikatz.exe not found after download."
fi

if check_cmd wine || check_cmd wine64; then
    info "Wine is available for mimikatz."
else
    warn "Wine is not available. mimikatz.exe was downloaded but cannot run on this VM yet."
fi

# ============================================================
# 11. Masscan
# ============================================================

CURRENT_TOOL="masscan"
info "[11/22] masscan"
install_apt masscan

# ============================================================
# 12. RustScan
# ============================================================

CURRENT_TOOL="rustscan"
info "[12/22] rustscan"

if check_cmd rustscan || [[ -x "$HOME/.cargo/bin/rustscan" ]]; then
    info "rustscan already installed"
else
    info "Trying Cargo installation..."
    (
        export TMPDIR="$TMPDIR"
        export TMP="$TMP"
        export TEMP="$TEMP"
        export CARGO_HOME="$CARGO_HOME"
        export RUSTUP_HOME="$RUSTUP_HOME"
        export CARGO_TARGET_DIR="$TMPDIR/rustscan-target"
        mkdir -p "$CARGO_TARGET_DIR"
        cargo install rustscan
    ) || {
        warn "cargo install rustscan failed."
        warn "Trying GitHub release binary..."

        RUSTSCAN_API="https://api.github.com/repos/RustScan/RustScan/releases/latest"

        if [[ "$ARCH" == "x86_64" ]]; then
            RUSTSCAN_PATTERN='x86_64.*linux|amd64.*linux'
        else
            RUSTSCAN_PATTERN='aarch64.*linux|arm64.*linux'
        fi

        if ! download_github_release_asset \
            "$RUSTSCAN_API" \
            "$RUSTSCAN_PATTERN" \
            "/usr/local/bin/rustscan" \
            "rustscan"; then
            warn "RustScan automatic binary download failed."
            warn "RustScan will be retried if this script is run again."
        fi
    }
fi

# ============================================================
# 13. fscan
#     PATCH: broader asset pattern + source build fallback.
# ============================================================

CURRENT_TOOL="fscan"
info "[13/22] fscan"

if check_cmd fscan || [[ -x "$HOME/go/bin/fscan" ]]; then
    info "fscan already installed"
else
    if [[ "$ARCH" == "x86_64" ]]; then
        FS_PATTERN='(linux.*(amd64|x86_64)|(amd64|x86_64).*linux)'
    else
        FS_PATTERN='(linux.*(arm64|aarch64)|(arm64|aarch64).*linux)'
    fi

    if ! download_github_release_asset \
        "https://api.github.com/repos/shadow1ng/fscan/releases/latest" \
        "$FS_PATTERN" \
        "/usr/local/bin/fscan" \
        "fscan"; then
        warn "No matching fscan release asset (asset names changed or API rate-limited)."
        warn "Building fscan from source with Go..."

        (
            export GOPATH="$GOPATH"
            export GOCACHE="$GOCACHE"
            export GOMODCACHE="$GOMODCACHE"
            export TMPDIR="$TMPDIR" TMP="$TMP" TEMP="$TEMP"
            export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"

            FS_SRC="$HOME/VAPT-AI/tools/fscan"
            rm -rf "$FS_SRC"
            git clone --depth 1 https://github.com/shadow1ng/fscan.git "$FS_SRC"
            cd "$FS_SRC"
            go build -o "$HOME/go/bin/fscan" .
        )

        if [[ -x "$HOME/go/bin/fscan" ]]; then
            info "fscan built from source: $HOME/go/bin/fscan"
        else
            warn "fscan build failed. Install manually later."
        fi
    fi
fi

# ============================================================
# 14. ffuf
# ============================================================

CURRENT_TOOL="ffuf"
info "[14/22] ffuf"
install_go_tool "github.com/ffuf/ffuf/v2" "ffuf"

# ============================================================
# 15. Katana
# ============================================================

CURRENT_TOOL="katana"
info "[15/22] katana"
install_go_tool "github.com/projectdiscovery/katana/cmd/katana" "katana"

# ============================================================
# 16. gau
# ============================================================

CURRENT_TOOL="gau"
info "[16/22] gau"
install_go_tool "github.com/lc/gau/v2/cmd/gau" "gau"

# ============================================================
# 17. waybackurls
# ============================================================

CURRENT_TOOL="waybackurls"
info "[17/22] waybackurls"
install_go_tool "github.com/tomnomnom/waybackurls" "waybackurls"

# ============================================================
# 18. Amass
# ============================================================

CURRENT_TOOL="amass"
info "[18/22] amass"

if check_cmd amass; then
    info "amass already installed"
elif install_apt_if_available amass; then
    :
else
    install_go_tool "github.com/owasp-amass/amass/v4/cmd/amass" "amass" || \
        warn "amass installation failed."
fi

# ============================================================
# 19. dnsenum
# ============================================================

CURRENT_TOOL="dnsenum"
info "[19/22] dnsenum"
install_apt dnsenum

# ============================================================
# 20. fierce
# ============================================================

CURRENT_TOOL="fierce"
info "[20/22] fierce"
install_apt fierce

# ============================================================
# 21. theHarvester — venv314 (latest requires Python >= 3.14)
# ============================================================

CURRENT_TOOL="theHarvester"
info "[21/22] theHarvester"

if check_cmd theHarvester || check_cmd theharvester || [[ -f /usr/local/bin/theHarvester ]]; then
    info "theHarvester already installed"
else
    # Remove the fake PyPI placeholder from the MAIN venv if present.
    if "$PYTHON_BIN" -m pip show theHarvester >/dev/null 2>&1; then
        FAKE_VERSION="$("$PYTHON_BIN" -m pip show theHarvester 2>/dev/null | awk '/^Version:/ {print $2}')"
        if [[ "$FAKE_VERSION" == "0.0.1" ]]; then
            warn "Removing fake PyPI 'theHarvester' 0.0.1 from main venv..."
            "$PYTHON_BIN" -m pip uninstall -y theHarvester
        fi
    fi

    warn "Latest theHarvester requires Python >= 3.14."
    install_pip314 "git+https://github.com/laramies/theHarvester" \
        || warn "theHarvester installation failed."

    if [[ -x "$VENV314_DIR/bin/theHarvester" ]]; then
        sudo tee /usr/local/bin/theHarvester >/dev/null <<'EOF'
#!/usr/bin/env bash
exec "$HOME/VAPT-AI/venv314/bin/theHarvester" "$@"
EOF
        sudo chmod 0755 /usr/local/bin/theHarvester
        info "Wrapper created: /usr/local/bin/theHarvester"
    fi
fi

# ============================================================
# 22. WPScan
# ============================================================

CURRENT_TOOL="wpscan"
info "[22/22] wpscan"

if check_cmd wpscan; then
    info "wpscan already installed"
else
    export GEM_HOME="$HOME/.gem"
    export GEM_PATH="$GEM_HOME"
    export PATH="$GEM_HOME/bin:$PATH"

    if command -v gem >/dev/null 2>&1; then
        info "Installing wpscan into user gem directory..."
        TMPDIR="$TMPDIR" TMP="$TMP" TEMP="$TEMP" \
            gem install --no-document wpscan \
        || warn "wpscan gem installation failed."
    else
        warn "Ruby/gem is not available; WPScan requires manual installation."
    fi
fi

# ============================================================
# Clean package/download caches
# ============================================================

info "===== Cleanup ====="

cleanup_tmp

sudo apt clean
sudo apt autoclean -y

sudo rm -rf /var/lib/apt/lists/*

rm -rf "$HOME/.cache/pip" 2>/dev/null || true

rm -rf "$CARGO_HOME/registry/cache" 2>/dev/null || true
rm -rf "$CARGO_HOME/registry/src" 2>/dev/null || true

rm -rf "$TMPDIR/rustscan-target" 2>/dev/null || true

# ============================================================
# Verification
# ============================================================

echo ""
info "===== Verification ====="
echo ""

declare -A TOOL_CHECKS=(
    ["msfconsole"]="msfconsole"
    ["impacket"]="impacket-smbexec"
    ["netexec"]="netexec"
    ["responder"]="responder"
    ["hashcat"]="hashcat"
    ["john"]="john"
    ["hydra"]="hydra"
    ["linpeas"]="linpeas"
    ["winpeas"]="winpeas.exe"
    ["mimikatz"]="mimikatz"
    ["masscan"]="masscan"
    ["rustscan"]="rustscan"
    ["fscan"]="fscan"
    ["ffuf"]="ffuf"
    ["katana"]="katana"
    ["gau"]="gau"
    ["waybackurls"]="waybackurls"
    ["amass"]="amass"
    ["dnsenum"]="dnsenum"
    ["fierce"]="fierce"
    ["theHarvester"]="theHarvester"
    ["wpscan"]="wpscan"
)

INSTALLED=0
FAILED=0

for label in "${!TOOL_CHECKS[@]}"; do
    cmd="${TOOL_CHECKS[$label]}"

    if check_cmd "$cmd" || [[ -f "/usr/local/bin/$cmd" ]]; then
        echo -e "  ${GREEN}✓${NC} $label -> $cmd"
        INSTALLED=$((INSTALLED + 1))
    else
        echo -e "  ${RED}✗${NC} $label -> NOT FOUND"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
info "Installed/available: $INSTALLED/22"
info "Missing: $FAILED/22"

if [[ "$FAILED" -gt 0 ]]; then
    echo ""
    warn "===== Tools Requiring Manual Installation ====="
    for label in "${!TOOL_CHECKS[@]}"; do
        cmd="${TOOL_CHECKS[$label]}"
        if ! check_cmd "$cmd" && [[ ! -f "/usr/local/bin/$cmd" ]]; then
            warn "  - $label (expected command/file: $cmd)"
        fi
    done
fi

# ============================================================
# Storage report
# ============================================================

echo ""
info "===== Storage After Installation ====="
echo ""

info "Root filesystem:"
df -h /

echo ""
info "Home filesystem:"
df -h "$HOME"

echo ""
info "Inode usage:"
df -ih /

echo ""
info "Our temporary directory:"
du -sh "$TMPDIR" 2>/dev/null || true

echo ""
info "Go build cache:"
du -sh "$GOCACHE" 2>/dev/null || true

echo ""
info "Go module cache:"
du -sh "$GOMODCACHE" 2>/dev/null || true

echo ""
info "Cargo:"
du -sh "$CARGO_HOME" 2>/dev/null || true

echo ""
info "Ruby gems:"
du -sh "$GEM_HOME" 2>/dev/null || true

# ============================================================
# Final result
# ============================================================

echo ""

if [[ "$FAILED" -gt 0 ]]; then
    warn "============================================================"
    warn "Installation completed, but $FAILED tool(s) are missing."
    warn "These tools did NOT stop the installer."
    warn "Install the missing tools manually, then re-run this script."
    warn "Already-installed tools will be skipped."
    warn "Run: source ~/.bashrc"
    warn "============================================================"
else
    info "${GREEN}All 22 extended tools are available!${NC}"
fi

echo ""
info "Next steps:"
info "  1. Run: source ~/.bashrc"
info "  2. Verify dual-Python tools:"
info "       netexec --version        (venv314 / Python 3.14)"
info "       theHarvester --help      (venv314 / Python 3.14)"
info "       impacket-smbexec --help  (venv / Python 3.12)"
info "  3. Verify Go tools:"
info "       ffuf -V && katana -version"
info "  4. W5-B can create wrappers/configuration for these tools."