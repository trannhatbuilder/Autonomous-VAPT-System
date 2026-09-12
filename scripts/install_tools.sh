#!/usr/bin/env bash
# ============================================================
# VAPT-AI v3.2 — Security Tools Installer (W1 — 10 core tools)
# ============================================================
#
# Installs 10 core security tools needed for W1-W4.
#
# Tools installed:
#   1. nmap          — network scanner             (apt)
#   2. nuclei        — vulnerability scanner       (go install)
#   3. sqlmap        — SQL injection               (pip / apt fallback)
#   4. gobuster      — directory brute-force       (apt)
#   5. feroxbuster   — recursive content discovery (apt / cargo)
#   6. subfinder     — subdomain enumeration       (go install)
#   7. httpx         — HTTP probe                  (go install)
#   8. whatweb       — web technology fingerprint  (apt)
#   9. nikto         — web server scanner          (apt)
#  10. dalfox        — XSS scanner                 (go install)
#
# Also installs:
#   - Go 1.25.0 if Go is not already installed
#   - Required build dependencies
#
# IMPORTANT:
#   Ubuntu 26.04 VPS may have a small /tmp tmpfs.
#   Large Go projects such as Nuclei can exceed /tmp.
#
#   Therefore this script redirects temporary/build storage to:
#
#       $HOME/tmp
#
#   instead of:
#
#       /tmp
#
# Usage:
#   chmod +x scripts/install_tools.sh
#   ./scripts/install_tools.sh
#
# Re-run safe:
#   Already installed tools are skipped.
#
# Target:
#   Ubuntu 26.04 LTS
#   x86_64 / amd64
#
# ============================================================

set -euo pipefail

# ============================================================
# IMPORTANT: Redirect temporary files away from /tmp
# ============================================================
#
# Ubuntu VPS may mount /tmp as a small tmpfs (for example 1.7G).
# Go compilation of Nuclei can exceed this limit and produce:
#
#   write /tmp/go-build...: disk quota exceeded
#
# Use the root filesystem under the user's home directory instead.
# ============================================================

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"

mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# ============================================================
# Go cache / module cache
# ============================================================

export GOPATH="$HOME/go"
export GOCACHE="$HOME/.cache/go-build"
export GOMODCACHE="$HOME/go/pkg/mod"

mkdir -p "$GOPATH/bin"
mkdir -p "$GOCACHE"
mkdir -p "$GOMODCACHE"

# ============================================================
# Colors
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# Helper functions
# ============================================================

info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

check_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================
# APT package installer
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

    info "$pkg installed successfully."
}

# ============================================================
# Go tool installer
# ============================================================

install_go_tool() {
    local pkg_path="$1"
    local bin_name="$2"

    if check_cmd "$bin_name"; then
        info "$bin_name already installed (go)"
        return 0
    fi

    info "Installing $bin_name..."
    info "go install $pkg_path@latest"

    TMPDIR="$TMPDIR" \
    TMP="$TMP" \
    TEMP="$TEMP" \
    GOPATH="$GOPATH" \
    GOCACHE="$GOCACHE" \
    GOMODCACHE="$GOMODCACHE" \
    go install "$pkg_path@latest"

    if [ -x "$HOME/go/bin/$bin_name" ]; then
        info "$bin_name installed successfully: $HOME/go/bin/$bin_name"
    elif check_cmd "$bin_name"; then
        info "$bin_name installed successfully."
    else
        error "$bin_name installation failed."
        return 1
    fi
}

# ============================================================
# Python package installer
# ============================================================

install_python_package() {
    local pkg="$1"

    if python -m pip show "$pkg" >/dev/null 2>&1; then
        info "$pkg already installed in current Python environment."
        return 0
    fi

    info "Installing $pkg into current Python environment..."

    python -m pip install --no-cache-dir "$pkg"
}

# ============================================================
# Header
# ============================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}VAPT-AI v3.2 — Security Tools Installer (W1 — 10 core)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================
# System information
# ============================================================

info "Operating system:"
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "  ${PRETTY_NAME:-Unknown}"
fi

ARCH=$(uname -m)

case "$ARCH" in
    x86_64)
        GO_ARCH="amd64"
        ;;
    aarch64|arm64)
        GO_ARCH="arm64"
        ;;
    *)
        error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

info "Architecture: $ARCH"
info "Go architecture: $GO_ARCH"

echo ""

# ============================================================
# Python / venv check
# ============================================================

if [ -n "${VIRTUAL_ENV:-}" ]; then
    info "Active Python virtual environment: $VIRTUAL_ENV"
else
    warn "Python virtual environment is not active."
    warn "Recommended:"
    warn "  source ~/VAPT-AI/venv/bin/activate"
fi

if check_cmd python; then
    info "Python version: $(python --version 2>&1)"
    info "Python path: $(command -v python)"
else
    error "python command not found."
    exit 1
fi

echo ""

# ============================================================
# Temporary storage information
# ============================================================

info "Temporary directory configuration:"
echo "  TMPDIR    = $TMPDIR"
echo "  TMP       = $TMP"
echo "  TEMP      = $TEMP"
echo "  GOCACHE   = $GOCACHE"
echo "  GOMODCACHE= $GOMODCACHE"
echo "  GOPATH    = $GOPATH"

echo ""

info "Filesystem status:"
df -h "$HOME"

echo ""

# ============================================================
# Update APT
# ============================================================

info "Updating apt package index..."
sudo apt update -qq

# ============================================================
# Build dependencies
# ============================================================

info "Installing build dependencies..."

sudo apt install -y -qq \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libpcap-dev \
    python3-dev \
    python3-pip

info "Build dependencies installed."

# ============================================================
# Go installation
# ============================================================

echo ""
info "===== Go ====="

GO_VERSION="1.25.0"

if check_cmd go; then

    CURRENT_GO_VERSION=$(go version | awk '{print $3}' | sed 's/^go//')

    info "Go already installed: $CURRENT_GO_VERSION"

else

    info "Go not found."
    info "Installing Go ${GO_VERSION}..."

    GO_TARBALL="go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
    GO_URL="https://go.dev/dl/${GO_TARBALL}"
    GO_INSTALL_DIR="${HOME}/tmp/go-install"

    mkdir -p "$GO_INSTALL_DIR"

    cd "$GO_INSTALL_DIR"

    info "Downloading:"
    info "$GO_URL"

    rm -f "$GO_TARBALL"

    wget -q --show-progress \
        "$GO_URL" \
        -O "$GO_TARBALL"

    if [ ! -s "$GO_TARBALL" ]; then
        error "Failed to download Go."
        exit 1
    fi

    info "Installing Go to /usr/local/go..."

    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$GO_TARBALL"

    rm -f "$GO_TARBALL"

    export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"

    if ! grep -q '/usr/local/go/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"' >> "$HOME/.bashrc"
        info "Go PATH added to ~/.bashrc"
    fi

    if ! check_cmd go; then
        error "Go installation failed."
        exit 1
    fi

    info "Go installed successfully:"
    go version
fi

# ============================================================
# Ensure Go PATH
# ============================================================

export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"

if ! check_cmd go; then
    error "Go command is not available."
    exit 1
fi

info "Go version: $(go version)"

# ============================================================
# 10 Core Security Tools
# ============================================================

echo ""
info "===== Installing 10 Core Security Tools ====="
echo ""

# ------------------------------------------------------------
# 1. nmap
# ------------------------------------------------------------

info "[1/10] nmap"

install_apt nmap

# ------------------------------------------------------------
# 2. nuclei
# ------------------------------------------------------------

info "[2/10] nuclei"

install_go_tool \
    "github.com/projectdiscovery/nuclei/v3/cmd/nuclei" \
    "nuclei"

# ------------------------------------------------------------
# 3. sqlmap
# ------------------------------------------------------------

info "[3/10] sqlmap"

#
# Do NOT use:
#
#   pip install --user sqlmap
#
# because the script is intended to run inside the VAPT-AI
# virtual environment.
#
# Install into the active Python environment instead.
#

if check_cmd sqlmap; then

    info "sqlmap already installed."

else

    if python -m pip show sqlmap >/dev/null 2>&1; then

        info "sqlmap Python package already installed."

    else

        info "Installing sqlmap using Python pip..."

        python -m pip install --no-cache-dir sqlmap

    fi

fi

# If pip installation did not create a command, try apt.
if ! check_cmd sqlmap; then

    warn "sqlmap command not found after pip installation."
    warn "Trying apt installation..."

    install_apt sqlmap

fi

# ------------------------------------------------------------
# 4. gobuster
# ------------------------------------------------------------

info "[4/10] gobuster"

install_apt gobuster

# ------------------------------------------------------------
# 5. feroxbuster
# ------------------------------------------------------------

info "[5/10] feroxbuster"

if check_cmd feroxbuster; then
    info "feroxbuster already installed"
else
    info "feroxbuster not found"

    FERox_TMP_DIR="${TMPDIR}/feroxbuster-install"
    FERox_API="https://api.github.com/repos/epi052/feroxbuster/releases/latest"

    mkdir -p "$FERox_TMP_DIR"

    info "Checking latest feroxbuster release..."

    FERox_RELEASE_JSON=$(
        curl -fsSL "$FERox_API"
    )

    FERox_TAG=$(
        printf '%s' "$FERox_RELEASE_JSON" |
        grep -m1 '"tag_name":' |
        sed -E 's/.*"tag_name": "([^"]+)".*/\1/'
    )

    if [ -z "$FERox_TAG" ]; then
        error "Could not determine latest feroxbuster release."
        exit 1
    fi

    info "Latest feroxbuster release: $FERox_TAG"

    # Find x86_64 Linux tarball
    FERox_URL=$(
        printf '%s' "$FERox_RELEASE_JSON" |
        grep '"browser_download_url":' |
        grep -Ei 'x86_64|amd64' |
        grep -Ei '\.tar\.gz"' |
        sed -E 's/.*"browser_download_url": "([^"]+)".*/\1/' |
        head -1
    )

    if [ -z "$FERox_URL" ]; then
        error "Could not find x86_64 Linux feroxbuster release asset."
        exit 1
    fi

    info "Download URL:"
    info "$FERox_URL"

    cd "$FERox_TMP_DIR"

    FERox_ARCHIVE=$(basename "$FERox_URL")

    rm -f "$FERox_ARCHIVE"

    curl -fL \
        "$FERox_URL" \
        -o "$FERox_ARCHIVE"

    info "Extracting feroxbuster..."

    tar -xzf "$FERox_ARCHIVE"

    FERox_BINARY=$(find "$FERox_TMP_DIR" \
        -type f \
        -name "feroxbuster" \
        | head -1)

    if [ -z "${FERox_BINARY:-}" ] || [ ! -f "$FERox_BINARY" ]; then
        error "feroxbuster binary not found after extraction."
        exit 1
    fi

    chmod +x "$FERox_BINARY"

    sudo install \
        -m 0755 \
        "$FERox_BINARY" \
        /usr/local/bin/feroxbuster

    rm -rf "$FERox_TMP_DIR"

    if check_cmd feroxbuster; then
        info "feroxbuster installed successfully."
    else
        error "feroxbuster installation failed."
        exit 1
    fi
fi

# ------------------------------------------------------------
# 6. subfinder
# ------------------------------------------------------------

info "[6/10] subfinder"

install_go_tool \
    "github.com/projectdiscovery/subfinder/v2/cmd/subfinder" \
    "subfinder"

# ------------------------------------------------------------
# 7. httpx — ProjectDiscovery
# ------------------------------------------------------------

info "[7/10] httpx (ProjectDiscovery)"

install_go_tool \
    "github.com/projectdiscovery/httpx/cmd/httpx" \
    "httpx"

# ------------------------------------------------------------
# 8. whatweb
# ------------------------------------------------------------

info "[8/10] whatweb"

install_apt whatweb

# ------------------------------------------------------------
# 9. nikto
# ------------------------------------------------------------

info "[9/10] nikto"

install_apt nikto

# ------------------------------------------------------------
# 10. dalfox
# ------------------------------------------------------------

info "[10/10] dalfox"

install_go_tool \
    "github.com/hahwul/dalfox/v2" \
    "dalfox"

# ============================================================
# Verification
# ============================================================

echo ""
info "===== Verification ====="
echo ""

TOOLS=(
    "nmap"
    "nuclei"
    "sqlmap"
    "gobuster"
    "feroxbuster"
    "subfinder"
    "httpx"
    "whatweb"
    "nikto"
    "dalfox"
)

INSTALLED=0
FAILED=0

for tool in "${TOOLS[@]}"; do

    if check_cmd "$tool"; then

        VERSION="version-unknown"

        case "$tool" in

            nmap)
                VERSION=$(
                    "$tool" --version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            nuclei)
                VERSION=$(
                    "$tool" -version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            sqlmap)
                VERSION=$(
                    sqlmap --version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            gobuster)
                VERSION=$(
                    "$tool" version 2>&1 \
                    | head -1 \
                    || "$tool" --version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            feroxbuster)
                VERSION=$(
                    "$tool" --version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            subfinder)
                VERSION=$(
                    "$tool" -version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            httpx)
                VERSION=$(
                    "$tool" -version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            whatweb)
                VERSION=$(
                    "$tool" --version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            nikto)
                VERSION=$(
                    "$tool" -Version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

            dalfox)
                VERSION=$(
                    "$tool" version 2>&1 \
                    | head -1 \
                    || true
                )
                ;;

        esac

        echo -e "  ${GREEN}✓${NC} $tool: $VERSION"

        INSTALLED=$((INSTALLED + 1))

    else

        echo -e "  ${RED}✗${NC} $tool: NOT FOUND"

        FAILED=$((FAILED + 1))

    fi

done

# ============================================================
# Storage verification
# ============================================================

echo ""
info "===== Storage ====="
echo ""

info "Root filesystem:"
df -h /

echo ""

info "Temporary build directory:"
df -h "$TMPDIR"

echo ""

info "Temporary directory usage:"
du -sh "$TMPDIR" 2>/dev/null || true

echo ""

info "Go build cache:"
du -sh "$GOCACHE" 2>/dev/null || true

echo ""

info "Go module cache:"
du -sh "$GOMODCACHE" 2>/dev/null || true

# ============================================================
# Final result
# ============================================================

echo ""
info "===== Installation Result ====="
echo ""

info "Installed: $INSTALLED/10"

if [ "$FAILED" -gt 0 ]; then

    warn "Failed or unavailable: $FAILED/10"
    warn ""
    warn "Some tools may require:"
    warn "  - PATH refresh: source ~/.bashrc"
    warn "  - manual installation"
    warn "  - re-running this script"
    warn ""
    warn "The installer is safe to re-run."

    exit 1

fi

info "${GREEN}All 10 core tools installed successfully!${NC}"

echo ""

info "Next steps:"
info "  1. Run: source ~/.bashrc"
info "  2. Verify:"
info "       nuclei -version"
info "       subfinder -version"
info "       httpx -version"
info "       dalfox version"
info "  3. W2 can create MCP wrappers for these 10 tools."
info "  4. W5 can install extended tools separately."

echo ""