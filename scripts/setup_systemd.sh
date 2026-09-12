#!/usr/bin/env bash
# ============================================================
# VAPT-AI v3.2 — systemd services installer
# ============================================================
#
# Installs + enables + starts all 4 systemd services:
#   1. vapt-ai-app          — FastAPI backend
#   2. vapt-ai-celery-worker — background scan tasks
#   3. vapt-ai-celery-beat   — scheduled jobs
#   4. msfrpcd              — Metasploit RPC daemon
#
# Prerequisites:
#   - PostgreSQL running (see docs/POSTGRESQL_SETUP.md)
#   - Redis running (sudo systemctl start redis-server)
#   - VAPT-AI cloned to ~/VAPT-AI with venv + .env
#   - alembic upgrade head already run
#   - Metasploit Framework installed (apt install metasploit-framework) — W6 task
#
# Usage:
#   chmod +x scripts/setup_systemd.sh
#   ./scripts/setup_systemd.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}VAPT-AI v3.2 — systemd Services Installer${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ---------- Pre-flight checks ----------

info "Pre-flight checks..."

# Check script is run from VAPT-AI root
if [ ! -f "app/main.py" ]; then
    error "Must run from VAPT-AI project root (where app/main.py is)"
    exit 1
fi

PROJECT_ROOT="$(pwd)"
info "Project root: $PROJECT_ROOT"

# Check .env exists
if [ ! -f ".env" ]; then
    error ".env not found. Copy .env.example to .env and fill in values first."
    exit 1
fi

# Check venv exists
if [ ! -d "venv" ]; then
    error "venv/ not found. Create it first: python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check PostgreSQL running
if ! systemctl is-active --quiet postgresql; then
    warn "PostgreSQL not running. Start it: sudo systemctl start postgresql"
    warn "Continuing anyway — services will fail to start until PostgreSQL is up."
fi

# Check Redis running
if ! systemctl is-active --quiet redis-server; then
    warn "Redis not running. Starting it..."
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
fi

# Get current user (services will run as this user)
CURRENT_USER="$(whoami)"
info "Services will run as user: $CURRENT_USER"

# ---------- Update systemd unit files with current user ----------

info "Updating systemd unit files with current user ($CURRENT_USER)..."

SYSTEMD_DIR="scripts/systemd"
TEMP_DIR="$(mktemp -d)"
trap "rm -rf $TEMP_DIR" EXIT

for unit_file in "$SYSTEMD_DIR"/*.service; do
    unit_name="$(basename "$unit_file")"
    temp_unit="$TEMP_DIR/$unit_name"

    # Replace placeholder 'nhat' with current user
    sed "s/User=nhat/User=$CURRENT_USER/g; s/Group=nhat/Group=$CURRENT_USER/g; s|/home/nhat|/home/$CURRENT_USER|g" \
        "$unit_file" > "$temp_unit"

    # Skip msfrpcd if Metasploit not installed
    if [ "$unit_name" = "msfrpcd.service" ]; then
        if ! command -v msfrpcd >/dev/null 2>&1; then
            warn "msfrpcd not found — skipping msfrpcd.service"
            warn "Install Metasploit Framework first (W6 task): sudo apt install metasploit-framework"
            continue
        fi
    fi

    info "Installing $unit_name..."
    sudo cp "$temp_unit" "/etc/systemd/system/$unit_name"
done

# ---------- Reload + enable + start ----------

info "Reloading systemd daemon..."
sudo systemctl daemon-reload

info "Enabling services (start on boot)..."
for svc in vapt-ai-app vapt-ai-celery-worker vapt-ai-celery-beat; do
    if [ -f "/etc/systemd/system/$svc.service" ]; then
        sudo systemctl enable "$svc"
    fi
done

# Enable msfrpcd only if installed
if [ -f "/etc/systemd/system/msfrpcd.service" ]; then
    sudo systemctl enable msfrpcd
fi

info "Starting services..."
for svc in vapt-ai-app vapt-ai-celery-worker vapt-ai-celery-beat; do
    if [ -f "/etc/systemd/system/$svc.service" ]; then
        info "Starting $svc..."
        sudo systemctl start "$svc" || warn "Failed to start $svc (check: journalctl -u $svc -f)"
    fi
done

if [ -f "/etc/systemd/system/msfrpcd.service" ]; then
    info "Starting msfrpcd..."
    sudo systemctl start msfrpcd || warn "Failed to start msfrpcd (check: journalctl -u msfrpcd -f)"
fi

# ---------- Wait + verify ----------

info "Waiting 5 seconds for services to start..."
sleep 5

echo ""
info "===== Service Status ====="
echo ""

for svc in vapt-ai-app vapt-ai-celery-worker vapt-ai-celery-beat msfrpcd; do
    if [ -f "/etc/systemd/system/$svc.service" ]; then
        STATUS=$(systemctl is-active "$svc" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "active" ]; then
            echo -e "  ${GREEN}✓${NC} $svc: ${GREEN}active${NC}"
        else
            echo -e "  ${RED}✗${NC} $svc: ${RED}$STATUS${NC}"
            echo -e "      Check: journalctl -u $svc -n 50"
        fi
    fi
done

echo ""
info "${GREEN}systemd setup complete!${NC}"
echo ""
info "Useful commands:"
info "  sudo systemctl status vapt-ai-app      — check app status"
info "  sudo systemctl restart vapt-ai-app     — restart app"
info "  journalctl -u vapt-ai-app -f           — follow app logs"
info "  curl http://localhost:8000/health      — health check"
info "  curl http://localhost:8000/mcp/tools/list — MCP tools"
