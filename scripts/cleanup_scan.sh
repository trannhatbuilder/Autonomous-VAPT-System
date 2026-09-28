#!/bin/bash
# scripts/cleanup_scan.sh — post-scan artifact cleanup
# Replaces Docker container destroy (v3.2 has no Docker)
# Called by ScanRegistry.abort_scan() and after normal scan completion.
set -e
SCAN_ID="$1"

# Remove sqlmap session/output dirs
rm -rf /tmp/sqlmap-* 2>/dev/null || true

# Remove Metasploit loot + logs
rm -rf ~/.msf6/loot/* 2>/dev/null || true
rm -rf ~/.msf6/logs/* 2>/dev/null || true

# Remove nuclei output dirs
rm -rf /tmp/nuclei-* 2>/dev/null || true

# Log to /var/log/vapt-ai/cleanup/cleanup.log
mkdir -p /var/log/vapt-ai/cleanup 2>/dev/null || true
echo "$(date -Iseconds) | scan=${SCAN_ID} | cleanup complete" \
    >> /var/log/vapt-ai/cleanup/cleanup.log 2>/dev/null || true

echo "cleanup complete for scan ${SCAN_ID}"
