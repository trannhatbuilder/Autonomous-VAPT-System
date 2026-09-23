/**
 * VAPT-AI C2 Dashboard Page — W19-S3 (stub).
 *
 * CyberStrikeAI-pattern WebShell management: unified session table.
 * Reference: CyberStrikeAI webshell-management.png (D28 unified C2).
 *
 * W19 scope: session table + close button only.
 * Interactive terminal (xterm.js) deferred to W20+ — C2 is auxiliary.
 *
 * Layout:
 *   - Scan ID input (filter sessions by scan)
 *   - Unified session table (Python beacon + MSF meterpreter + sqlmap webshell)
 *   - Action: close session
 */

import {
  apiGet, apiPost, el, escapeHtml, formatTime, toast, showModal,
} from './utils.js';

// ---------- State ----------
let currentScanId = '';

// ---------- Main render ----------
export async function renderC2(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">C2 Sessions</h1>
        <p class="page-subtitle">Unified command-and-control session dashboard</p>
      </div>

      <!-- Scan ID filter -->
      <div class="card">
        <div class="filter-row">
          <label class="filter-label" for="scan-id-input">Scan ID:</label>
          <input
            type="text"
            id="scan-id-input"
            class="filter-input"
            placeholder="Enter scan ID to filter sessions"
            value="${escapeHtml(currentScanId)}"
          />
          <button class="btn btn-primary btn-small" id="load-sessions-btn">Load Sessions</button>
        </div>
      </div>

      <!-- Stats summary -->
      <div class="stats-cards" id="c2-stats-cards">
        <div class="stat-card">
          <div class="stat-icon">🌐</div>
          <div class="stat-content">
            <div class="stat-value" id="c2-stat-total">—</div>
            <div class="stat-label">Total Sessions</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🟢</div>
          <div class="stat-content">
            <div class="stat-value" id="c2-stat-active">—</div>
            <div class="stat-label">Active</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🔴</div>
          <div class="stat-content">
            <div class="stat-value" id="c2-stat-closed">—</div>
            <div class="stat-label">Closed</div>
          </div>
        </div>
      </div>

      <!-- Sessions table -->
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Sessions</h2>
          <button class="btn btn-secondary btn-small" id="refresh-c2-btn">↻ Refresh</button>
        </div>
        <div id="sessions-table-container">
          <div class="empty-state">
            <div class="empty-state-icon">🔐</div>
            <p>Enter a scan ID above and click "Load Sessions".</p>
          </div>
        </div>
      </div>
    </div>
  `;

  // Wire up
  const loadBtn = container.querySelector('#load-sessions-btn');
  const scanInput = container.querySelector('#scan-id-input');
  if (loadBtn) {
    loadBtn.addEventListener('click', () => {
      currentScanId = scanInput.value.trim();
      if (currentScanId) {
        loadSessions(container, currentScanId);
      } else {
        toast('Please enter a scan ID', 'warning');
      }
    });
  }

  const refreshBtn = container.querySelector('#refresh-c2-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      if (currentScanId) {
        loadSessions(container, currentScanId);
      } else {
        toast('Enter a scan ID first', 'warning');
      }
    });
  }

  // Enter key on input
  if (scanInput) {
    scanInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        loadBtn?.click();
      }
    });
  }
}


// ---------- Data loading ----------
async function loadSessions(container, scanId) {
  const tableContainer = container.querySelector('#sessions-table-container');
  if (!tableContainer) return;

  tableContainer.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">⏳</div>
      <p>Loading sessions for scan ${escapeHtml(scanId)}...</p>
    </div>
  `;

  try {
    const res = await apiGet(`/api/c2/sessions/${encodeURIComponent(scanId)}`);
    const sessions = res.sessions || [];

    // Update stats
    updateStats(container, sessions);

    if (sessions.length === 0) {
      tableContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📭</div>
          <h2 class="empty-state-title">No C2 sessions</h2>
          <p class="empty-state-description">
            No C2 sessions found for scan "${escapeHtml(scanId)}".
            Sessions appear here after a beacon checks in or an exploit succeeds.
          </p>
        </div>
      `;
      return;
    }

    // Build table
    const rows = sessions.map(s => {
      const beaconType = s.beacon_type || 'unknown';
      const status = s.status || 'unknown';
      const statusClass = status === 'active' ? 'badge-success' :
                          status === 'closed' ? 'badge-danger' : 'badge-warning';
      const beaconIcon = beaconType.includes('python') ? '🐍' :
                         beaconType.includes('msf') ? '🎯' :
                         beaconType.includes('webshell') ? '🌐' : '❓';
      return `
        <tr class="session-row" data-session-id="${escapeHtml(s.id)}">
          <td class="col-beacon">${beaconIcon} ${escapeHtml(beaconType)}</td>
          <td class="col-listener">${escapeHtml(s.listener_type || '—')}</td>
          <td class="col-addr">${escapeHtml(s.remote_address || '—')}</td>
          <td class="col-host">${escapeHtml(s.hostname || '—')}</td>
          <td class="col-os">${escapeHtml(s.os || '—')}</td>
          <td class="col-checkin">${s.checkin_at ? formatTime(s.checkin_at) : '—'}</td>
          <td class="col-status">
            <span class="badge ${statusClass}">${escapeHtml(status)}</span>
          </td>
          <td class="col-actions">
            <button class="btn btn-secondary btn-small view-tasks-btn" data-session-id="${escapeHtml(s.id)}">
              View Tasks
            </button>
            ${status === 'active' ? `
              <button class="btn btn-danger btn-small close-session-btn" data-session-id="${escapeHtml(s.id)}">
                Close
              </button>
            ` : ''}
          </td>
        </tr>
      `;
    }).join('');

    tableContainer.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Beacon</th>
            <th>Listener</th>
            <th>Remote Addr</th>
            <th>Hostname</th>
            <th>OS</th>
            <th>Check-in</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    // Wire up buttons
    tableContainer.querySelectorAll('.close-session-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sessionId = btn.dataset.sessionId;
        await closeSession(scanId, sessionId);
        loadSessions(container, scanId); // reload
      });
    });

    tableContainer.querySelectorAll('.view-tasks-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const sessionId = btn.dataset.sessionId;
        viewTasks(scanId, sessionId);
      });
    });
  } catch (err) {
    tableContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>Failed to load sessions: ${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}


function updateStats(container, sessions) {
  const total = sessions.length;
  const active = sessions.filter(s => s.status === 'active').length;
  const closed = sessions.filter(s => s.status === 'closed').length;

  const totalEl = container.querySelector('#c2-stat-total');
  const activeEl = container.querySelector('#c2-stat-active');
  const closedEl = container.querySelector('#c2-stat-closed');

  if (totalEl) totalEl.textContent = String(total);
  if (activeEl) activeEl.textContent = String(active);
  if (closedEl) closedEl.textContent = String(closed);
}


async function closeSession(scanId, sessionId) {
  if (!confirm(`Close session ${sessionId}?\n\nThis will kill the beacon connection.`)) {
    return;
  }
  try {
    await apiPost(`/api/c2/sessions/${encodeURIComponent(scanId)}/${encodeURIComponent(sessionId)}/close`);
    toast('Session closed', 'success');
  } catch (err) {
    toast('Failed to close session: ' + err.message, 'error');
  }
}


async function viewTasks(scanId, sessionId) {
  try {
    const res = await apiGet(`/api/c2/sessions/${encodeURIComponent(scanId)}/${encodeURIComponent(sessionId)}/tasks`);
    const tasks = res.tasks || [];

    const tasksHtml = tasks.length === 0
      ? '<p class="empty-state-description">No tasks for this session.</p>'
      : `
        <table class="data-table">
          <thead>
            <tr><th>Level</th><th>Command</th><th>Status</th><th>Sent</th></tr>
          </thead>
          <tbody>
            ${tasks.map(t => `
              <tr>
                <td>L${t.level || '?'}</td>
                <td>${escapeHtml(t.command || '—')}</td>
                <td><span class="badge">${escapeHtml(t.status || '—')}</span></td>
                <td>${t.sent_at ? formatTime(t.sent_at) : '—'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;

    showModal({
      title: `Session Tasks — ${sessionId.slice(0, 8)}`,
      body: `<div style="max-height: 400px; overflow-y: auto;">${tasksHtml}</div>`,
      actions: [{ label: 'Close', onClick: (close) => close() }],
    });
  } catch (err) {
    toast('Failed to load tasks: ' + err.message, 'error');
  }
}