/**
 * VAPT-AI Dashboard Page — W19-S2.
 *
 * CyberStrikeAI-pattern dashboard: system overview + scan history + quick-start.
 * Reference: CyberStrikeAI dashboard.png + EVVO DashboardPage.tsx.
 *
 * Layout:
 *   - Stats cards row (total scans, total findings, critical count, RL ε)
 *   - Quick-start button → navigate to chat
 *   - Recent scans table (last 10 conversations)
 */

import {
  apiGet, el, escapeHtml, formatRelativeTime, navigate, toast,
} from './utils.js';

// ---------- Main render ----------
export async function renderDashboard(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Dashboard</h1>
        <p class="page-subtitle">System overview + recent scan activity</p>
      </div>

      <!-- Stats cards row -->
      <div class="stats-cards" id="stats-cards">
        <div class="stat-card">
          <div class="stat-icon">🎯</div>
          <div class="stat-content">
            <div class="stat-value" id="stat-scans">—</div>
            <div class="stat-label">Total Scans</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🔍</div>
          <div class="stat-content">
            <div class="stat-value" id="stat-findings">—</div>
            <div class="stat-label">Total Findings</div>
          </div>
        </div>
        <div class="stat-card stat-card-critical">
          <div class="stat-icon">⚠️</div>
          <div class="stat-content">
            <div class="stat-value" id="stat-critical">—</div>
            <div class="stat-label">Critical Findings</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🧠</div>
          <div class="stat-content">
            <div class="stat-value" id="stat-epsilon">—</div>
            <div class="stat-label">RL Epsilon</div>
          </div>
        </div>
      </div>

      <!-- Quick-start -->
      <div class="card quick-start-card">
        <div class="quick-start-content">
          <h2 class="quick-start-title">Start a New Scan</h2>
          <p class="quick-start-description">
            Open the chat, type a target URL or IP, and the AI agent will
            automatically plan and execute a pentest.
          </p>
          <button class="btn btn-primary btn-large" id="quick-start-btn">
            <span class="btn-icon">▶</span> Start Scan
          </button>
        </div>
      </div>

      <!-- Recent scans table -->
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Recent Scans</h2>
          <button class="btn btn-secondary btn-small" id="refresh-btn">↻ Refresh</button>
        </div>
        <div id="scans-table-container">
          <div class="empty-state">
            <div class="empty-state-icon">⏳</div>
            <p>Loading scans...</p>
          </div>
        </div>
      </div>
    </div>
  `;

  // Wire up buttons
  const quickStartBtn = container.querySelector('#quick-start-btn');
  if (quickStartBtn) {
    quickStartBtn.addEventListener('click', () => navigate('/chat'));
  }

  const refreshBtn = container.querySelector('#refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadDashboardData(container));
  }

  // Load data
  await loadDashboardData(container);
}


// ---------- Data loading ----------
async function loadDashboardData(container) {
  await Promise.all([
    loadStats(container),
    loadRecentScans(container),
  ]);
}


async function loadStats(container) {
  try {
    // Load conversations for scan count + findings count
    const convRes = await apiGet('/api/conversations?limit=100');
    const conversations = convRes.conversations || [];
    const totalScans = conversations.length;
    const totalMessages = conversations.reduce(
      (sum, c) => sum + (c.message_count || 0), 0
    );

    // Try to load RL stats (may fail if RL not bootstrapped)
    let epsilon = 'N/A';
    try {
      const rlRes = await apiGet('/api/rl/stats');
      if (rlRes.bootstrapped && rlRes.policy) {
        epsilon = rlRes.policy.epsilon.toFixed(4);
      }
    } catch (e) {
      // RL not bootstrapped — leave as N/A
    }

    // Update stat cards
    const scansEl = container.querySelector('#stat-scans');
    const findingsEl = container.querySelector('#stat-findings');
    const criticalEl = container.querySelector('#stat-critical');
    const epsilonEl = container.querySelector('#stat-epsilon');

    if (scansEl) scansEl.textContent = String(totalScans);
    if (findingsEl) findingsEl.textContent = String(totalMessages); // proxy until /api/findings
    if (criticalEl) criticalEl.textContent = '—'; // requires findings endpoint
    if (epsilonEl) epsilonEl.textContent = epsilon;
  } catch (err) {
    toast('Failed to load stats: ' + err.message, 'error');
  }
}


async function loadRecentScans(container) {
  const tableContainer = container.querySelector('#scans-table-container');
  if (!tableContainer) return;

  try {
    const res = await apiGet('/api/conversations?limit=10');
    const conversations = res.conversations || [];

    if (conversations.length === 0) {
      tableContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📭</div>
          <h2 class="empty-state-title">No scans yet</h2>
          <p class="empty-state-description">
            Click "Start Scan" above to begin your first pentest.
          </p>
        </div>
      `;
      return;
    }

    // Build table
    const rows = conversations.map(c => `
      <tr class="scan-row" data-conversation-id="${escapeHtml(c.id)}">
        <td class="col-title">${escapeHtml(c.title || 'Untitled scan')}</td>
        <td class="col-messages">${c.message_count || 0}</td>
        <td class="col-time">${formatRelativeTime(c.last_message_at || c.created_at)}</td>
      </tr>
    `).join('');

    tableContainer.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Scan Title</th>
            <th>Messages</th>
            <th>Last Activity</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    // Wire up row clicks → navigate to chat with that conversation
    tableContainer.querySelectorAll('.scan-row').forEach(row => {
      row.addEventListener('click', () => {
        const convId = row.dataset.conversationId;
        // Store the conversation ID for chat.js to pick up
        sessionStorage.setItem('vapt_active_conversation', convId);
        navigate('/chat');
      });
    });
  } catch (err) {
    tableContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>Failed to load scans: ${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}