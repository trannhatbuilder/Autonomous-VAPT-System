/**
 * VAPT-AI Reports Page — W19-S5 (updated).
 *
 * CyberStrikeAI-pattern task management with reports.
 * Reference: CyberStrikeAI task-management.png + EVVO report history.
 *
 * W19 update: Added PDF + SARIF download buttons for each scan.
 *
 * Lists all conversations (scans) with:
 *   - Summary info (title, messages, last activity)
 *   - PDF download button → GET /api/orchestration/scans/{id}/report.pdf
 *   - SARIF download button → GET /api/orchestration/scans/{id}/report.sarif
 *   - View button → navigate to chat
 */

import { apiGet, getAccessToken, el, escapeHtml, formatRelativeTime, navigate, toast } from './utils.js';

export async function renderReports(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Reports</h1>
        <p class="page-subtitle">All scan conversations + downloadable reports</p>
      </div>
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Scan Reports</h2>
          <button class="btn btn-secondary btn-small" id="refresh-reports-btn">↻ Refresh</button>
        </div>
        <div id="reports-list">
          <div class="empty-state">
            <div class="empty-state-icon">📄</div>
            <p>Loading reports...</p>
          </div>
        </div>
      </div>
    </div>
  `;

  const refreshBtn = container.querySelector('#refresh-reports-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadReports());
  }

  await loadReports();
}


async function loadReports() {
  const list = document.getElementById('reports-list');
  if (!list) return;

  try {
    const res = await apiGet('/api/conversations?limit=100');
    renderReportsList(res.conversations || []);
  } catch (err) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>Failed to load reports: ${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}


function renderReportsList(conversations) {
  const list = document.getElementById('reports-list');
  if (!list) return;

  if (conversations.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <h2 class="empty-state-title">No reports yet</h2>
        <p class="empty-state-description">Run a scan from the chat page to generate reports.</p>
      </div>
    `;
    return;
  }

  list.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Title</th>
          <th>Messages</th>
          <th>Last Activity</th>
          <th>Created</th>
          <th>Downloads</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${conversations.map(c => `
          <tr class="report-row" data-id="${escapeHtml(c.id)}">
            <td class="col-title"><strong>${escapeHtml(c.title || 'Untitled')}</strong></td>
            <td class="col-messages">${c.message_count || 0}</td>
            <td class="col-time">${formatRelativeTime(c.last_message_at || c.created_at)}</td>
            <td class="col-time">${formatRelativeTime(c.created_at)}</td>
            <td class="col-downloads">
              <button class="btn btn-secondary btn-small download-pdf-btn"
                      data-scan-id="${escapeHtml(c.id)}"
                      title="Download PDF report">
                📄 PDF
              </button>
              <button class="btn btn-secondary btn-small download-sarif-btn"
                      data-scan-id="${escapeHtml(c.id)}"
                      title="Download SARIF report">
                📋 SARIF
              </button>
            </td>
            <td>
              <button class="btn btn-ghost btn-small view-report-btn"
                      data-id="${escapeHtml(c.id)}">View →</button>
            </td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;

  // Wire up PDF download
  list.querySelectorAll('.download-pdf-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const scanId = btn.dataset.scanId;
      await downloadReport(scanId, 'pdf');
    });
  });

  // Wire up SARIF download
  list.querySelectorAll('.download-sarif-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const scanId = btn.dataset.scanId;
      await downloadReport(scanId, 'sarif');
    });
  });

  // Wire up view button → chat
  list.querySelectorAll('.view-report-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      sessionStorage.setItem('vapt_active_conversation', id);
      navigate('/chat');
    });
  });

  // Wire up row click → chat
  list.querySelectorAll('.report-row').forEach(tr => {
    tr.addEventListener('click', () => {
      sessionStorage.setItem('vapt_active_conversation', tr.dataset.id);
      navigate('/chat');
    });
  });
}


async function downloadReport(scanId, format) {
  const ext = format === 'pdf' ? 'pdf' : 'sarif';
  const url = `/api/orchestration/scans/${encodeURIComponent(scanId)}/report.${ext}`;
  const token = getAccessToken();

  toast(`Generating ${format.toUpperCase()} report...`, 'info');

  try {
    const response = await fetch(url, {
      headers: { 'Authorization': `Bearer ${token}` },
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    // Get the blob
    const blob = await response.blob();

    // Trigger download
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `scan_${scanId}.${ext === 'sarif' ? 'sarif.json' : 'pdf'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(downloadUrl);

    toast(`${format.toUpperCase()} report downloaded`, 'success');
  } catch (err) {
    toast(`Failed to download ${format.toUpperCase()}: ${err.message}`, 'error');
  }
}