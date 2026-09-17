/**
 * VAPT-AI Reports Page — W7-D-v2.
 *
 * Lists all conversations (scans) with summary info.
 * Click to view conversation detail.
 */

import { apiGet, el, escapeHtml, formatRelativeTime, navigate, toast } from './utils.js';

export async function renderReports(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Reports</h1>
        <p class="page-subtitle">All scan conversations + results</p>
      </div>
      <div class="card">
        <div id="reports-list">
          <div class="empty-state">
            <div class="empty-state-icon">📄</div>
            <p>Loading reports...</p>
          </div>
        </div>
      </div>
    </div>
  `;

  try {
    const res = await apiGet('/api/conversations?limit=100');
    renderReportsList(res.conversations || []);
  } catch (err) {
    document.getElementById('reports-list').innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>Failed to load reports: ${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderReportsList(conversations) {
  const list = document.getElementById('reports-list');

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
    <table class="table">
      <thead>
        <tr>
          <th>Title</th>
          <th>Messages</th>
          <th>Last Activity</th>
          <th>Created</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${conversations.map(c => `
          <tr style="cursor: pointer" data-id="${c.id}">
            <td><strong>${escapeHtml(c.title)}</strong></td>
            <td>${c.message_count}</td>
            <td>${formatRelativeTime(c.last_message_at || c.created_at)}</td>
            <td>${formatRelativeTime(c.created_at)}</td>
            <td><button class="btn btn-ghost btn-sm" data-id="${c.id}">View →</button></td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;

  // Attach click handlers
  list.querySelectorAll('tr[data-id]').forEach(tr => {
    tr.addEventListener('click', () => {
      navigate('/chat');
      // The chat page will load the conversation
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('select-conversation', { detail: tr.dataset.id }));
      }, 100);
    });
  });
}
