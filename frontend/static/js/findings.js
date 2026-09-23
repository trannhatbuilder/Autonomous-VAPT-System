/**
 * VAPT-AI Findings Page — W19-S4 (rewrite).
 *
 * CyberStrikeAI-pattern vulnerability management: findings table + detail.
 * Reference: CyberStrikeAI vulnerability-management.png + EVVO FindingsDashboard.
 *
 * Features:
 *   - Findings table with filters (severity, scan_id, verified)
 *   - Click row → detail modal (evidence chain, PoC, remediation)
 *   - Pagination
 *
 * API: GET /api/findings?scan_id=&severity=&verified=&limit=&offset=
 */

import {
  apiGet, el, escapeHtml, toast, showModal,
} from './utils.js';

// ---------- State ----------
let currentFilter = {
  scan_id: '',
  severity: '',
  verified: '',
  limit: 50,
  offset: 0,
};
let totalCount = 0;

// ---------- Main render ----------
export async function renderFindings(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Findings</h1>
        <p class="page-subtitle">All vulnerability findings across scans</p>
      </div>

      <!-- Filters -->
      <div class="card">
        <div class="filters-row">
          <div class="filter-group">
            <label class="filter-label">Scan ID</label>
            <input type="text" id="filter-scan-id" class="filter-input"
                   placeholder="All scans" value="${escapeHtml(currentFilter.scan_id)}" />
          </div>
          <div class="filter-group">
            <label class="filter-label">Severity</label>
            <select id="filter-severity" class="filter-select">
              <option value="">All</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="info">Info</option>
            </select>
          </div>
          <div class="filter-group">
            <label class="filter-label">Verified</label>
            <select id="filter-verified" class="filter-select">
              <option value="">All</option>
              <option value="true">Verified only</option>
              <option value="false">Unverified only</option>
            </select>
          </div>
          <button class="btn btn-primary btn-small" id="apply-filters-btn">Apply Filters</button>
          <button class="btn btn-secondary btn-small" id="clear-filters-btn">Clear</button>
        </div>
      </div>

      <!-- Findings table -->
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Findings <span id="findings-count" class="badge badge-info">0</span></h2>
          <button class="btn btn-secondary btn-small" id="refresh-findings-btn">↻ Refresh</button>
        </div>
        <div id="findings-table-container">
          <div class="empty-state">
            <div class="empty-state-icon">⏳</div>
            <p>Loading findings...</p>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div class="pagination-row" id="pagination-row" style="display: none;">
        <button class="btn btn-secondary btn-small" id="prev-page-btn">← Previous</button>
        <span id="pagination-info">—</span>
        <button class="btn btn-secondary btn-small" id="next-page-btn">Next →</button>
      </div>
    </div>
  `;

  // Wire up filters
  const applyBtn = container.querySelector('#apply-filters-btn');
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      currentFilter.scan_id = container.querySelector('#filter-scan-id').value.trim();
      currentFilter.severity = container.querySelector('#filter-severity').value;
      currentFilter.verified = container.querySelector('#filter-verified').value;
      currentFilter.offset = 0;
      loadFindings(container);
    });
  }

  const clearBtn = container.querySelector('#clear-filters-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      currentFilter = { scan_id: '', severity: '', verified: '', limit: 50, offset: 0 };
      container.querySelector('#filter-scan-id').value = '';
      container.querySelector('#filter-severity').value = '';
      container.querySelector('#filter-verified').value = '';
      loadFindings(container);
    });
  }

  const refreshBtn = container.querySelector('#refresh-findings-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadFindings(container));
  }

  const prevBtn = container.querySelector('#prev-page-btn');
  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (currentFilter.offset >= currentFilter.limit) {
        currentFilter.offset -= currentFilter.limit;
        loadFindings(container);
      }
    });
  }

  const nextBtn = container.querySelector('#next-page-btn');
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      if (currentFilter.offset + currentFilter.limit < totalCount) {
        currentFilter.offset += currentFilter.limit;
        loadFindings(container);
      }
    });
  }

  // Initial load
  await loadFindings(container);
}


// ---------- Data loading ----------
async function loadFindings(container) {
  const tableContainer = container.querySelector('#findings-table-container');
  if (!tableContainer) return;

  // Build query string
  const params = new URLSearchParams();
  params.set('limit', String(currentFilter.limit));
  params.set('offset', String(currentFilter.offset));
  if (currentFilter.scan_id) params.set('scan_id', currentFilter.scan_id);
  if (currentFilter.severity) params.set('severity', currentFilter.severity);
  if (currentFilter.verified) params.set('verified', currentFilter.verified);

  try {
    const res = await apiGet(`/api/findings?${params.toString()}`);
    const findings = res.findings || [];
    totalCount = res.total || findings.length;

    // Update count badge
    const countEl = container.querySelector('#findings-count');
    if (countEl) countEl.textContent = String(totalCount);

    // Update pagination
    updatePagination(container);

    if (findings.length === 0) {
      tableContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <h2 class="empty-state-title">No findings</h2>
          <p class="empty-state-description">
            No vulnerability findings match your filters.
            Run a scan to generate findings.
          </p>
        </div>
      `;
      return;
    }

    // Build table
    const rows = findings.map(f => {
      const severity = (f.severity || 'info').toLowerCase();
      const severityClass = `severity-${severity}`;
      const cvss = f.cvss_score ? f.cvss_score.toFixed(1) : '—';
      const wstg = f.wstg_test_id || '—';
      const mitre = f.mitre_attack_technique || '—';
      const verifiedBadge = f.verified
        ? '<span class="badge badge-success">✓ Verified</span>'
        : '<span class="badge badge-warning">Unverified</span>';
      const fpBadge = f.false_positive
        ? '<span class="badge badge-danger">FP</span>'
        : '';

      return `
        <tr class="finding-row" data-finding-id="${escapeHtml(f.id)}">
          <td class="col-severity">
            <span class="severity-badge ${severityClass}">${escapeHtml(severity.toUpperCase())}</span>
          </td>
          <td class="col-name">${escapeHtml(f.name || 'Untitled')}</td>
          <td class="col-cvss">${cvss}</td>
          <td class="col-wstg">${escapeHtml(wstg)}</td>
          <td class="col-mitre">${escapeHtml(mitre)}</td>
          <td class="col-verified">${verifiedBadge} ${fpBadge}</td>
        </tr>
      `;
    }).join('');

    tableContainer.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Severity</th>
            <th>Name</th>
            <th>CVSS</th>
            <th>WSTG ID</th>
            <th>MITRE ATT&CK</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    // Wire up row clicks → detail modal
    tableContainer.querySelectorAll('.finding-row').forEach(row => {
      row.addEventListener('click', () => {
        const findingId = row.dataset.findingId;
        showFindingDetail(findings.find(f => f.id === findingId));
      });
    });
  } catch (err) {
    tableContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>Failed to load findings: ${escapeHtml(err.message)}</p>
        <p class="empty-state-description" style="margin-top: 8px;">
          Note: The /api/findings endpoint requires W19-S9 backend update.
        </p>
      </div>
    `;
  }
}


function updatePagination(container) {
  const paginationRow = container.querySelector('#pagination-row');
  const paginationInfo = container.querySelector('#pagination-info');
  const prevBtn = container.querySelector('#prev-page-btn');
  const nextBtn = container.querySelector('#next-page-btn');

  if (!paginationRow) return;

  if (totalCount > currentFilter.limit) {
    paginationRow.style.display = 'flex';
    const from = currentFilter.offset + 1;
    const to = Math.min(currentFilter.offset + currentFilter.limit, totalCount);
    if (paginationInfo) {
      paginationInfo.textContent = `${from}-${to} of ${totalCount}`;
    }
    if (prevBtn) prevBtn.disabled = currentFilter.offset === 0;
    if (nextBtn) nextBtn.disabled = to >= totalCount;
  } else {
    paginationRow.style.display = 'none';
  }
}


function showFindingDetail(finding) {
  if (!finding) {
    toast('Finding not found', 'error');
    return;
  }

  const severity = (finding.severity || 'info').toLowerCase();
  const cvss = finding.cvss_score ? finding.cvss_score.toFixed(1) : 'N/A';
  const cvssVector = finding.cvss_vector || 'N/A';

  const detailHtml = `
    <div class="finding-detail">
      <div class="detail-section">
        <h3>${escapeHtml(finding.name || 'Untitled Finding')}</h3>
        <div class="detail-meta">
          <span class="severity-badge severity-${severity}">${escapeHtml(severity.toUpperCase())}</span>
          <span class="meta-item"><strong>CVSS:</strong> ${cvss} <code>${escapeHtml(cvssVector)}</code></span>
        </div>
      </div>

      <div class="detail-section">
        <h4>Location</h4>
        <p>${escapeHtml(finding.location || 'N/A')}</p>
      </div>

      <div class="detail-section">
        <h4>Description</h4>
        <p>${escapeHtml(finding.description || 'No description available.')}</p>
      </div>

      <div class="detail-section">
        <h4>Standards Mapping</h4>
        <ul class="detail-list">
          <li><strong>WSTG ID:</strong> ${escapeHtml(finding.wstg_test_id || 'N/A')}</li>
          <li><strong>MITRE ATT&CK:</strong> ${escapeHtml(finding.mitre_attack_technique || 'N/A')}</li>
          <li><strong>CWE:</strong> ${escapeHtml(finding.cwe_id || 'N/A')}</li>
          <li><strong>CVE:</strong> ${escapeHtml(finding.cve_id || 'N/A')}</li>
        </ul>
      </div>

      <div class="detail-section">
        <h4>PoC Status</h4>
        <p>
          <span class="badge ${finding.poc_status === 'successful' ? 'badge-success' : 'badge-warning'}">
            ${escapeHtml(finding.poc_status || 'not_attempted')}
          </span>
        </p>
      </div>

      <div class="detail-section">
        <h4>Remediation</h4>
        <p>${escapeHtml(finding.remediation || 'No remediation guidance available.')}</p>
      </div>

      <div class="detail-section">
        <h4>Scan Info</h4>
        <ul class="detail-list">
          <li><strong>Scan ID:</strong> ${escapeHtml(finding.scan_id || 'N/A')}</li>
          <li><strong>Verified:</strong> ${finding.verified ? 'Yes' : 'No'}</li>
          <li><strong>False Positive:</strong> ${finding.false_positive ? 'Yes' : 'No'}</li>
        </ul>
      </div>
    </div>
  `;

  showModal({
    title: 'Finding Detail',
    body: detailHtml,
    actions: [{ label: 'Close', onClick: (close) => close() }],
  });
}