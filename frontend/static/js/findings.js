/**
 * VAPT-AI Findings Page — W7-D-v2.
 *
 * Placeholder — will show all findings from all scans once the
 * findings endpoint is built (W12 Evidence Auditor).
 * For now, shows a "coming soon" message with the current schema.
 */

import { el, escapeHtml } from './utils.js';

export async function renderFindings(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Findings</h1>
        <p class="page-subtitle">All vulnerability findings across scans</p>
      </div>
      <div class="card">
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <h2 class="empty-state-title">Findings Database — Coming in W12</h2>
          <p class="empty-state-description">
            The findings database will be populated by the Evidence Auditor (W12)
            and display all verified vulnerabilities with:
          </p>
          <ul style="text-align: left; display: inline-block; margin-top: 16px; font-size: 13px; color: var(--text-secondary)">
            <li>• Severity (Critical / High / Medium / Low / Info)</li>
            <li>• CVSS v3.1 score + vector</li>
            <li>• WSTG ID + MITRE ATT&CK mapping</li>
            <li>• Evidence chain (5-layer)</li>
            <li>• PoC results</li>
            <li>• Export to PDF / SARIF</li>
          </ul>
          <p style="margin-top: 24px; font-size: 12px; color: var(--text-tertiary)">
            For now, view individual scan findings in the <a href="#/reports" style="color: var(--accent)">Reports</a> page.
          </p>
        </div>
      </div>
    </div>
  `;
}
