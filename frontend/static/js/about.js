/**
 * VAPT-AI About Page — W7-D-v2.
 *
 * System info: version, tech stack, architecture.
 */

import { apiGet, el, escapeHtml } from './utils.js';

export async function renderAbout(container) {
  let health = null;
  try {
    health = await apiGet('/health');
  } catch { /* ignore */ }

  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">About VAPT-AI</h1>
        <p class="page-subtitle">Autonomous Vulnerability Assessment & Penetration Testing AI</p>
      </div>

      <div class="settings-section">
        <div class="card">
          <h2 class="settings-section-title">System Status</h2>
          <div class="settings-row">
            <div>
              <div class="settings-row-label">Version</div>
              <div class="settings-row-description">Current build</div>
            </div>
            <span class="badge badge-accent">v${health?.version || '3.2.1'}</span>
          </div>
          <div class="settings-row">
            <div>
              <div class="settings-row-label">Environment</div>
              <div class="settings-row-description">dev / staging / prod</div>
            </div>
            <span class="badge">${health?.environment || 'unknown'}</span>
          </div>
          <div class="settings-row">
            <div>
              <div class="settings-row-label">Database</div>
              <div class="settings-row-description">PostgreSQL connection status</div>
            </div>
            <span class="badge ${health?.db_connected ? 'badge-success' : 'badge-danger'}">
              ${health?.db_connected ? '✓ connected' : '✗ disconnected'}
            </span>
          </div>
        </div>
      </div>

      <div class="settings-section">
        <div class="card">
          <h2 class="settings-section-title">Tech Stack</h2>
          <div class="settings-row">
            <div><div class="settings-row-label">Backend</div></div>
            <code class="text-sm">Python 3.12 + FastAPI + SQLAlchemy 2.0 async + PostgreSQL</code>
          </div>
          <div class="settings-row">
            <div><div class="settings-row-label">Frontend</div></div>
            <code class="text-sm">Vanilla HTML/CSS/JS (no framework)</code>
          </div>
          <div class="settings-row">
            <div><div class="settings-row-label">LLM Gateway</div></div>
            <code class="text-sm">LiteLLM (multi-provider)</code>
          </div>
          <div class="settings-row">
            <div><div class="settings-row-label">HITL Mode</div></div>
            <code class="text-sm">Option C — A-in-the-Loop (LLM critic)</code>
          </div>
          <div class="settings-row">
            <div><div class="settings-row-label">Multi-Agent</div></div>
            <code class="text-sm">LangGraph (W9+)</code>
          </div>
          <div class="settings-row">
            <div><div class="settings-row-label">RL Layer</div></div>
            <code class="text-sm">Dueling Double DQN (W16+)</code>
          </div>
        </div>
      </div>

      <div class="settings-section">
        <div class="card">
          <h2 class="settings-section-title">Architecture</h2>
          <pre style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; font-size: 11px; overflow-x: auto; line-height: 1.6">
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Vanilla JS)                                      │
│  Chat UI · Reports · Findings · Settings · About            │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP + SSE
┌────────────────────────────┴────────────────────────────────┐
│  FastAPI Backend                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Auth     │ │ Scans    │ │ HITL     │ │ Convers. │       │
│  │ (JWT)    │ │ (SSE)    │ │ (Audit)  │ │ (Chat)   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Sandbox  │ │ Exploit  │ │ Pentest  │ │ Audit    │       │
│  │ Executor │ │ (MSF)    │ │ Blackbd. │ │ Log      │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────┐
│  PostgreSQL (W1-C+D, 26 tables)                             │
│  scans · findings · evidence · hitl_approvals · convs · ... │
└─────────────────────────────────────────────────────────────┘
          </pre>
        </div>
      </div>

      <div class="settings-section">
        <div class="card">
          <h2 class="settings-section-title">Source Lineage</h2>
          <div class="settings-row">
            <div>
              <div class="settings-row-label">CyberStrikeAI (Go, Apache 2.0)</div>
              <div class="settings-row-description">Ported: 16 agents, 24 skills, 90 tool YAMLs, subprocess executor, HITL gate, C2 framework, audit log</div>
            </div>
            <span class="badge badge-info">Ported</span>
          </div>
          <div class="settings-row">
            <div>
              <div class="settings-row-label">EVVO Sentinel (Python)</div>
              <div class="settings-row-description">Adapted: RL layer, Dynamic KG, replay traces, harness bridge, scope guard (W14-W18)</div>
            </div>
            <span class="badge badge-info">Adapted</span>
          </div>
        </div>
      </div>

      <div style="margin-top: 32px; text-align: center; font-size: 12px; color: var(--text-tertiary)">
        Built with ❤️ for autonomous security testing · W7 complete
      </div>
    </div>
  `;
}
