/**
 * VAPT-AI Frontend App — W7-D-v2.
 *
 * Main entry point: router + app shell (sidebar + main content).
 * Vanilla JS — no framework, no build step.
 *
 * Routes (hash-based):
 *   #/login     — Auth page
 *   #/chat      — Chat page (default)
 *   #/reports   — Reports list
 *   #/findings  — Findings database (placeholder)
 *   #/settings  — Settings
 *   #/about     — About
 */

import { registerRoute, startRouter, navigate, requireAuth, logout, checkAuth } from './utils.js';
import { renderLogin } from './auth.js';
import { renderChat } from './chat.js';
import { renderReports } from './reports.js';
import { renderFindings } from './findings.js';
import { renderSettings } from './settings.js';

// ---------- App shell (sidebar + main content) ----------
async function renderShell(container, pageRenderer) {
  const user = await requireAuth();
  if (!user) return;

  container.innerHTML = `
    <div class="app-layout">
      <aside class="app-sidebar">
        <div class="app-sidebar-header">
          <div class="app-sidebar-logo">
            <span class="app-sidebar-logo-icon">🛡️</span>
            <span>VAPT-AI</span>
          </div>
        </div>
        <nav class="app-sidebar-nav">
          <div class="nav-item" data-route="/chat">
            <span class="nav-item-icon">💬</span>
            <span>Chat</span>
          </div>
          <div class="nav-item" data-route="/reports">
            <span class="nav-item-icon">📄</span>
            <span>Reports</span>
          </div>
          <div class="nav-item" data-route="/findings">
            <span class="nav-item-icon">🔍</span>
            <span>Findings</span>
          </div>
          <div class="nav-item" data-route="/settings">
            <span class="nav-item-icon">⚙️</span>
            <span>Settings</span>
          </div>
        </nav>
        <div class="app-sidebar-footer">
          <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px">
            <span style="font-size: 11px; color: var(--text-tertiary)">${user.email}</span>
            <button class="btn btn-ghost btn-sm" id="sidebar-logout" title="Logout">⏏</button>
          </div>
        </div>
      </aside>
      <main class="app-main" id="app-main"></main>
    </div>
  `;

  // Attach nav handlers
  container.querySelectorAll('.nav-item[data-route]').forEach(item => {
    item.addEventListener('click', () => navigate(item.dataset.route));
  });

  // Highlight active route
  const currentHash = window.location.hash.slice(1) || '/chat';
  container.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.route === currentHash);
  });

  // Logout button
  const logoutBtn = document.getElementById('sidebar-logout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', logout);
  }

  // Render the page into #app-main
  const main = document.getElementById('app-main');
  await pageRenderer(main);
}

// ---------- Routes ----------
registerRoute('/login', async (container) => {
  // If already logged in, redirect to chat
  const user = await checkAuth();
  if (user) {
    navigate('/chat');
    return;
  }
  await renderLogin(container);
});

registerRoute('/chat', async (container) => {
  await renderShell(container, renderChat);
});

registerRoute('/reports', async (container) => {
  await renderShell(container, renderReports);
});

registerRoute('/findings', async (container) => {
  await renderShell(container, renderFindings);
});

registerRoute('/settings', async (container) => {
  await renderShell(container, renderSettings);
});

registerRoute('/404', async (container) => {
  container.innerHTML = `
    <div class="empty-state" style="padding: 100px 20px">
      <div class="empty-state-icon">🔍</div>
      <h2 class="empty-state-title">Page not found</h2>
      <p class="empty-state-description">The page you're looking for doesn't exist.</p>
      <button class="btn btn-primary" style="margin-top: 16px" onclick="window.location.hash = '#/chat'">Back to Chat</button>
    </div>
  `;
});

// ---------- Start ----------
startRouter();