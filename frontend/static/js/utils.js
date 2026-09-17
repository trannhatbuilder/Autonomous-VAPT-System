/**
 * VAPT-AI Frontend Utils — W7-D-v2.
 *
 * Helpers: API fetch wrapper, toast notifications, router, DOM helpers.
 */

// ---------- DOM helpers ----------
export function $(selector, parent = document) {
  return parent.querySelector(selector);
}

export function $$(selector, parent = document) {
  return Array.from(parent.querySelectorAll(selector));
}

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value !== null && value !== undefined) {
      node.setAttribute(key, value);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

// ---------- Token management ----------
const ACCESS_TOKEN_KEY = 'vapt_access_token';
const REFRESH_TOKEN_KEY = 'vapt_refresh_token';

export function setTokens(accessToken, refreshToken) {
  if (accessToken) localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  if (refreshToken) localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function getAccessToken() {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function clearTokens() {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

// ---------- API fetch wrapper ----------
const API_BASE = '';

export async function api(path, options = {}) {
  const accessToken = getAccessToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  // Attach Bearer token if available (and not a login/refresh request)
  if (accessToken && !path.includes('/api/auth/login') && !path.includes('/api/auth/refresh')) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  const res = await fetch(API_BASE + path, {
    credentials: 'include',
    headers,
    ...options,
  });

  if (res.status === 401) {
    // Try refresh token once before redirecting to login
    const refreshed = await tryRefresh();
    if (refreshed) {
      // Retry original request with new token
      const newToken = getAccessToken();
      headers['Authorization'] = `Bearer ${newToken}`;
      const retryRes = await fetch(API_BASE + path, {
        credentials: 'include',
        headers,
        ...options,
      });
      if (retryRes.ok) {
        if (retryRes.status === 204) return null;
        const ct = retryRes.headers.get('content-type') || '';
        if (ct.includes('application/json')) return retryRes.json();
        return retryRes.text();
      }
    }
    // Refresh failed → clear tokens + redirect to login
    clearTokens();
    if (!window.location.hash.startsWith('#/login')) {
      window.location.hash = '#/login';
    }
    throw new Error('Unauthorized');
  }

  if (!res.ok) {
    let detail;
    try { detail = await res.json(); } catch { detail = await res.text(); }
    // Extract human-readable error message from FastAPI 422 validation errors
    let msg;
    if (typeof detail === 'object' && detail?.detail) {
      if (Array.isArray(detail.detail)) {
        // Pydantic 422: [{type, loc, msg, input}, ...]
        const errors = detail.detail.map(e => {
          const field = e.loc ? e.loc.join('.') : 'unknown';
          return `${field}: ${e.msg}`;
        });
        msg = errors.join('; ');
      } else {
        msg = String(detail.detail);
      }
    } else if (typeof detail === 'string') {
      msg = detail;
    } else {
      msg = `HTTP ${res.status}`;
    }
    throw new Error(msg);
  }

  if (res.status === 204) return null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

// Try to refresh the access token using the refresh token
async function tryRefresh() {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    if (data.access_token) {
      setTokens(data.access_token, data.refresh_token || refreshToken);
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

export const apiGet = (path) => api(path, { method: 'GET' });
export const apiPost = (path, body) => api(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined });
export const apiPatch = (path, body) => api(path, { method: 'PATCH', body: JSON.stringify(body) });
export const apiDelete = (path) => api(path, { method: 'DELETE' });

// ---------- Toast notifications ----------
export function toast(message, type = 'info', duration = 4000) {
  const container = $('#toast-container');
  if (!container) return;
  const t = el('div', { class: `toast toast-${type}` }, message);
  container.append(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transition = 'opacity 0.2s';
    setTimeout(() => t.remove(), 200);
  }, duration);
}

toast.success = (msg) => toast(msg, 'success');
toast.error = (msg) => toast(msg, 'error');
toast.warning = (msg) => toast(msg, 'warning');
toast.info = (msg) => toast(msg, 'info');

// ---------- Modal ----------
export function showModal({ title, body, actions = [] }) {
  const container = $('#modal-container');
  const overlay = el('div', { class: 'modal-overlay' });
  const card = el('div', { class: 'modal-card' });
  card.append(el('h3', { class: 'modal-title' }, title));
  card.append(el('div', { class: 'modal-body' }, body));
  const actionsDiv = el('div', { class: 'modal-actions' });
  for (const action of actions) {
    const btn = el('button', {
      class: `btn ${action.variant === 'danger' ? 'btn-danger' : action.variant === 'primary' ? 'btn-primary' : 'btn-secondary'}`,
      onclick: () => {
        if (action.onClick) action.onClick();
        if (action.close !== false) overlay.remove();
      },
    }, action.label);
    actionsDiv.append(btn);
  }
  card.append(actionsDiv);
  overlay.append(card);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.remove();
  });
  container.append(overlay);
  return overlay;
}

// ---------- Hash router ----------
const routes = {};
let currentRoute = null;

export function registerRoute(path, handler) {
  routes[path] = handler;
}

export function navigate(path) {
  window.location.hash = path;
}

export function startRouter() {
  window.addEventListener('hashchange', renderRoute);
  renderRoute();
}

async function renderRoute() {
  const hash = window.location.hash.slice(1) || '/chat';
  const app = $('#app');
  app.innerHTML = '';

  // Find matching route
  let handler = routes[hash];
  if (!handler) {
    // Try pattern match (e.g. /reports/:id)
    for (const [pattern, h] of Object.entries(routes)) {
      if (pattern.includes(':')) {
        const regex = new RegExp('^' + pattern.replace(/:[^/]+/g, '([^/]+)') + '$');
        const match = hash.match(regex);
        if (match) {
          handler = () => h(...match.slice(1));
          break;
        }
      }
    }
  }

  if (!handler) {
    handler = routes['/404'] || (() => app.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><h2 class="empty-state-title">Page not found</h2></div>');
  }

  currentRoute = hash;
  try {
    await handler(app);
  } catch (err) {
    console.error('Route error:', err);
    app.innerHTML = `<div class="empty-state"><div class="empty-state-icon">⚠️</div><h2 class="empty-state-title">Error</h2><p class="empty-state-description">${err.message}</p></div>`;
  }
}

// ---------- Format helpers ----------
export function formatRelativeTime(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now - date;
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export function formatTime(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ---------- Auth helpers ----------
export async function checkAuth() {
  try {
    const user = await apiGet('/api/auth/me');
    return user;
  } catch {
    return null;
  }
}

export async function requireAuth() {
  const user = await checkAuth();
  if (!user) {
    navigate('/login');
    return null;
  }
  return user;
}

export async function logout() {
  // Clear tokens from localStorage
  clearTokens();
  // Best-effort: notify backend to revoke refresh token
  try {
    const refreshToken = getRefreshToken();
    if (refreshToken) {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    }
  } catch (err) {
    console.error('Logout error:', err);
  }
  // Clear again (in case logout API re-set something)
  clearTokens();
  navigate('/login');
}