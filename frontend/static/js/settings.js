/**
 * VAPT-AI Settings Page — W7-D-v2 (updated).
 *
 * 3 sections:
 *   1. AI Channel Config (LLM provider) — provider, base_url, api_key, model, tokens
 *   2. HITL Audit Agent Config — separate model for audit agent (or reuse main)
 *   3. Account — logout
 *
 * Backend: GET/POST /api/settings/llm + POST /api/settings/llm/test
 */

import { apiGet, apiPost, el, escapeHtml, toast, logout } from './utils.js';

export async function renderSettings(container) {
  container.innerHTML = `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Settings</h1>
        <p class="page-subtitle">Configure LLM provider for the AI agent</p>
      </div>

      <div id="settings-content">
        <div class="empty-state">
          <div class="loading-spinner" style="margin: 0 auto"></div>
          <p style="margin-top: 12px">Loading settings...</p>
        </div>
      </div>
    </div>
  `;

  const content = document.getElementById('settings-content');

  try {
    const { llm } = await apiGet('/api/settings/llm');
    renderSettingsForm(content, llm || {});
  } catch (err) {
    content.innerHTML = `
      <div class="card">
        <div class="empty-state">
          <div class="empty-state-icon">⚠️</div>
          <h2 class="empty-state-title">Failed to load settings</h2>
          <p class="empty-state-description">${escapeHtml(err.message)}</p>
        </div>
      </div>
    `;
  }
}

function renderSettingsForm(container, llm) {
  container.innerHTML = `
    <!-- Section 1: AI Channel Config -->
    <div class="settings-section">
      <div class="card">
        <div class="settings-section-title" style="margin-bottom: 16px">
          🤖 AI Channel Configuration
        </div>
        <p style="font-size: 13px; color: var(--text-tertiary); margin-bottom: 20px">
          Configure the main LLM provider used by the AI agent for scan decisions.
        </p>

        <div class="settings-form">
          <div class="form-row">
            <div class="form-group">
              <label for="llm-provider">Provider</label>
              <select id="llm-provider" class="input">
                <option value="openai" ${llm.provider === 'openai' ? 'selected' : ''}>OpenAI / OpenAI-compatible</option>
                <option value="anthropic" ${llm.provider === 'anthropic' ? 'selected' : ''}>Anthropic (Claude)</option>
                <option value="glm" ${llm.provider === 'glm' ? 'selected' : ''}>GLM (Zhipu)</option>
                <option value="minimax" ${llm.provider === 'minimax' ? 'selected' : ''}>Minimax</option>
                <option value="deepseek" ${llm.provider === 'deepseek' ? 'selected' : ''}>DeepSeek</option>
                <option value="groq" ${llm.provider === 'groq' ? 'selected' : ''}>Groq</option>
                <option value="google" ${llm.provider === 'google' ? 'selected' : ''}>Google (Gemini)</option>
                <option value="ollama" ${llm.provider === 'ollama' ? 'selected' : ''}>Ollama (local)</option>
              </select>
            </div>
            <div class="form-group">
              <label for="llm-model">Model</label>
              <input type="text" id="llm-model" class="input" placeholder="gpt-4o-mini / glm-4-flash / claude-3-5-haiku-latest" value="${escapeHtml(llm.model || '')}" />
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="llm-base-url">Base URL</label>
              <input type="text" id="llm-base-url" class="input" placeholder="https://api.openai.com/v1" value="${escapeHtml(llm.base_url || '')}" />
            </div>
            <div class="form-group">
              <label for="llm-api-key">API Key</label>
              <input type="password" id="llm-api-key" class="input" placeholder="sk-..." value="${escapeHtml(llm.api_key || '')}" />
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="llm-max-total-tokens">Max Context Tokens</label>
              <input type="number" id="llm-max-total-tokens" class="input" placeholder="120000" min="1000" step="1000" value="${llm.max_total_tokens || 120000}" />
            </div>
            <div class="form-group">
              <label for="llm-max-completion-tokens">Max Output Tokens</label>
              <input type="number" id="llm-max-completion-tokens" class="input" placeholder="16384" min="1" step="256" value="${llm.max_completion_tokens || 16384}" />
            </div>
            <div class="form-group">
              <label for="llm-temperature">Temperature</label>
              <input type="number" id="llm-temperature" class="input" placeholder="0.7" min="0" max="2" step="0.1" value="${llm.temperature ?? 0.7}" />
            </div>
          </div>

          <div style="display: flex; gap: 8px; margin-top: 16px">
            <button class="btn btn-secondary" id="test-llm-btn">🔌 Test Connection</button>
            <button class="btn btn-primary" id="save-llm-btn">💾 Save</button>
          </div>
          <div id="llm-test-result" style="margin-top: 8px; font-size: 13px"></div>
        </div>
      </div>
    </div>

    <!-- Section 3: System Info -->
    <div class="settings-section">
      <div class="card">
        <div class="settings-section-title" style="margin-bottom: 16px">
          ⚙️ System
        </div>
        <div class="settings-row">
          <div>
            <div class="settings-row-label">HITL Mode</div>
            <div class="settings-row-description">audit_agent (default) — LLM critic reviews destructive ops</div>
          </div>
          <span class="badge badge-accent">audit_agent</span>
        </div>
        <div class="settings-row">
          <div>
            <div class="settings-row-label">Fallback Policy</div>
            <div class="settings-row-description">When LLM call fails: reject (safe default)</div>
          </div>
          <span class="badge badge-warning">reject</span>
        </div>
        <div class="settings-row">
          <div>
            <div class="settings-row-label">D18 Guardrails</div>
            <div class="settings-row-description">Max 30 decisions/scan · Max 2M tokens · Max 4h</div>
          </div>
          <span class="badge">enforced</span>
        </div>
      </div>
    </div>

    <!-- Section 4: Account -->
    <div class="settings-section">
      <div class="card">
        <div class="settings-section-title" style="margin-bottom: 16px">
          👤 Account
        </div>
        <div class="settings-row">
          <div>
            <div class="settings-row-label">Logout</div>
            <div class="settings-row-description">Sign out of VAPT-AI</div>
          </div>
          <button class="btn btn-secondary btn-sm" id="logout-btn">Logout</button>
        </div>
      </div>
    </div>

    <style>
      .settings-form { display: flex; flex-direction: column; gap: 16px; }
      .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .form-group { display: flex; flex-direction: column; gap: 6px; }
      .form-group label { font-size: 13px; font-weight: 500; color: var(--text-secondary); }
      @media (max-width: 700px) { .form-row { grid-template-columns: 1fr; } }
    </style>
  `;

  // Attach handlers
  document.getElementById('save-llm-btn').addEventListener('click', saveLLMSettings);
  document.getElementById('test-llm-btn').addEventListener('click', testLLMConnection);
  document.getElementById('logout-btn').addEventListener('click', logout);
}

async function collectLLMFormData() {
  return {
    provider: document.getElementById('llm-provider').value,
    base_url: document.getElementById('llm-base-url').value.trim(),
    api_key: document.getElementById('llm-api-key').value,
    model: document.getElementById('llm-model').value.trim(),
    max_total_tokens: parseInt(document.getElementById('llm-max-total-tokens').value) || 120000,
    max_completion_tokens: parseInt(document.getElementById('llm-max-completion-tokens').value) || 16384,
    temperature: parseFloat(document.getElementById('llm-temperature').value) || 0.7,
  };
}

async function saveLLMSettings() {
  try {
    const data = await collectLLMFormData();
    await apiPost('/api/settings/llm', data);
    toast.success('Settings saved');
    // Reload to show masked keys
    setTimeout(() => renderSettings(document.getElementById('app')), 500);
  } catch (err) {
    toast.error(`Failed to save: ${err.message}`);
  }
}

async function testLLMConnection() {
  const resultDiv = document.getElementById('llm-test-result');
  resultDiv.innerHTML = '<span style="color: var(--text-tertiary)">Testing...</span>';

  try {
    const data = await collectLLMFormData();
    const result = await apiPost('/api/settings/llm/test', data);
    if (result.success) {
      resultDiv.innerHTML = `<span style="color: var(--success)">✓ ${escapeHtml(result.message)}</span>`;
      toast.success('Connection test passed');
    } else {
      resultDiv.innerHTML = `<span style="color: var(--danger)">✗ ${escapeHtml(result.error)}</span>`;
      toast.error('Connection test failed');
    }
  } catch (err) {
    resultDiv.innerHTML = `<span style="color: var(--danger)">✗ ${escapeHtml(err.message)}</span>`;
  }
}