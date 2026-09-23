/**
 * VAPT-AI Chat Page — W7-D-v2.
 *
 * CyberStrikeAI-pattern chat:
 *   - Left: conversation sidebar (history + new chat)
 *   - Center: chat messages + input
 *   - Right: HITL decision feed + panic button
 *
 * User types natural language → AI agent auto-extracts target → runs scan.
 */

import {
  apiGet, apiPost, apiDelete, el, formatRelativeTime, formatTime,
  escapeHtml, toast, showModal, getAccessToken,
} from './utils.js';
import { renderHITLPanel } from './hitl.js';

// ---------- State ----------
let conversations = [];
let activeConversation = null;
let isScanning = false;
let activeScanId = null;
let sseEventSource = null;
let sseEvents = [];

// ---------- Extract target from user message ----------
function extractTarget(content) {
  const urlMatch = content.match(/https?:\/\/[^\s,;]+/i);
  if (urlMatch) return urlMatch[0];
  const ipMatch = content.match(/\b(\d{1,3}\.){3}\d{1,3}\b/);
  if (ipMatch) return ipMatch[0];
  const domainMatch = content.match(/\b[a-z0-9-]+\.[a-z]{2,}\b/i);
  if (domainMatch) return domainMatch[0];
  return 'auto';
}

// ---------- Main render ----------
export async function renderChat(container) {
  container.innerHTML = `
    <div class="chat-layout">
      <!-- Conversation sidebar -->
      <aside class="chat-sidebar">
        <div class="chat-sidebar-header">
          <button class="btn btn-primary" style="width:100%" id="new-chat-btn">+ New chat</button>
        </div>
        <div class="chat-sidebar-list" id="conversation-list">
          <div class="empty-state" style="padding: 24px 16px">
            <div style="font-size: 24px; opacity: 0.5">💬</div>
            <p style="font-size: 12px; margin-top: 8px">No conversations yet</p>
          </div>
        </div>
      </aside>

      <!-- Chat main -->
      <main class="chat-main">
        <header class="chat-header">
          <div class="chat-header-title" id="chat-title">VAPT-AI Chat</div>
          <div class="flex items-center gap-2">
            <span id="scan-status" class="hidden" style="font-size: 12px; color: var(--warning)">● scanning...</span>
            <button class="btn btn-ghost btn-sm" id="export-btn" title="Export conversation">📄 Export</button>
          </div>
        </header>

        <div class="chat-messages" id="chat-messages">
          <div class="empty-state">
            <div class="empty-state-icon">🛡️</div>
            <h2 class="empty-state-title">VAPT-AI Chat</h2>
            <p class="empty-state-description">
              Type your scan request below to start. The AI agent will automatically
              extract the target and run the scan.
            </p>
            <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 16px">Examples:</p>
            <ul style="font-size: 12px; color: var(--text-tertiary); margin-top: 4px; text-align: left; display: inline-block">
              <li>• Scan 10.10.10.5 for vulnerabilities</li>
              <li>• Test https://example.com for SQL injection</li>
              <li>• Run a full pentest on 192.168.1.0/24</li>
            </ul>
          </div>
        </div>

        <div class="chat-input-area">
          <div class="chat-input-wrapper">
            <textarea id="chat-input" placeholder="Type your scan request... (e.g. 'Scan 10.10.10.5 for vulnerabilities')" rows="1"></textarea>
            <button class="btn btn-primary btn-icon" id="send-btn" title="Send (Enter)">➤</button>
          </div>
          <div class="chat-input-hint">
            Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for newline
          </div>
        </div>
      </main>

      <!-- HITL panel -->
      <aside class="hitl-panel" id="hitl-panel"></aside>
    </div>
  `;

  // Initialize
  await loadConversations();
  renderHITLPanel(document.getElementById('hitl-panel'), null);
  attachEventListeners();
}

// ---------- Event listeners ----------
function attachEventListeners() {
  // New chat button
  document.getElementById('new-chat-btn').addEventListener('click', createConversation);

  // Send button + textarea
  const textarea = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');

  sendBtn.addEventListener('click', () => sendMessage());

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  });

  // Export button
  document.getElementById('export-btn').addEventListener('click', exportConversation);
}

// ---------- Conversations ----------
async function loadConversations() {
  try {
    const res = await apiGet('/api/conversations?limit=200');
    conversations = res.conversations || [];
    renderConversationList();
  } catch (err) {
    console.error('Failed to load conversations:', err);
    toast.error('Failed to load conversations');
  }
}

function renderConversationList() {
  const list = document.getElementById('conversation-list');
  if (conversations.length === 0) {
    list.innerHTML = `
      <div class="empty-state" style="padding: 24px 16px">
        <div style="font-size: 24px; opacity: 0.5">💬</div>
        <p style="font-size: 12px; margin-top: 8px">No conversations yet</p>
      </div>
    `;
    return;
  }

  list.innerHTML = '';
  for (const conv of conversations) {
    const item = el('div', {
      class: `conversation-item ${activeConversation?.id === conv.id ? 'active' : ''}`,
      onclick: () => selectConversation(conv.id),
    });
    item.innerHTML = `
      <div class="conversation-item-content">
        <div class="conversation-item-title">${escapeHtml(conv.title)}</div>
        <div class="conversation-item-meta">
          ${conv.message_count} msgs · ${formatRelativeTime(conv.last_message_at || conv.created_at)}
        </div>
      </div>
      <button class="conversation-item-delete" title="Delete" data-id="${conv.id}">🗑️</button>
    `;
    // Delete button
    item.querySelector('.conversation-item-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    });
    list.append(item);
  }
}

async function createConversation() {
  try {
    const res = await apiPost('/api/conversations', { title: 'New conversation' });
    await loadConversations();
    await selectConversation(res.id);
    toast.success('New conversation created');
  } catch (err) {
    toast.error('Failed to create conversation');
  }
}

async function selectConversation(id) {
  // Close any existing SSE
  if (sseEventSource) {
    sseEventSource.close();
    sseEventSource = null;
  }
  isScanning = false;
  sseEvents = [];

  try {
    activeConversation = await apiGet(`/api/conversations/${id}`);
    renderConversationList();
    renderMessages();
    document.getElementById('chat-title').textContent = activeConversation.title || 'Untitled';

    // Update HITL panel with last scan_id if any
    const lastMsgWithScan = [...activeConversation.messages].reverse().find(m => m.scan_id);
    renderHITLPanel(document.getElementById('hitl-panel'), lastMsgWithScan?.scan_id || null);
  } catch (err) {
    toast.error('Failed to load conversation');
  }
}

async function deleteConversation(id) {
  showModal({
    title: 'Delete conversation?',
    body: 'This will permanently delete the conversation and all its messages. This action cannot be undone.',
    actions: [
      { label: 'Cancel', variant: 'secondary' },
      {
        label: 'Delete',
        variant: 'danger',
        onClick: async () => {
          try {
            await apiDelete(`/api/conversations/${id}`);
            if (activeConversation?.id === id) {
              activeConversation = null;
              renderMessages();
              document.getElementById('chat-title').textContent = 'VAPT-AI Chat';
              renderHITLPanel(document.getElementById('hitl-panel'), null);
            }
            await loadConversations();
            toast.success('Conversation deleted');
          } catch (err) {
            toast.error('Failed to delete conversation');
          }
        },
      },
    ],
  });
}

// ---------- Messages ----------
function renderMessages() {
  const container = document.getElementById('chat-messages');
  const messages = activeConversation?.messages || [];

  if (messages.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🛡️</div>
        <h2 class="empty-state-title">VAPT-AI Chat</h2>
        <p class="empty-state-description">
          Type your scan request below to start. The AI agent will automatically
          extract the target and run the scan.
        </p>
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  for (const msg of messages) {
    container.append(renderMessageBubble(msg));
  }

  // Scroll to bottom
  container.scrollTop = container.scrollHeight;
}

function renderMessageBubble(msg) {
  const isUser = msg.role === 'user';
  const isError = msg.metadata?.error === true;
  const avatarClass = isUser ? 'message-avatar-user' : isError ? 'message-avatar-error' : 'message-avatar-assistant';
  const avatarIcon = isUser ? '👤' : isError ? '⚠️' : '🤖';

  const bubble = el('div', { class: `message message-${msg.role}` });
  bubble.innerHTML = `
    <div class="message-avatar ${avatarClass}">${avatarIcon}</div>
    <div>
      <div class="message-bubble">${escapeHtml(msg.content)}</div>
      ${msg.scan_id ? `<div class="message-meta">scan: <code>${escapeHtml(msg.scan_id)}</code></div>` : ''}
      <div class="message-meta">${formatTime(msg.created_at)}</div>
    </div>
  `;
  return bubble;
}

// ---------- Send message + trigger scan ----------
async function sendMessage() {
  const textarea = document.getElementById('chat-input');
  const content = textarea.value.trim();
  if (!content || isScanning) return;

  // Auto-create conversation if none active
  let convId = activeConversation?.id;
  if (!convId) {
    try {
      const res = await apiPost('/api/conversations', { title: 'New conversation' });
      convId = res.id;
      await loadConversations();
      await selectConversation(convId);
    } catch (err) {
      toast.error('Failed to create conversation');
      return;
    }
  }

  // Clear input
  textarea.value = '';
  textarea.style.height = 'auto';

  // Add user message to DB
  let userMsg;
  try {
    userMsg = await apiPost(`/api/conversations/${convId}/messages`, { content });
  } catch (err) {
    toast.error('Failed to send message');
    return;
  }

  // Optimistic UI: append user message
  if (!activeConversation) activeConversation = { messages: [] };
  activeConversation.messages.push({ ...userMsg, role: 'user' });
  activeConversation.message_count++;
  renderMessages();
  document.getElementById('chat-title').textContent = activeConversation.title;

  // Start scan — CyberStrikeAI pattern: user types freely, AI extracts target.
  // Mode is auto-selected by orchestrator (Supervisor default). No manual scope input.
  const target = extractTarget(content);
  isScanning = true;
  activeScanId = null;
  sseEvents = [];
  updateScanStatus();

  // Add scanning indicator
  appendSSELog();

  try {
    const scanResult = await apiPost('/api/scans/start', {
      target,
      user_prompt: content,
    });
    activeScanId = scanResult.scan_id;

    // Update HITL panel to listen for this scan's events
    renderHITLPanel(document.getElementById('hitl-panel'), activeScanId);

    // Connect SSE
    connectSSE(scanResult.scan_id, convId, target);
  } catch (err) {
    isScanning = false;
    updateScanStatus();
    toast.error(`Failed to start scan: ${err.message}`);

    // Add error assistant message
    try {
      const errorMsg = await apiPost(`/api/conversations/${convId}/assistant-message`, {
        content: `Failed to start scan: ${err.message}`,
        metadata: { error: true },
      });
      activeConversation.messages.push({ ...errorMsg, role: 'assistant' });
      renderMessages();
    } catch (e) {
      console.error('Failed to persist error message:', e);
    }
  }
}

// ---------- SSE ----------
function connectSSE(scanId, convId, target) {
  if (sseEventSource) sseEventSource.close();

  sseEventSource = new EventSource(`/api/scans/${scanId}/events?token=${encodeURIComponent(getAccessToken() || '')}`);

  sseEventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      sseEvents.push(payload);
      if (sseEvents.length > 200) sseEvents = sseEvents.slice(-200);
      appendSSEEvent(payload);

      if (payload.event === 'scan_complete' || payload.event === 'scan_error') {
        sseEventSource.close();
        sseEventSource = null;
        isScanning = false;
        updateScanStatus();

        // Add assistant message with results
        finalizeScan(convId, scanId, target, payload);
      }
    } catch (err) {
      console.error('Failed to parse SSE event:', err);
    }
  };

  sseEventSource.onerror = () => {
    // EventSource auto-reconnects; only close on explicit complete/error
  };
}

function appendSSELog() {
  const container = document.getElementById('chat-messages');
  let log = document.getElementById('sse-log');
  if (!log) {
    log = el('div', { class: 'sse-log', id: 'sse-log' });
    container.append(log);
  }
  log.innerHTML = '<div style="color: var(--text-tertiary); margin-bottom: 4px">⏳ Scan in progress...</div>';
  container.scrollTop = container.scrollHeight;
}

function appendSSEEvent(event) {
  const log = document.getElementById('sse-log');
  if (!log) return;

  // Skip heartbeat events (they're just keepalive)
  if (event.event === 'heartbeat') return;

  const icon = getSSEIcon(event.event);
  // For scan_progress: tool_name contains the display text (from react_agent)
  // For scan_started: target contains the URL
  // For scan_complete: status + findings_count
  // For scan_error: error message
  let text = '';
  if (event.event === 'scan_progress') {
    text = event.tool_name || event.thought || '';
  } else if (event.event === 'scan_started') {
    text = event.target || '';
  } else if (event.event === 'scan_complete') {
    text = `Findings: ${event.findings_count ?? 0} | Duration: ${event.duration_seconds ? event.duration_seconds.toFixed(1) + 's' : 'N/A'}`;
  } else if (event.event === 'scan_error') {
    text = event.error || 'Unknown error';
  } else {
    text = event.thought || event.tool_name || event.error || event.comment || event.target || '';
  }

  const decisionBadge = event.decision ? ` <span class="badge ${event.decision === 'approve' ? 'badge-success' : event.decision === 'reject' ? 'badge-danger' : 'badge-warning'}">${event.decision}</span>` : '';

  const row = el('div', { class: 'sse-event' });
  row.innerHTML = `
    <span>${icon}</span>
    <span class="sse-event-type">[${event.event}]</span>
    <span class="sse-event-content">${escapeHtml(text)}${decisionBadge}</span>
  `;
  log.append(row);
  log.scrollTop = log.scrollHeight;

  // Also scroll the messages container
  const container = document.getElementById('chat-messages');
  container.scrollTop = container.scrollHeight;
}

function getSSEIcon(eventType) {
  const icons = {
    scan_started: '🚀',
    scan_progress: '⏳',
    finding_detected: '⚠️',
    hitl_approval_required: '🔒',
    hitl_decision_made: '✅',
    scan_complete: '✓',
    scan_error: '❌',
    heartbeat: '💓',
  };
  return icons[eventType] || '•';
}

async function finalizeScan(convId, scanId, target, event) {
  const isComplete = event.event === 'scan_complete';
  const content = isComplete
    ? `Scan completed for target: \`${target}\`\n\nStatus: ${event.status || 'completed'}\nFindings: ${event.findings_count ?? 'N/A'}\nDuration: ${event.duration_seconds ? event.duration_seconds.toFixed(1) + 's' : 'N/A'}\n\nScan ID: \`${scanId}\``
    : `Scan failed: ${event.error || 'unknown error'}\n\nScan ID: \`${scanId}\``;

  try {
    const msg = await apiPost(`/api/conversations/${convId}/assistant-message`, {
      content,
      scan_id: scanId,
      metadata: {
        scan_status: event.event,
        findings_count: event.findings_count,
        duration_seconds: event.duration_seconds,
        target,
        error: !isComplete,
      },
    });
    activeConversation.messages.push({ ...msg, role: 'assistant' });

    // Remove SSE log
    const log = document.getElementById('sse-log');
    if (log) log.remove();

    renderMessages();
    await loadConversations(); // refresh sidebar (title may have updated)
  } catch (err) {
    console.error('Failed to persist assistant message:', err);
  }
}

function updateScanStatus() {
  const status = document.getElementById('scan-status');
  const sendBtn = document.getElementById('send-btn');
  const textarea = document.getElementById('chat-input');

  if (isScanning) {
    status.classList.remove('hidden');
    sendBtn.disabled = true;
    textarea.disabled = true;
    textarea.placeholder = 'Scan in progress... (wait for completion)';
  } else {
    status.classList.add('hidden');
    sendBtn.disabled = false;
    textarea.disabled = false;
    textarea.placeholder = "Type your scan request... (e.g. 'Scan 10.10.10.5 for vulnerabilities')";
  }
}

// ---------- Export conversation ----------
function exportConversation() {
  if (!activeConversation || activeConversation.messages.length === 0) {
    toast.warning('No conversation to export');
    return;
  }

  showModal({
    title: 'Export conversation',
    body: 'Choose export format:',
    actions: [
      { label: 'Cancel', variant: 'secondary' },
      {
        label: 'Markdown',
        variant: 'primary',
        onClick: () => exportAsMarkdown(),
      },
      {
        label: 'PDF (print)',
        variant: 'primary',
        onClick: () => exportAsPrint(),
      },
    ],
  });
}

function exportAsMarkdown() {
  const conv = activeConversation;
  let md = `# ${conv.title}\n\n`;
  md += `**Created:** ${conv.created_at}\n`;
  md += `**Messages:** ${conv.message_count}\n\n---\n\n`;

  for (const msg of conv.messages) {
    const role = msg.role === 'user' ? '👤 **User**' : msg.role === 'assistant' ? '🤖 **Assistant**' : '⚙️ **System**';
    md += `### ${role} — ${formatTime(msg.created_at)}\n\n${msg.content}\n\n`;
    if (msg.scan_id) md += `*Scan ID: \`${msg.scan_id}\`*\n\n`;
  }

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation-${conv.id.slice(0, 8)}.md`;
  a.click();
  URL.revokeObjectURL(url);
  toast.success('Exported as Markdown');
}

function exportAsPrint() {
  const conv = activeConversation;
  const win = window.open('', '_blank');
  win.document.write(`
    <html><head><title>${escapeHtml(conv.title)}</title>
    <style>
      body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #1a1a1a; }
      h1 { border-bottom: 2px solid #6366f1; padding-bottom: 10px; }
      .msg { margin: 20px 0; padding: 12px; border-radius: 8px; }
      .msg-user { background: #eef2ff; }
      .msg-assistant { background: #f4f4f5; }
      .role { font-weight: 600; font-size: 13px; color: #6366f1; margin-bottom: 4px; }
      .content { white-space: pre-wrap; font-size: 14px; }
      .meta { font-size: 11px; color: #999; margin-top: 8px; }
      code { background: #f4f4f5; padding: 2px 4px; border-radius: 3px; font-size: 12px; }
    </style></head><body>
    <h1>${escapeHtml(conv.title)}</h1>
    <p style="color: #666">Created: ${conv.created_at} · ${conv.message_count} messages</p>
    ${conv.messages.map(m => `
      <div class="msg msg-${m.role}">
        <div class="role">${m.role === 'user' ? '👤 User' : '🤖 Assistant'} — ${formatTime(m.created_at)}</div>
        <div class="content">${escapeHtml(m.content)}</div>
        ${m.scan_id ? `<div class="meta">Scan ID: <code>${m.scan_id}</code></div>` : ''}
      </div>
    `).join('')}
    </body></html>
  `);
  win.document.close();
  setTimeout(() => win.print(), 500);
  toast.success('Opening print dialog...');
}