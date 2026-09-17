/**
 * VAPT-AI HITL Panel — W7-D-v2.
 *
 * Right panel showing real-time HITL decisions + panic button.
 * Listens to SSE events for the active scan.
 */

import { apiPost, el, escapeHtml, toast, showModal, getAccessToken } from './utils.js';

let hitlEventSource = null;
let hitlDecisions = [];
let currentHitlScanId = null;

export function renderHITLPanel(container, scanId) {
  // Close existing SSE
  if (hitlEventSource) {
    hitlEventSource.close();
    hitlEventSource = null;
  }

  currentHitlScanId = scanId;
  hitlDecisions = [];

  container.innerHTML = `
    <div class="hitl-panel-header">
      <div class="hitl-panel-title">
        <span>🛡️</span>
        <span>HITL Decisions</span>
      </div>
      ${scanId ? `<button class="btn btn-danger btn-sm" id="panic-btn">⚠ Abort</button>` : ''}
    </div>
    <div class="hitl-decisions" id="hitl-decisions">
      <div class="hitl-empty">
        <div style="font-size: 32px; opacity: 0.3; margin-bottom: 8px">🛡️</div>
        <p>No HITL decisions yet.</p>
        <p style="font-size: 11px; margin-top: 4px">Decisions will appear here when the audit agent reviews destructive operations.</p>
      </div>
    </div>
  `;

  // Attach panic button
  const panicBtn = document.getElementById('panic-btn');
  if (panicBtn) {
    panicBtn.addEventListener('click', triggerPanic);
  }

  // Connect SSE if scanId provided
  if (scanId) {
    connectHITLSSE(scanId);
  }
}

function connectHITLSSE(scanId) {
  if (hitlEventSource) hitlEventSource.close();

  try {
    hitlEventSource = new EventSource(`/api/scans/${scanId}/events?token=${encodeURIComponent(getAccessToken() || '')}`);

    hitlEventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);

        if (payload.event === 'hitl_decision_made') {
          hitlDecisions.unshift({
            id: payload.hitl_id + Date.now(),
            hitl_id: payload.hitl_id,
            tool_name: payload.tool_name,
            target: payload.target,
            decision: payload.decision,
            decided_by: payload.decided_by,
            comment: payload.comment,
            suggested_args: payload.suggested_args,
            timestamp: Date.now(),
          });
          if (hitlDecisions.length > 50) hitlDecisions = hitlDecisions.slice(0, 50);
          renderDecisions();
        }
      } catch (err) {
        console.error('Failed to parse HITL SSE event:', err);
      }
    };
  } catch (err) {
    console.error('Failed to connect HITL SSE:', err);
  }
}

function renderDecisions() {
  const container = document.getElementById('hitl-decisions');
  if (!container) return;

  if (hitlDecisions.length === 0) {
    container.innerHTML = `
      <div class="hitl-empty">
        <div style="font-size: 32px; opacity: 0.3; margin-bottom: 8px">🛡️</div>
        <p>No HITL decisions yet.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  for (const d of hitlDecisions) {
    const card = el('div', { class: 'hitl-decision-card' });
    const badgeClass = d.decision === 'approve' ? 'badge-success'
      : d.decision === 'reject' ? 'badge-danger'
      : d.decision === 'suggest_alternative' ? 'badge-warning'
      : 'badge-info';

    card.innerHTML = `
      <div class="hitl-decision-header">
        <span class="hitl-decision-tool">${escapeHtml(d.tool_name)}</span>
        <span class="badge ${badgeClass}">${d.decision}</span>
      </div>
      <div class="text-sm text-tertiary">target: ${escapeHtml(d.target)}</div>
      <div class="text-sm text-secondary" style="margin-top: 4px">${escapeHtml(d.comment || '')}</div>
      <div class="text-sm text-tertiary" style="margin-top: 4px">by: ${escapeHtml(d.decided_by)}</div>
    `;
    container.append(card);
  }
}

async function triggerPanic() {
  if (!currentHitlScanId) return;

  showModal({
    title: 'Abort scan immediately?',
    body: `
      <p>This will <strong>immediately stop</strong> the scan and:</p>
      <ul style="margin: 12px 0; padding-left: 20px">
        <li>Kill all running subprocesses (SIGKILL)</li>
        <li>Run cleanup script (removes sqlmap/msf/nuclei artifacts)</li>
        <li>Mark the scan as aborted</li>
      </ul>
      <p style="font-size: 12px; color: var(--text-tertiary)">Scan ID: <code>${currentHitlScanId}</code></p>
      <p style="margin-top: 12px"><strong>This action cannot be undone.</strong></p>
    `,
    actions: [
      { label: 'Cancel', variant: 'secondary' },
      {
        label: 'Yes, abort now',
        variant: 'danger',
        onClick: async () => {
          try {
            const result = await apiPost(`/api/scans/${currentHitlScanId}/abort?reason=user_panic_button&run_cleanup=true`, {});
            if (result.aborted) {
              const killed = result.killed_subprocess_pids?.length || 0;
              const cleanup = result.cleanup?.success ? 'OK' : 'failed';
              toast.success(`Scan aborted — killed ${killed} subprocess(es), cleanup ${cleanup}`);
            } else {
              toast.info(`Scan not aborted: ${result.message || result.reason}`);
            }
          } catch (err) {
            toast.error(`Failed to abort: ${err.message}`);
          }
        },
      },
    ],
  });
}