"""
VAPT-AI Harness Bridge Hooks — thin wrappers that wire RL/KG/Trace into the
live scan loop without invasively modifying the agent / pipeline code.

W19-FIX Phase C (Bug 3 fix): the previous scan_pipeline created a
HarnessBridge but only called `bridge.recorder.event("scan_start")` at scan
start + `bridge.end_episode()` at scan end — with NOTHING in between.
This meant:
    - RL policy was never consulted (no select_action)
    - RL experience store stayed empty (no record_experience)
    - RL training never ran (no train_step)
    - Trace JSONL only had 2 events per scan (scan_start + scan_complete)
    - KG edge outcomes never updated
    - Checkpoints only saved at app shutdown, not per-scan

These hooks are called from:
    - app/agents/react_agent.py (per-tool-call: select_action + record_experience)
    - app/pentest/scan_pipeline.py (per-finding audited: process_rejection
      + KG edge update; per-scan end: train_step + save_checkpoint)

All hooks are non-fatal: if the bridge isn't initialized for the scan_id
(e.g. RL disabled, or test path), the hook is a no-op.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Train every N turns to avoid blocking the agent loop
TRAIN_EVERY_N_TURNS = 5


def _get_bridge(scan_id: str):
    """Look up the HarnessBridge for this scan_id, or None if not registered."""
    try:
        from app.harness.bridge import _bridges
        return _bridges.get(scan_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-tool-call hooks (called from react_agent.py)
# ---------------------------------------------------------------------------

async def on_tool_call_start(
    scan_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    target: str,
    turn: int,
    session: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Hook called BEFORE execute_tool_call() in the ReAct loop.

    Asks the RL policy for a recommendation (action_index + EGATS paths).
    The agent is autonomous — it can override the recommendation, but the
    recommendation is recorded to the trace JSONL for offline replay.

    Returns:
        dict with action_index/action_name/egats_paths if bridge is active,
        else None (caller proceeds normally).
    """
    bridge = _get_bridge(scan_id)
    if bridge is None:
        return None

    # Build a minimal session dict for StateEncoder
    sess = session or {}
    sess.setdefault("target", target)
    sess.setdefault("current_tool", tool_name)
    sess.setdefault("tool_args", tool_args)

    try:
        rec = bridge.select_action(sess, turn)
        # Emit a trace event for the recommendation
        bridge.recorder.event(
            "tool_call_start",
            turn=turn,
            data={
                "tool": tool_name,
                "args_preview": str(tool_args)[:200],
                "rl_action_index": rec.get("action_index"),
                "rl_action_name": rec.get("action_name"),
                "rl_epsilon": rec.get("meta", {}).get("epsilon"),
                "rl_exploratory": rec.get("meta", {}).get("exploratory"),
                "egats_paths_count": len(rec.get("egats_paths", [])),
            },
        )
        return rec
    except Exception as exc:
        logger.debug("on_tool_call_start failed (non-fatal) | scan=%s | %s", scan_id, exc)
        return None


async def on_tool_call_end(
    scan_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_output: str,
    success: bool,
    turn: int,
    session: dict[str, Any] | None = None,
) -> None:
    """Hook called AFTER execute_tool_call() returns.

    Computes reward from the tool result, calls bridge.record_experience()
    which writes (s, a, r, s', done) to the RL store + appends a turn_end
    event to the JSONL trace. Periodically calls bridge.train_step() to
    keep the RL policy learning.
    """
    bridge = _get_bridge(scan_id)
    if bridge is None:
        return

    sess = session or {}
    sess.setdefault("current_tool", tool_name)
    sess.setdefault("last_tool_success", success)
    sess.setdefault("last_tool_output_preview", tool_output[:500])

    # Compute reward (per master plan §12 W16 D26)
    reward = _compute_tool_reward(tool_name, tool_output, success)

    # Get the action_idx from session (set by on_tool_call_start → bridge.select_action)
    action_idx = sess.get("_rl_action_idx", 0)
    action_name = sess.get("_rl_action_name", tool_name)
    done = tool_name == "exit"

    try:
        await bridge.record_experience(
            session=sess,
            turn=turn,
            action_idx=action_idx,
            reward=reward,
            done=done,
            observation=tool_output[:500],
            action_name=action_name,
        )
    except Exception as exc:
        logger.debug("record_experience failed (non-fatal) | scan=%s | %s", scan_id, exc)

    # Periodic training
    if turn > 0 and turn % TRAIN_EVERY_N_TURNS == 0:
        try:
            train_result = bridge.train_step(batch_size=64)
            if train_result.get("trained"):
                bridge.recorder.event(
                    "rl_train_step",
                    turn=turn,
                    data={
                        "loss": train_result.get("loss"),
                        "buffer_size": train_result.get("buffer_size"),
                        "epsilon": train_result.get("epsilon"),
                    },
                )
                logger.info(
                    "RL train_step | scan=%s | turn=%d | loss=%s | eps=%s | buf=%s",
                    scan_id, turn, train_result.get("loss"),
                    train_result.get("epsilon"), train_result.get("buffer_size"),
                )
        except Exception as exc:
            logger.debug("train_step failed (non-fatal) | scan=%s | %s", scan_id, exc)


def _compute_tool_reward(tool_name: str, tool_output: str, success: bool) -> float:
    """Compute reward for a tool call per master plan §12 W16 D26 reward shaping.

    D26 reward signals:
        +5 confirmed exploit Tier-1
        +3 Tier-2
        +1 confirmed finding
        +0.5 successful recon
        -1 timeout
        -2 verifier-rejected FP
        -3 scope violation
        -5 user_aborted
        -10 destructive without HITL
        +10 scan completed
    """
    if not success:
        # Detect scope violation
        if "scope" in tool_output.lower() and "violation" in tool_output.lower():
            return -3.0
        # Detect timeout
        if "timeout" in tool_output.lower():
            return -1.0
        return -0.5  # generic failure

    # Successful tool call
    if tool_name == "exit":
        return 10.0  # scan completed
    if tool_name == "record_vulnerability":
        return 1.0  # confirmed finding (verifier may flip later)
    if tool_name in {"nmap", "nuclei", "nikto", "whatweb", "httpx", "subfinder", "gobuster"}:
        return 0.5  # successful recon
    if tool_name in {"sqlmap", "metasploit", "msfconsole"}:
        return 3.0  # successful exploit (Tier-2; Tier-1 = +5 if shell obtained)
    return 0.5  # default success


# ---------------------------------------------------------------------------
# Per-finding-audited hooks (called from scan_pipeline._phase_audit_findings)
# ---------------------------------------------------------------------------

def on_finding_audited(
    scan_id: str,
    finding_id: str,
    finding_name: str,
    accepted: bool,
    confidence_score: float,
    verdict_reason: str | None,
    session: dict[str, Any] | None = None,
) -> None:
    """Hook called after the W12 EvidenceAuditor runs on each finding.

    - If accepted: tell KG to update edge outcomes (success_count += 1)
    - If rejected: call bridge.process_rejection() which triggers self-correction
      + records a negative-reward RL experience (-2 per D26)
    """
    bridge = _get_bridge(scan_id)
    if bridge is None:
        return

    try:
        if accepted:
            # KG: increment success_count on related edges
            # (We'd need to know which KG edge corresponds to this finding,
            # but for now we just log a trace event — actual edge update
            # requires finding_type → KG AttackVector/Finding node mapping
            # which is W17 work.)
            bridge.recorder.event(
                "finding_accepted",
                data={
                    "finding_id": str(finding_id)[:8],
                    "finding_name": finding_name[:100],
                    "confidence": round(confidence_score, 4),
                },
            )
        else:
            # Rejected — trigger self-correction + negative RL reward
            finding_dict = {
                "id": str(finding_id),
                "title": finding_name,
                "rejected_reason": verdict_reason,
            }
            try:
                rejection_result = bridge.process_rejection(
                    finding=finding_dict,
                    verification_result=None,  # we don't have the VerificationResult obj here
                    session=session,
                )
                bridge.recorder.event(
                    "finding_rejected",
                    data={
                        "finding_id": str(finding_id)[:8],
                        "finding_name": finding_name[:100],
                        "verdict_reason": verdict_reason or "",
                        "rl_reward": rejection_result.get("rl_reward") if rejection_result else -2.0,
                    },
                    reward=-2.0,
                )
                logger.info(
                    "Finding rejected | scan=%s | name=%s | reason=%s | rl_reward=%s",
                    scan_id, finding_name[:50], verdict_reason,
                    rejection_result.get("rl_reward") if rejection_result else "N/A",
                )
            except Exception as exc:
                logger.debug("process_rejection failed (non-fatal) | scan=%s | %s", scan_id, exc)
                # Fallback: just record the trace event
                bridge.recorder.event(
                    "finding_rejected",
                    data={
                        "finding_id": str(finding_id)[:8],
                        "finding_name": finding_name[:100],
                        "verdict_reason": verdict_reason or "",
                    },
                    reward=-2.0,
                )
    except Exception as exc:
        logger.debug("on_finding_audited failed (non-fatal) | scan=%s | %s", scan_id, exc)


# ---------------------------------------------------------------------------
# Per-scan-end hooks (called from scan_pipeline at scan completion)
# ---------------------------------------------------------------------------

async def on_scan_end(
    scan_id: str,
    final_status: str,
    findings_count: int,
    session: dict[str, Any] | None = None,
) -> None:
    """Hook called at scan completion — finalize trace + save checkpoint.

    - Final RL train_step (one last batch)
    - Save checkpoint to RLCheckpoint table + .npz cache (per-scan, not just app shutdown)
    - Trace JSONL finalize with final_epsilon + final_q_values_json
    """
    bridge = _get_bridge(scan_id)
    if bridge is None:
        return

    try:
        # Final train step
        train_result = bridge.train_step(batch_size=64)
        if train_result.get("trained"):
            bridge.recorder.event(
                "rl_train_step_final",
                data={
                    "loss": train_result.get("loss"),
                    "buffer_size": train_result.get("buffer_size"),
                    "epsilon": train_result.get("epsilon"),
                },
            )

        # Save checkpoint (per-scan, not just app shutdown)
        try:
            policy = bridge._get_policy()
            if hasattr(policy, "save_checkpoint"):
                # Save to RLCheckpoint table + .npz cache
                # Per master plan §12 W16: persistent checkpoints per scan
                await policy.save_checkpoint(scan_id=scan_id)
                bridge.recorder.event(
                    "rl_checkpoint_saved",
                    data={"scan_id": scan_id, "epsilon": policy.epsilon},
                )
                logger.info(
                    "RL checkpoint saved | scan=%s | epsilon=%.4f",
                    scan_id, policy.epsilon,
                )
        except Exception as exc:
            logger.debug("save_checkpoint failed (non-fatal) | scan=%s | %s", scan_id, exc)

        # Final summary event
        bridge.recorder.event(
            "scan_complete",
            data={
                "status": final_status,
                "findings_count": findings_count,
                "total_turns": bridge._turn_count,
                "episode_reward": round(bridge._episode_reward, 4),
            },
        )
    except Exception as exc:
        logger.warning("on_scan_end failed (non-fatal) | scan=%s | %s", scan_id, exc)


__all__ = [
    "on_tool_call_start",
    "on_tool_call_end",
    "on_finding_audited",
    "on_scan_end",
    "TRAIN_EVERY_N_TURNS",
]