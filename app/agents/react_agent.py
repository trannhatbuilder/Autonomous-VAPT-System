"""
VAPT-AI Real ReAct Agent — LLM + Tools + SSE events.

This is the core scan engine. Replaces the W9 stub with a real ReAct loop
that mirrors CyberStrikeAI's agent flow:

    user message → LLM (with tools) → tool_calls → execute tools →
    tool_results → LLM → ... → exit tool → done

Each iteration emits SSE events (scan_progress) so the frontend chat
shows real-time AI thinking + tool calls + results.

Architecture:
    1. Build system prompt (Vietnamese — per master plan §2.3)
    2. Build tool schemas (32 security tools + record_vulnerability + exit)
    3. P1 Preflight: 1-shot LLM ping to verify OpenAI API key works
       (fail fast instead of cryptic failure mid-loop)
    4. Loop:
        a. Call LLM via chat_completion with messages + tools
        b. If LLM returns tool_calls:
            - Emit SSE: scan_progress (thought + tool_name)
            - Execute each tool via SubprocessExecutor
            - P1: detect "Binary not found" → inject install hint into
              observation so LLM knows the tool isn't installed (and
              can recover by switching to a different tool)
            - Emit SSE: scan_progress (observation)
            - Append tool results to messages
            - Continue loop
        c. If LLM returns content only (no tool_calls):
            - LLM is done thinking — append assistant message
            - Continue (LLM may call exit tool next)
        d. If LLM calls "exit" tool:
            - Scan complete — return summary
        e. Max iterations reached:
            - Return what we have
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, UTC
from typing import Any

from app.agents.llm_client import chat_completion
from app.agents.tool_bridge import build_tool_schemas, execute_tool_call, create_executor

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30  # D18 guardrail: max 30 decisions per scan


# ── P1: preflight LLM check ──────────────────────────────────────────────

async def _preflight_llm_check(llm_config: dict[str, Any]) -> tuple[bool, str]:
    """P1: verify the OpenAI API key works before entering the ReAct loop.

    Sends a 1-token "ping" message to the configured model. If the call
    succeeds (any 2xx response), the API key + base URL + model name are
    all valid. If it fails, we return a human-readable error instead of
    crashing mid-loop after spending tokens on the system prompt.

    Returns:
        (ok, error_message) — error_message is "" when ok=True.
    """
    try:
        response = await chat_completion(
            llm_config=llm_config,
            messages=[
                {"role": "user", "content": "ping"},
            ],
            tools=None,  # no tools — just a minimal call
            temperature=0.0,
        )
        # A 2xx response proves the key/base_url/model are valid. Reasoning
        # models may leave `content` empty (their answer sits in
        # `reasoning_content`), so accept reasoning output and usage too.
        content = (response.get("content") or "").strip()
        reasoning = (response.get("reasoning_content") or "").strip()
        if content or reasoning or response.get("tool_calls") or response.get("usage"):
            return True, ""
        return False, "LLM returned empty response — check provider config"
    except Exception as exc:
        msg = str(exc)
        # Common error patterns → friendlier messages
        if "401" in msg or "Invalid API key" in msg or "Incorrect API key" in msg:
            return False, (
                f"OpenAI API key invalid or unauthorized. "
                f"Go to Settings → AI Channels and re-enter the key. "
                f"Underlying error: {msg}"
            )
        if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
            return False, (
                f"Model {llm_config.get('model', '?')} not found on this "
                f"provider. Check the model name in Settings. "
                f"Underlying error: {msg}"
            )
        if "Connection" in msg or "timeout" in msg.lower() or "connect" in msg.lower():
            base_url = llm_config.get("base_url", "default")
            return False, (
                f"Cannot reach LLM endpoint ({base_url}). "
                f"Check network + base_url setting. "
                f"Underlying error: {msg}"
            )
        return False, f"LLM preflight failed: {msg}"


# ── P1: install hint for missing binaries ────────────────────────────────

# Maps binary names → install instructions for the LLM observation message.
# When SubprocessExecutor.execute() returns error="Binary not found: X",
# we replace the raw error with this friendly hint so the LLM knows the tool
# isn't installed on the host and can switch to a different available tool
# (or stop attempting this tool family).
BINARY_INSTALL_HINTS: dict[str, str] = {
    "nmap":         "nmap is not installed on the VAPT-AI host. Install with: `apt install nmap` (Debian/Ubuntu).",
    "nuclei":       "nuclei is not installed. Install with: `go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest` and add $GOPATH/bin to PATH.",
    "sqlmap":       "sqlmap is not installed. Install with: `apt install sqlmap` or `git clone --depth 1 https://github.com/sqlmapproject/sqlmap.git /opt/sqlmap`.",
    "ffuf":         "ffuf is not installed. Install with: `go install github.com/ffuf/ffuf/v2@latest`.",
    "gobuster":     "gobuster is not installed. Install with: `apt install gobuster`.",
    "feroxbuster":  "feroxbuster is not installed. Install from: https://github.com/epi052/feroxbuster/releases.",
    "httpx":        "httpx (projectdiscovery) is not installed. Install with: `go install github.com/projectdiscovery/httpx/cmd/httpx@latest`.",
    "whatweb":      "whatweb is not installed. Install with: `apt install whatweb`.",
    "nikto":        "nikto is not installed. Install with: `apt install nikto` or `git clone https://github.com/sullo/nikto.git`.",
    "dalfox":       "dalfox is not installed. Install with: `go install github.com/hahwul/dalfox@latest`.",
    "wpscan":       "wpscan is not installed. Install from: https://github.com/wpscanner/wpscan#installers (gem install wpscan).",
    "subfinder":    "subfinder is not installed. Install with: `go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest`.",
    "katana":       "katana is not installed. Install with: `go install github.com/projectdiscovery/katana/cmd/katana@latest`.",
    "amass":        "amass is not installed. Install with: `go install -v github.com/owasp-amass/amass/v4/...@master`.",
    "masscan":      "masscan is not installed. Install with: `apt install masscan` (needs root for raw sockets).",
    "rustscan":     "rustscan is not installed. Install with: `cargo install rustscan` or use the docker image `rustscan/rustscan:latest`.",
    "gau":          "gau is not installed. Install with: `go install github.com/lc/gau/v2/cmd/gau@latest`.",
    "waybackurls":  "waybackurls is not installed. Install with: `go install github.com/tomnomnom/waybackurls@latest`.",
    "hydra":        "hydra is not installed. Install with: `apt install hydra`.",
    "hashcat":      "hashcat is not installed. Install with: `apt install hashcat`.",
    "john":         "john (John the Ripper) is not installed. Install with: `apt install john`.",
    "fierce":       "fierce is not installed. Install with: `apt install fierce` or `gem install fierce`.",
    "dnsenum":      "dnsenum is not installed. Install with: `apt install dnsenum`.",
    "impacket-smbexec":       "impacket is not installed. Install with: `pipx install impacket` (or `pip install impacket`).",
    "impacket-psexec":         "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-wmiexec":        "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-secretsdump":    "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-GetUserSPNs":   "impacket is not installed. Install with: `pipx install impacket`.",
    "netexec":      "netexec (a.k.a. nxc) is not installed. Install with: `pipx install netexec` or `pip install netexec`.",
    "responder":    "responder is not installed. Install with: `apt install responder` or `git clone https://github.com/lgandx/Responder.git`.",
    "linpeas":      "linpeas is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (LinPEAS).",
    "winpeas.exe":  "winpeas.exe is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (WinPEAS).",
    "mimikatz.exe": "mimikatz.exe is not installed. Download from: https://github.com/gentilkiwi/mimikatz/releases.",
}


def _format_binary_not_found_hint(error_or_output: str) -> str:
    """If input contains 'Binary not found: X', return output with install hint.

    Used in two flows:
        1. Exception path: error = "Binary not found: nmap" (raised)
        2. Tool-result path: output = "...\\n[error] Binary not found: nmap"
           (returned by execute_tool_call embedding SubprocessExecutor's
           ToolResult.error string)

    For flow 2, preserves the original output text + appends the install
    hint as a clear "HINT" block so the LLM understands both what happened
    and what to do next.

    If no "Binary not found" marker, returns the input unchanged.
    """
    if "Binary not found" not in error_or_output:
        return error_or_output

    # Extract binary name from "Binary not found: <name>"
    # Use rfind to handle "Binary not found:" appearing in error context.
    marker_idx = error_or_output.find("Binary not found:")
    after_marker = error_or_output[marker_idx + len("Binary not found:"):]
    # Take first whitespace-delimited token (could have leading space)
    try:
        binary = after_marker.strip().split()[0].rstrip(",.;:")
    except Exception:
        return error_or_output

    hint = BINARY_INSTALL_HINTS.get(binary)
    if not hint:
        hint = (
            f"binary '{binary}' is not installed on the VAPT-AI host. "
            f"Ask the operator to install it, or try a different available tool. "
            f"Do NOT retry the same tool."
        )

    # Append the hint to the existing output/error text
    return (
        f"{error_or_output}\n\n"
        f"───── INSTALL HINT ─────\n"
        f"{hint}\n"
        f"───── END HINT ─────\n"
    )


# ── System prompt (Vietnamese per master plan §2.3) ─────────────────────

SYSTEM_PROMPT = """Bạn là VAPT-AI — một AI agent chuyên kiểm thử bảo mật (penetration testing).

Nhiệm vụ: Thực hiện quét bảo mật trên target mà người dùng cung cấp. Bạn có quyền
đã được xác thực trước (consent form đã được chấp nhận). KHÔNG cần hỏi lại quyền.

Quy trình scan:
1. RECON: Dùng nmap, httpx, whatweb để khám phá target (port, service, tech stack)
2. VULN SCAN: Dùng nuclei, nikto, dalfox để tìm lỗ hổng
3. EXPLOIT: Dùng sqlmap, ffuf, gobuster để verify lỗ hổng (chỉ với HITL approval)
4. RECORD: Khi tìm thấy lỗ hổng verified, gọi record_vulnerability để ghi nhận
5. EXIT: Khi đã scan xong, gọi exit với summary

Quy tắc:
- Luôn bắt đầu bằng recon (nmap + httpx) trước khi scan lỗ hổng
- Chỉ ghi nhận finding khi CÓ EVIDENCE (tool output chứng minh)
- KHÔNG bịa đặt lỗ hổng — chỉ báo cáo những gì tool phát hiện
- Gọi record_vulnerability cho MỖI lỗ hổng tìm được
- Khi xong, gọi exit với tóm tắt kết quả

Bạn có các công cụ (tools) sau:
- nmap: port scan, service detection
- nuclei: vulnerability scanner (CVE, misconfigurations)
- sqlmap: SQL injection detection + exploitation
- httpx: HTTP probe, tech detection
- whatweb: web tech fingerprinting
- nikto: web server vulnerability scanner
- dalfox: XSS scanner
- gobuster/ffuf/feroxbuster: directory/file brute-force
- nikto: web server scanner
- record_vulnerability: ghi nhận lỗ hổng
- exit: kết thúc scan

QUAN TRỌNG: Hãy thực sự CHẠY các tool (gọi tool function), KHÔNG chỉ mô tả
những gì bạn sẽ làm. Mỗi lỗ hổng phải được ghi nhận qua record_vulnerability.
"""


async def run_react_scan(
    target: str,
    user_prompt: str,
    llm_config: dict[str, Any],
    scan_id: str,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run a real ReAct scan loop.

    Args:
        target: target URL or IP (e.g. "http://example.com")
        user_prompt: user's natural-language scan request
        llm_config: LLM config from DB (provider, api_key, model, ...)
        scan_id: scan ID for SSE events + DB attribution
        max_iterations: max ReAct iterations (D18: default 30)

    Returns:
        dict with:
            status: "completed" | "max_iterations" | "error"
            findings_count: int
            iterations: int
            total_tokens: int
            final_summary: str
            error: str | None
    """
    from app.pentest.events import emit_scan_progress

    # ── P1: preflight LLM check ──────────────────────────────────
    # Send a 1-token "ping" to verify the OpenAI API key + model name +
    # base_url are all valid. If not, fail fast with a friendly error
    # instead of crashing mid-loop after spending tokens on the system
    # prompt.
    await emit_scan_progress(scan_id, thought="Đang kiểm tra cấu hình LLM (OpenAI API key, model, base_url)...",
                             agent_name="orchestrator", progress=5)

    ok, err = await _preflight_llm_check(llm_config)
    if not ok:
        logger.error("LLM preflight failed | scan=%s | error=%s", scan_id, err)
        await emit_scan_progress(scan_id, thought=f"❌ LLM preflight thất bại: {err}",
                                 agent_name="orchestrator", progress=5)
        return {
            "status": "error",
            "findings_count": 0,
            "iterations": 0,
            "total_tokens": 0,
            "final_summary": f"LLM preflight failed: {err}",
            "error": err,
        }

    logger.info("LLM preflight OK | scan=%s | model=%s",
                scan_id, llm_config.get("model"))
    await emit_scan_progress(scan_id, thought="✅ LLM cấu hình OK. Bắt đầu ReAct loop...",
                             agent_name="orchestrator", progress=10)

    # Build tool schemas
    tool_schemas = build_tool_schemas()

    # Create executor for this scan
    executor = create_executor(target, scan_id)

    # Build initial messages
    user_message = f"Target: {target}\n\nYêu cầu: {user_prompt}\n\nHãy bắt đầu scan target này."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    total_tokens = 0
    findings_count = 0
    iterations = 0
    final_summary = ""

    logger.info("ReAct scan started | scan=%s | target=%s | max_iter=%d",
                scan_id, target, max_iterations)

    # ── ReAct loop ──────────────────────────────────────────────
    for iteration in range(max_iterations):
        iterations = iteration + 1

        # ── Phase D: abort check ──────────────────────────────────
        # Honor the panic button BEFORE the next LLM call — without this,
        # aborting a scan only takes effect after max_iterations.
        from app.pentest.scan_registry import scan_registry
        if await scan_registry.is_aborted(scan_id):
            logger.warning(
                "ReAct scan aborting — scan_registry.abort_event is set | scan=%s | iter=%d",
                scan_id, iterations,
            )
            return {
                "status": "aborted",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan aborted at iteration {iterations} (panic button).",
                "error": "user_panic_button",
            }

        try:
            # Call LLM
            await emit_scan_progress(scan_id, thought=f"Đang suy nghĩ... (vòng {iterations}/{max_iterations})",
                                     agent_name="orchestrator", progress=20 + iteration * 2)

            response = await chat_completion(
                llm_config=llm_config,
                messages=messages,
                tools=tool_schemas,
            )

            total_tokens += response["usage"].get("total_tokens", 0)

            # Append assistant message to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response["content"]}
            if response["tool_calls"]:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in response["tool_calls"]
                ]
            messages.append(assistant_msg)

            # If LLM is thinking (content but no tool_calls), continue to next iteration
            if not response["tool_calls"]:
                if response["content"]:
                    logger.info("LLM thinking (no tool calls) | scan=%s | iter=%d | content=%s",
                                scan_id, iterations, response["content"][:100])
                    await emit_scan_progress(scan_id, thought=response["content"][:200],
                                             agent_name="orchestrator", progress=20 + iteration * 2)
                continue

            # ── Execute tool calls ───────────────────────────────
            for tc in response["tool_calls"]:
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                # Emit SSE: tool_call
                await emit_scan_progress(scan_id, thought=f"🔧 Đang chạy: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})",
                                         agent_name="orchestrator", tool_name=tool_name,
                                         progress=20 + iteration * 2)

                logger.info("Tool call | scan=%s | iter=%d | tool=%s | args=%s",
                            scan_id, iterations, tool_name,
                            json.dumps(tool_args, ensure_ascii=False)[:80])

                # Check for exit tool
                if tool_name == "exit":
                    final_summary = tool_args.get("summary", "Scan complete.")
                    logger.info("Scan exit | scan=%s | iter=%d | summary=%s",
                                scan_id, iterations, final_summary[:100])
                    await emit_scan_progress(scan_id, thought=f"✅ Hoàn thành: {final_summary[:150]}",
                                             agent_name="orchestrator", progress=95)

                    # Count findings from DB
                    try:
                        from app.db.session import async_session
                        from app.db.models.pentest import Finding
                        from sqlalchemy import select, func
                        async with async_session() as session:
                            count = await session.scalar(
                                select(func.count(Finding.id)).where(Finding.scan_id == scan_id)
                            ) or 0
                            findings_count = count
                    except Exception:
                        pass

                    return {
                        "status": "completed",
                        "findings_count": findings_count,
                        "iterations": iterations,
                        "total_tokens": total_tokens,
                        "final_summary": final_summary,
                        "error": None,
                    }

                # Execute the tool
                try:
                    tool_output = await execute_tool_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        target=target,
                        scan_id=scan_id,
                        executor=executor,
                    )

                    # P1: detect "Binary not found" inside the returned
                    # tool_output (SubprocessExecutor doesn't raise — it
                    # returns ToolResult.error embedded in the string).
                    # Inject install hint so the LLM can recover.
                    if "Binary not found" in tool_output:
                        tool_output = _format_binary_not_found_hint(tool_output)

                    if tool_name == "record_vulnerability":
                        findings_count += 1
                        await emit_scan_progress(scan_id, thought=f"📋 Đã ghi nhận lỗ hổng #{findings_count}",
                                                 agent_name="orchestrator", tool_name="record_vulnerability",
                                                 progress=20 + iteration * 2)
                    else:
                        # Emit observation (truncated for SSE)
                        obs_preview = tool_output[:200].replace("\n", " ")
                        await emit_scan_progress(scan_id, thought=f"📤 Kết quả {tool_name}: {obs_preview}",
                                                 agent_name="orchestrator", tool_name=tool_name,
                                                 progress=20 + iteration * 2)

                except Exception as exc:
                    tool_output = f"Error executing {tool_name}: {exc}"
                    # P1: if the error is "Binary not found" raised as an
                    # exception (rare — usually returned in tool_output),
                    # enrich with an install hint.
                    err_str = str(exc)
                    if "Binary not found" in err_str or "FileNotFoundError" in err_str:
                        tool_output = _format_binary_not_found_hint(err_str)
                    logger.error("Tool execution error | scan=%s | tool=%s | error=%s",
                                 scan_id, tool_name, exc)
                    await emit_scan_progress(scan_id, thought=f"❌ Lỗi {tool_name}: {str(exc)[:100]}",
                                             agent_name="orchestrator", tool_name=tool_name,
                                             progress=20 + iteration * 2)

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

        except Exception as exc:
            logger.error("ReAct loop error | scan=%s | iter=%d | error=%s",
                         scan_id, iterations, exc)
            return {
                "status": "error",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan failed at iteration {iterations}: {exc}",
                "error": str(exc),
            }

    # Max iterations reached
    logger.warning("Max iterations reached | scan=%s | iter=%d", scan_id, iterations)
    await emit_scan_progress(scan_id, thought=f"⚠️ Đạt giới hạn {max_iterations} vòng. Kết thúc scan.",
                             agent_name="orchestrator", progress=90)

    return {
        "status": "max_iterations",
        "findings_count": findings_count,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "final_summary": f"Scan reached max iterations ({max_iterations}). Partial results may be available.",
        "error": None,
    }


__all__ = [
    "run_react_scan",
    "SYSTEM_PROMPT",
    "MAX_ITERATIONS",
]