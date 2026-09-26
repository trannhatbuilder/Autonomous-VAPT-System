#!/usr/bin/env python3
"""
VAPT-AI base.py version verifier.

Run from anywhere (venv active or not):
    python verify_base.py /path/to/your/local/base.py

Or via curl download + run:
    python verify_base.py

It checks that your local base.py contains all the Phase D-2 + D-4 fixes:
  - Abort check at start of each iteration
  - asyncio race pattern (cancel LLM call on panic button)
  - scan_registry + asyncio imports
  - All new event emitters (iteration, thinking, tool_call_started/completed)

Exit code 0 = OK, all markers found.
Exit code 1 = some markers missing — your base.py is OUTDATED.
"""
import sys
import hashlib
from pathlib import Path

# Default path (VAPT-AI standard layout)
DEFAULT_PATH = Path.home() / "VAPT-AI" / "app" / "agents" / "base.py"

# Markers that MUST be in the latest base.py
MARKERS = [
    # Phase D-1: new event emitters
    ("emit_tool_call_started", "Phase D-1: tool_call_started emitter"),
    ("emit_tool_call_completed", "Phase D-1: tool_call_completed emitter"),
    ("emit_assistant_message", "Phase D-1: assistant_message emitter"),
    ("emit_thinking", "Phase D-1: thinking emitter"),
    ("emit_iteration", "Phase D-1: iteration emitter"),
    # Phase D-2: abort check
    ("Phase D: abort check", "Phase D-2: abort check at start of iteration"),
    ("is_aborted(self.scan_id)", "Phase D-2: actual is_aborted call"),
    ("user_panic_button", "Phase D-2: panic button error code"),
    # Phase D-4: asyncio race
    ("race LLM call against abort_event", "Phase D-4: asyncio race pattern"),
    ("asyncio.create_task(abort_event.wait())", "Phase D-4: abort_event task"),
    ("return_when=asyncio.FIRST_COMPLETED", "Phase D-4: asyncio.wait FIRST_COMPLETED"),
    # Imports
    ("from app.pentest.scan_registry import scan_registry", "scan_registry import"),
    ("^import asyncio", "asyncio import (module level)"),
]

def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not target.exists():
        print(f"❌ File not found: {target}")
        print(f"   Usage: python {sys.argv[0]} /path/to/base.py")
        return 1

    content = target.read_text()
    lines = content.splitlines()
    size_kb = target.stat().st_size / 1024
    md5 = hashlib.md5(content.encode()).hexdigest()

    print(f"=== {target} ===")
    print(f"Size:   {len(lines)} lines, {size_kb:.1f} KB")
    print(f"MD5:    {md5}")
    print()

    missing = []
    found = []
    for marker, desc in MARKERS:
        if marker.startswith("^"):
            # Match start-of-line
            needle = marker[1:]
            present = any(line.startswith(needle) for line in lines)
        else:
            present = marker in content
        if present:
            found.append((marker, desc))
            print(f"  ✅ {desc}")
        else:
            missing.append((marker, desc))
            print(f"  ❌ MISSING: {desc}")
            print(f"     Looked for: {marker!r}")

    print()
    print(f"=== Summary: {len(found)}/{len(MARKERS)} markers found ===")
    if missing:
        print()
        print("⚠️  Your local base.py is OUTDATED — missing Phase D fixes.")
        print("    Download the latest from: /home/z/my-project/download/vapt-ai-patches/base.py")
        print("    Then:")
        print("      cp ~/Downloads/base.py ~/VAPT-AI/app/agents/")
        print("      sudo systemctl restart vapt-ai-app")
        return 1
    else:
        print()
        print("✅ Your local base.py is up to date with all Phase D fixes.")
        print("   Abort button + asyncio race pattern are present.")
        return 0

if __name__ == "__main__":
    sys.exit(main())