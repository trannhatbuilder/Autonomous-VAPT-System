#!/usr/bin/env python3
"""
Apply Tailwind dark-mode class fixes across all frontend view files.

Problem: many views use hardcoded dark-mode classes like `text-zinc-50`,
`bg-zinc-900/60`, `border-zinc-800` WITHOUT the `dark:` prefix. When the
user switches to light mode, these classes still apply → invisible text
(white-on-white) or dark card on white background.

Fix: replace dark-only patterns with light+dark pairs that work in BOTH
themes. Replacements are conservative — only touch clearly-broken
dark-only patterns, never touch existing `dark:` prefixes.

Idempotent: re-running produces the same output.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Files to patch (relative to frontend/src)
TARGET_FILES = [
    "components/views/findings-view.tsx",
    "components/views/dashboard-view.tsx",
    "components/views/scans-view.tsx",
    "components/views/settings-view.tsx",
    "components/views/report-history-view.tsx",
    "components/views/hitl-approval-modal.tsx",
    "components/auth/login-form.tsx",
    "app/page.tsx",
]

# Map of dark-only class → light+dark pair.
CLASS_MAP: list[tuple[str, str]] = [
    # Text colors
    ("text-zinc-50",  "text-zinc-900 dark:text-zinc-50"),
    ("text-zinc-100", "text-zinc-900 dark:text-zinc-100"),
    ("text-zinc-200", "text-zinc-800 dark:text-zinc-200"),
    ("text-zinc-300", "text-zinc-700 dark:text-zinc-300"),
    ("text-zinc-400", "text-zinc-600 dark:text-zinc-400"),
    # Backgrounds (zinc)
    ("bg-zinc-900/60", "bg-white dark:bg-zinc-900/60"),
    ("bg-zinc-900/40", "bg-white dark:bg-zinc-900/40"),
    ("bg-zinc-900/80", "bg-white/80 dark:bg-zinc-900/80"),
    ("bg-zinc-900",    "bg-white dark:bg-zinc-900"),
    ("bg-zinc-950",    "bg-zinc-50 dark:bg-zinc-950"),
    ("bg-zinc-800",    "bg-zinc-100 dark:bg-zinc-800"),
    ("bg-zinc-800/60", "bg-zinc-100/60 dark:bg-zinc-800/60"),
    # Borders (zinc)
    ("border-zinc-800", "border-zinc-200 dark:border-zinc-800"),
    ("border-zinc-700", "border-zinc-300 dark:border-zinc-700"),
    ("border-zinc-900/50", "border-zinc-200 dark:border-zinc-900/50"),
    ("border-zinc-900", "border-zinc-200 dark:border-zinc-900"),
    # Placeholders
    ("placeholder-zinc-600", "placeholder-zinc-400 dark:placeholder-zinc-600"),
    # Hover variants
    ("hover:bg-zinc-800", "hover:bg-zinc-100 dark:hover:bg-zinc-800"),

    # ── Dark-only red surfaces (various opacities) ──
    ("bg-red-950/40", "bg-red-50 dark:bg-red-950/40"),
    ("bg-red-950/30", "bg-red-50 dark:bg-red-950/30"),
    ("bg-red-950/20", "bg-red-50 dark:bg-red-950/20"),
    ("bg-red-950", "bg-red-50 dark:bg-red-950"),
    ("border-red-900/50", "border-red-200 dark:border-red-900/50"),
    ("border-red-900", "border-red-300 dark:border-red-900"),

    # ── Dark-only emerald surfaces (various opacities) ──
    ("bg-emerald-950/30", "bg-emerald-50 dark:bg-emerald-950/30"),
    ("bg-emerald-950/20", "bg-emerald-50 dark:bg-emerald-950/20"),
    ("bg-emerald-950", "bg-emerald-50 dark:bg-emerald-950"),
    ("border-emerald-900/50", "border-emerald-200 dark:border-emerald-900/50"),
    ("border-emerald-900", "border-emerald-300 dark:border-emerald-900"),

    # ── Dark-only amber/orange surfaces (various opacities) ──
    ("bg-amber-950/30", "bg-amber-50 dark:bg-amber-950/30"),
    ("bg-amber-950/20", "bg-amber-50 dark:bg-amber-950/20"),
    ("bg-amber-950", "bg-amber-50 dark:bg-amber-950"),
    ("border-amber-900/50", "border-amber-200 dark:border-amber-900/50"),
    ("border-amber-900", "border-amber-300 dark:border-amber-900"),

    # ── Dark-only blue surfaces (various opacities) ──
    ("bg-blue-950/20", "bg-blue-50 dark:bg-blue-950/20"),
    ("bg-blue-950/30", "bg-blue-50 dark:bg-blue-950/30"),
    ("bg-blue-950", "bg-blue-50 dark:bg-blue-950"),
    ("border-blue-900/50", "border-blue-200 dark:border-blue-900/50"),
    ("border-blue-900", "border-blue-300 dark:border-blue-900"),

    # ── Dark-only purple surfaces ──
    ("bg-purple-950/30", "bg-purple-50 dark:bg-purple-950/30"),
    ("bg-purple-950/20", "bg-purple-50 dark:bg-purple-950/20"),
    ("bg-purple-950", "bg-purple-50 dark:bg-purple-950"),
    ("border-purple-900/50", "border-purple-200 dark:border-purple-900/50"),
    ("border-purple-900", "border-purple-300 dark:border-purple-900"),

    # ── Light text colors used on dark backgrounds (need light pair) ──
    # These appear in scan timeline result blocks → invisible on light mode
    ("text-amber-200",  "text-amber-700 dark:text-amber-200"),
    ("text-amber-300",  "text-amber-700 dark:text-amber-300"),
    ("text-emerald-200", "text-emerald-700 dark:text-emerald-200"),
    ("text-emerald-300", "text-emerald-700 dark:text-emerald-300"),
    ("text-red-200",  "text-red-700 dark:text-red-200"),
    ("text-red-300",  "text-red-700 dark:text-red-300"),
    ("text-blue-200", "text-blue-700 dark:text-blue-200"),
    ("text-blue-300", "text-blue-700 dark:text-blue-300"),
    ("text-purple-200", "text-purple-700 dark:text-purple-200"),
    ("text-purple-300", "text-purple-700 dark:text-purple-300"),

    # ── Pale accent colors that are unreadable on white ──
    ("text-amber-400", "text-amber-500 dark:text-amber-400"),
    ("text-blue-400", "text-blue-500 dark:text-blue-400"),
    ("hover:text-blue-300", "hover:text-blue-600 dark:hover:text-blue-300"),
    ("text-emerald-400", "text-emerald-500 dark:text-emerald-400"),
    ("text-red-400",  "text-red-500 dark:text-red-400"),
    ("text-purple-400", "text-purple-500 dark:text-purple-400"),
]


def patch_file(path: Path) -> int:
    """Apply all replacement rules to one file. Returns # of replacements."""
    src = path.read_text(encoding="utf-8")
    original = src

    for dark_only, light_dark_pair in CLASS_MAP:
        # Match the dark-only token, but NOT when preceded by `:` (which
        # would mean it's already part of a `dark:` chain).
        pattern = re.compile(
            r"(?<![A-Za-z0-9:_/])"
            + re.escape(dark_only)
            + r"(?![A-Za-z0-9:_/.\-])"
        )
        src = pattern.sub(light_dark_pair, src)

    if src != original:
        path.write_text(src, encoding="utf-8")
        diff = 0
        for dark_only, _ in CLASS_MAP:
            diff += original.count(dark_only) - src.count(dark_only)
        return diff
    return 0


def main() -> int:
    # Repo root is the parent of this script's `scripts/` directory.
    repo_root = Path(__file__).resolve().parent.parent
    frontend_src = repo_root / "frontend" / "src"
    if not frontend_src.is_dir():
        print(f"ERROR: frontend src dir not found at {frontend_src}", file=sys.stderr)
        return 2

    total = 0
    for rel in TARGET_FILES:
        path = frontend_src / rel
        if not path.is_file():
            print(f"SKIP (missing): {rel}")
            continue
        n = patch_file(path)
        print(f"{'PATCHED' if n > 0 else 'unchanged':>9}: {rel}  ({n} replacements)")
        total += n

    print(f"\nTotal replacements: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())