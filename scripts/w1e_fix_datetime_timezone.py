#!/usr/bin/env python3
"""Fix all datetime columns to use DateTime(timezone=True)."""
import re
from pathlib import Path

MODELS_DIR = Path("/home/nhat/VAPT-AI/app/db/models")

# Pattern: Mapped[datetime...] = mapped_column(  (not followed by DateTime)
# We need to add DateTime(timezone=True) as the first arg

def fix_file(filepath: Path) -> int:
    """Fix datetime columns in a file. Returns number of fixes applied."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    fixes = 0

    # Pattern 1: single-line mapped_column for datetime without DateTime
    # e.g. `expires_at: Mapped[datetime | None] = mapped_column(nullable=True)`
    # → `expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)`
    pattern1 = re.compile(
        r'(Mapped\[datetime[^\]]*\]\s*=\s*mapped_column\()\s*'
        r'(?!DateTime)'
        r'(nullable=[^)]+)\)',
        re.MULTILINE
    )
    def replace1(m):
        nonlocal fixes
        fixes += 1
        return f"{m.group(1)}DateTime(timezone=True), {m.group(2)})"
    content = pattern1.sub(replace1, content)

    # Pattern 2: multi-line mapped_column for datetime
    # e.g. `created_at: Mapped[datetime] = mapped_column(\n    nullable=False, server_default=func.now(), index=True\n)`
    pattern2 = re.compile(
        r'(Mapped\[datetime[^\]]*\]\s*=\s*mapped_column\()\s*\n\s*'
        r'(?!DateTime)'
        r'(nullable=[^)\n]+)',
        re.MULTILINE
    )
    def replace2(m):
        nonlocal fixes
        fixes += 1
        return f"{m.group(1)}DateTime(timezone=True),\n        {m.group(2)}"
    content = pattern2.sub(replace2, content)

    # Pattern 3: mapped_column(nullable=False) for datetime (hitl expires_at, audit expires_at)
    # already covered by pattern 1

    # Pattern 4: mapped_column(nullable=False, index=True) for datetime (audit expires_at)
    # already covered by pattern 1

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        print(f"  ✓ {filepath.name}: {fixes} fix(es) applied")
    else:
        print(f"  - {filepath.name}: no changes needed")
    return fixes

total = 0
for f in sorted(MODELS_DIR.glob("*.py")):
    if f.name == "__init__.py":
        continue
    total += fix_file(f)
print(f"\nTotal fixes: {total}")