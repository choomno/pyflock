"""Check repository-local Markdown links and public documentation files."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "docs/testing.md",
    "docs/reproduction.md",
    "docs/mathematical-reference/main.tex",
    "docs/mathematical-reference/mathematical-reference.pdf",
)


def check():
    errors = [
        f"Missing documentation: {path}"
        for path in REQUIRED
        if not (ROOT / path).is_file()
    ]
    for source in (
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ):
        if not source.is_file():
            continue
        content = source.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]+)\)", content):
            target = match.group(1).split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            target = target.strip("<>")
            resolved = (source.parent / target).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"{source.relative_to(ROOT)}: broken local link {target}")
    if errors:
        raise ValueError("\n".join(errors))
    print("Documentation links and required files pass")


if __name__ == "__main__":
    check()
