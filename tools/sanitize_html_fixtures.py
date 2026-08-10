"""Strip third-party scripts from the cached Google Patents HTML fixtures.

Every patents.google.com page embeds Google's own public API key for the Help
widget, inside an inline `<script>`:

    helpApi = window.help.service.Lazy.create(0, {apiKey: 'AIza...'})

That is Google's key, not ours, and it is public on every patent page — nothing
of ours is exposed by it. But committing a verbatim copy of the page trips
secret scanning and ships ~700 KB of third-party JavaScript we never use.

The fixtures exist so the structured-source tests can run offline (PRD R7.2 /
AC-1.4). Those tests read the description section, the claims and the tables —
never a script. So the scripts come out and the assertions still hold.

Run:  .venv/bin/python tools/sanitize_html_fixtures.py
"""

from __future__ import annotations

import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "source"

SCRIPT = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
STYLE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
NOSCRIPT = re.compile(r"<noscript\b[^>]*>.*?</noscript\s*>", re.IGNORECASE | re.DOTALL)
# Belt and braces: any Google API key shaped token that survives the above.
API_KEY = re.compile(r"AIzaSy[A-Za-z0-9_\-]{20,}")


def sanitize(text: str) -> str:
    for pattern in (SCRIPT, STYLE, NOSCRIPT):
        text = pattern.sub("", text)
    return API_KEY.sub("REDACTED-THIRD-PARTY-KEY", text)


def main() -> None:
    for path in sorted(FIXTURES.glob("*.html")):
        original = path.read_text("utf-8", errors="replace")
        cleaned = sanitize(original)
        path.write_text(cleaned, encoding="utf-8")
        print(
            f"{path.name}: {len(original):,} -> {len(cleaned):,} chars "
            f"({100 * (1 - len(cleaned) / len(original)):.0f}% smaller), "
            f"keys remaining: {len(API_KEY.findall(cleaned))}"
        )


if __name__ == "__main__":
    main()
