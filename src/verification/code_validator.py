"""Lightweight syntactic validation for code-type claims.

Tutorial videos teach code; if the vision pass mis-OCR's a snippet we don't
want to publish broken code. After claim extraction we run each code claim
through `ast.parse` (Python) or a regex-based balance check (other langs)
and tag the claim's `extra` field with the result so the draft prompt can
prefer parseable claims and the audit report can flag the rest.

Non-Python languages: we only check brace/paren/bracket balance, since
running a full parser per language is out of scope. This still catches
truncated snippets — the most common vision failure.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass
class CodeCheckResult:
    parseable: bool
    language: str
    reason: str = ""


_LANG_HINTS = {
    "py": "python", "python": "python",
    "c": "c", "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
    "js": "javascript", "javascript": "javascript", "ts": "typescript",
    "rs": "rust", "go": "go", "java": "java",
}

_LANG_RE = re.compile(r"```?\s*([a-zA-Z+]+)")


def _detect_language(content: str, hint: str | None) -> str:
    if hint:
        norm = hint.strip().lower()
        return _LANG_HINTS.get(norm, norm)
    m = _LANG_RE.match(content.lstrip())
    if m:
        return _LANG_HINTS.get(m.group(1).lower(), m.group(1).lower())
    # Heuristic: presence of `def `, `import `, indentation suggests Python
    if re.search(r"^\s*(def |class |import |from )", content, re.MULTILINE):
        return "python"
    return "unknown"


def _strip_fence(content: str) -> str:
    """Remove leading/trailing triple-backtick fences if present."""
    s = content.strip()
    if s.startswith("```"):
        # Drop opening fence line
        s = re.sub(r"^```[a-zA-Z+]*\s*\n?", "", s, count=1)
    if s.endswith("```"):
        s = s[: -3].rstrip()
    return s


def check_python(code: str) -> CodeCheckResult:
    try:
        ast.parse(code)
        return CodeCheckResult(parseable=True, language="python")
    except SyntaxError as e:
        return CodeCheckResult(
            parseable=False,
            language="python",
            reason=f"SyntaxError line {e.lineno}: {e.msg}",
        )


_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = set(_BRACKET_PAIRS.values())


def check_brackets(code: str) -> CodeCheckResult:
    """Generic brace/bracket/paren balance check for non-Python code."""
    stack: list[str] = []
    in_string: str | None = None
    i = 0
    while i < len(code):
        ch = code[i]
        if in_string:
            if ch == "\\" and i + 1 < len(code):
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'", "`"):
            in_string = ch
        elif ch in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[ch])
        elif ch in _CLOSERS:
            if not stack or stack[-1] != ch:
                return CodeCheckResult(
                    parseable=False,
                    language="generic",
                    reason=f"Unmatched bracket {ch!r} at offset {i}",
                )
            stack.pop()
        i += 1
    if stack:
        return CodeCheckResult(
            parseable=False, language="generic", reason=f"Unclosed: {stack}"
        )
    return CodeCheckResult(parseable=True, language="generic")


def check_code(content: str, language_hint: str | None = None) -> CodeCheckResult:
    """Validate a code claim. Returns a CodeCheckResult."""
    if not content or not content.strip():
        return CodeCheckResult(parseable=False, language="unknown", reason="empty")
    code = _strip_fence(content)
    lang = _detect_language(code, language_hint)
    if lang == "python":
        return check_python(code)
    return check_brackets(code)
