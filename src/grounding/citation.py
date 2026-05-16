"""Sentence splitting + citation tag extraction for chapter drafts.

The draft prompt instructs the model to use inline citations of the form
[c47] or [c12, c47]. This module parses those out and also classifies
sentence types (code/equation/figure/narrative) without depending on the
verification LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

SentenceType = Literal["factual", "narrative", "code_block", "equation", "figure_directive"]

# Citation: [c123] or [c12, c47]
_CIT_RE = re.compile(r"\[c(\d+(?:\s*,\s*c?\d+)*)\]", re.IGNORECASE)
# GAP marker
_GAP_RE = re.compile(r"\[GAP:[^\]]+\]")
# FIGURE marker
_FIG_RE = re.compile(r"\[FIGURE:[^\]]+\]")
# Block math: $$...$$
_BLOCK_MATH_RE = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
# Code fence
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


@dataclass
class Sentence:
    index: int
    text: str
    type: SentenceType
    cited_claim_ids: list[int] = field(default_factory=list)


def extract_citations(text: str) -> list[int]:
    """Return list of claim IDs referenced in inline `[cNNN]` tags."""
    out: list[int] = []
    for m in _CIT_RE.finditer(text):
        for piece in m.group(1).split(","):
            piece = piece.strip().lstrip("cC")
            try:
                out.append(int(piece))
            except ValueError:
                continue
    return out


def _split_off_code_blocks(md: str) -> list[tuple[str, str]]:
    """Yield (kind, content) tuples where kind ∈ {'text', 'code', 'block_math'}."""
    out: list[tuple[str, str]] = []
    lines = md.split("\n")
    i = 0
    in_code = False
    buf_text: list[str] = []
    buf_code: list[str] = []

    def flush_text() -> None:
        if buf_text:
            out.append(("text", "\n".join(buf_text)))
            buf_text.clear()

    def flush_code() -> None:
        if buf_code:
            out.append(("code", "\n".join(buf_code)))
            buf_code.clear()

    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            if not in_code:
                flush_text()
                in_code = True
                buf_code.append(line)
            else:
                buf_code.append(line)
                flush_code()
                in_code = False
            i += 1
            continue
        if in_code:
            buf_code.append(line)
        else:
            buf_text.append(line)
        i += 1
    flush_text()
    flush_code()

    # Now split text segments to extract block math
    refined: list[tuple[str, str]] = []
    for kind, content in out:
        if kind != "text":
            refined.append((kind, content))
            continue
        pos = 0
        for m in _BLOCK_MATH_RE.finditer(content):
            if m.start() > pos:
                refined.append(("text", content[pos:m.start()]))
            refined.append(("block_math", m.group(0)))
            pos = m.end()
        if pos < len(content):
            refined.append(("text", content[pos:]))
    return refined


# Sentence splitter for plain text segments.
# Conservative: splits on sentence-ending punctuation followed by space/newline + capital.
# Keeps `[FIGURE: ...]` and `[GAP: ...]` markers attached to whichever side they sit on.
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\[`])"
)


def _split_sentences_in_text(text: str, start_index: int) -> list[Sentence]:
    sentences: list[Sentence] = []
    # Split paragraph-by-paragraph to respect markdown breaks
    paras = re.split(r"\n\s*\n", text)
    idx = start_index
    for para in paras:
        para = para.strip()
        if not para:
            continue
        # If the whole paragraph is just a heading, keep as narrative
        if para.startswith("#"):
            sentences.append(Sentence(index=idx, text=para, type="narrative"))
            idx += 1
            continue
        # If paragraph contains a FIGURE directive, emit as figure_directive
        if _FIG_RE.search(para) and len(_FIG_RE.findall(para)) == 1 and len(para) < 200:
            sentences.append(Sentence(index=idx, text=para, type="figure_directive"))
            idx += 1
            continue
        # Split into sentences
        parts = _SENT_SPLIT_RE.split(para)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            cids = extract_citations(p)
            stype: SentenceType = "factual" if cids or _has_factual_signal(p) else "narrative"
            if _GAP_RE.search(p) and not cids:
                stype = "factual"
            sentences.append(Sentence(index=idx, text=p, type=stype, cited_claim_ids=cids))
            idx += 1
    return sentences


_NUMERIC_OR_TECH_RE = re.compile(r"\b(\d+(\.\d+)?|[A-Z]{2,})\b")


def _has_factual_signal(text: str) -> bool:
    """Heuristic for whether a sentence makes a factual assertion.

    Used only when no citation is present. Anything matching this *and* lacking
    a citation will be flagged as unsupported by the verifier; this is a
    pre-filter to label it sensibly in the dataclass.
    """
    if _NUMERIC_OR_TECH_RE.search(text):
        return True
    factual_verbs = (
        " is ", " are ", " was ", " were ", " uses ", " has ", " have ", " contains ",
        " requires ", " produces ", " equals ", " computes ", " applies ", " defines ",
    )
    lower = " " + text.lower() + " "
    return any(v in lower for v in factual_verbs)


def split_sentences(chapter_md: str) -> list[Sentence]:
    """Split a chapter draft into typed sentences."""
    sentences: list[Sentence] = []
    idx = 0
    for kind, content in _split_off_code_blocks(chapter_md):
        if kind == "code":
            # one sentence representing the whole code block
            first = next((ln for ln in content.split("\n")[1:] if ln.strip()), content[:80])
            sentences.append(Sentence(index=idx, text=first.strip(), type="code_block"))
            idx += 1
        elif kind == "block_math":
            sentences.append(Sentence(index=idx, text=content.strip(), type="equation"))
            idx += 1
        else:
            new = _split_sentences_in_text(content, idx)
            sentences.extend(new)
            idx += len(new)
    return sentences


def parse_chapter_for_audit(chapter_md: str) -> list[dict]:
    """Return per-sentence dicts in the verify-prompt input shape.

    Useful as a fallback when the verify LLM output is malformed; the orchestrator
    can fall back to a 'no-citation = unsupported' rule.
    """
    out = []
    for s in split_sentences(chapter_md):
        out.append(
            {
                "index": s.index,
                "text": s.text,
                "type": s.type,
                "cited_claim_ids": [f"c{i}" for i in s.cited_claim_ids],
            }
        )
    return out


class CitationParser:
    """Convenience class wrapping the splitter for stage code."""

    def parse(self, md: str) -> list[Sentence]:
        return split_sentences(md)

    @staticmethod
    def strip_unsupported(md: str, unsupported_indices: set[int]) -> str:
        """Remove sentences whose indices are flagged unsupported. Conservative:
        only operates on text paragraphs (not code/math blocks).

        Strategy: ordered walk. For each sentence in document order, if its
        index is unsupported, remove the *first remaining* occurrence of that
        sentence text starting from the current scan offset. This avoids
        re-removing identical sentences that appear later in the document and
        survives repeated phrasing.
        """
        sentences = split_sentences(md)
        if not sentences:
            return md
        bad = sorted(
            [s for s in sentences if s.index in unsupported_indices],
            key=lambda s: s.index,
        )
        if not bad:
            return md

        out = md
        scan_from = 0
        for s in bad:
            t = s.text.strip()
            if not t:
                continue
            # Find the next occurrence at or after scan_from
            idx = out.find(t, scan_from)
            if idx < 0:
                # Sentence not located verbatim (e.g., minor whitespace diffs);
                # fall back to a tolerant regex but bounded to the tail.
                pattern = re.compile(re.escape(t).replace(r"\ ", r"\s+"))
                m = pattern.search(out, scan_from)
                if not m:
                    continue
                idx, end = m.start(), m.end()
            else:
                end = idx + len(t)
            out = out[:idx] + out[end:]
            scan_from = idx  # next bad sentence appears after this hole
        return out
