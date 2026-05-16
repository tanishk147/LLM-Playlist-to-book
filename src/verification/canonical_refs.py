"""Canonical-reference confirmation pass.

Used only to *confirm* playlist-derived claims that the verifier marked
`needs_external_check`. Never introduces new claims.

The reference set is declared in config/pipeline.yaml `canonical_refs`.
For citation purposes we just need a stable bibkey and human-readable
metadata; the actual confirmation can be done by:
- a manual offline review (default behavior: queue for human),
- or arxiv abstract fetch (if you want some automation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import arxiv  # optional
except Exception:  # pragma: no cover
    arxiv = None  # type: ignore


@dataclass
class ExternalConfirmation:
    sentence_text: str
    chapter_id: str
    sentence_index: int
    candidate_refs: list[str] = field(default_factory=list)
    confirmed_ref: Optional[str] = None
    note: str = ""


class CanonicalRefIndex:
    """Indexes the canonical_refs list from pipeline.yaml."""

    def __init__(self, refs: list[dict]):
        self.refs = refs
        self._by_id = {r["id"]: r for r in refs}

    def bib_entries(self) -> str:
        """Return a BibTeX string for refs.bib."""
        lines = []
        for r in self.refs:
            kind = "article" if r.get("arxiv") else "book"
            lines.append(f"@{kind}{{{r['id']},")
            lines.append(f'  title = {{{r["title"]}}},')
            if r.get("author"):
                lines.append(f'  author = {{{r["author"]}}},')
            if r.get("arxiv"):
                lines.append(f'  journal = {{arXiv preprint arXiv:{r["arxiv"]}}},')
                lines.append(f'  eprint = {{{r["arxiv"]}}},')
            if r.get("year"):
                lines.append(f'  year = {{{r["year"]}}},')
            lines.append("}\n")
        return "\n".join(lines)

    def get(self, ref_id: str) -> dict | None:
        return self._by_id.get(ref_id)

    def fetch_arxiv_abstract(self, ref_id: str) -> Optional[str]:
        """Pull abstract from arxiv. Optional. Used to support confirmations."""
        ref = self._by_id.get(ref_id)
        if not ref or not ref.get("arxiv") or arxiv is None:
            return None
        try:
            search = arxiv.Search(id_list=[ref["arxiv"]])
            result = next(search.results())
            return result.summary
        except Exception:
            return None

    def queue_for_review(
        self,
        items: list[ExternalConfirmation],
        out_path: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump([item.__dict__ for item in items], f, indent=2)
