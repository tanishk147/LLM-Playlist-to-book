"""Generate reports/grounding_audit.html from the citations table."""
from __future__ import annotations

import html
from collections import defaultdict
from datetime import datetime

from ..grounding import ClaimStore
from ..utils.cache import read_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


_STATUS_COLORS = {
    "grounded": "#2ca02c",
    "narrative_ok": "#7f7f7f",
    "needs_external_check": "#ff7f0e",
    "external_ok": "#1f77b4",
    "unsupported": "#d62728",
    "contradicted": "#8b0000",
}


def _status_bar(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    segments = []
    legend = []
    for status, n in counts.items():
        pct = 100.0 * n / total
        color = _STATUS_COLORS.get(status, "#bbb")
        segments.append(
            f'<div style="flex:{pct}; background:{color};" title="{status}: {n}"></div>'
        )
        legend.append(
            f'<span><i style="background:{color};"></i> '
            f'{html.escape(status)} ({n})</span>'
        )
    bar = (
        '<div style="display:flex; height:24px; border-radius:4px; overflow:hidden;">'
        + "".join(segments)
        + "</div>"
    )
    return bar + '<div class="legend">' + " ".join(legend) + "</div>"


def _format_chapter_section(chap_id: str, title: str, rows: list[dict]) -> str:
    # Aggregate by sentence_index (one citation row per claim referenced)
    by_sent: dict[int, dict] = {}
    for r in rows:
        idx = r["sentence_index"]
        if idx not in by_sent:
            by_sent[idx] = {
                "index": idx,
                "text": r["sentence_text"],
                "type": r["sentence_type"],
                "status": r["status"],
                "reason": r.get("reason", ""),
                "claim_ids": [],
            }
        if r["claim_id"] is not None:
            by_sent[idx]["claim_ids"].append(r["claim_id"])

    counts: dict[str, int] = defaultdict(int)
    for s in by_sent.values():
        counts[s["status"]] += 1

    rows_html = []
    for s in sorted(by_sent.values(), key=lambda x: x["index"]):
        color = _STATUS_COLORS.get(s["status"], "#bbb")
        cids = ", ".join(f"c{c}" for c in s["claim_ids"]) or "-"
        rows_html.append(
            f'<tr>'
            f'<td>{s["index"]}</td>'
            f'<td><span class="pill" style="background:{color};">{s["status"]}</span></td>'
            f'<td>{html.escape(s["type"])}</td>'
            f'<td>{cids}</td>'
            f'<td>{html.escape(s["text"][:240])}</td>'
            f'<td>{html.escape(s.get("reason", ""))}</td>'
            f'</tr>'
        )

    return (
        f'<section><h3>{html.escape(title)} <code>({chap_id})</code></h3>'
        + _status_bar(counts)
        + '<table><thead><tr><th>#</th><th>Status</th><th>Type</th>'
        '<th>Cited</th><th>Sentence</th><th>Reason</th></tr></thead>'
        '<tbody>' + "".join(rows_html) + '</tbody></table></section>'
    )


def _build_html(cfg: Config, store: ClaimStore) -> str:
    outline = read_json(cfg.path("outline"))
    all_chapters = (outline.get("chapters") or []) + (outline.get("appendices") or [])

    overall = store.audit_summary()
    overall_bar = _status_bar(overall) if overall else "<p>No citations recorded yet.</p>"

    sections = []
    for ch in all_chapters:
        rows = store.audit_for_chapter(ch["id"])
        if not rows:
            continue
        sections.append(_format_chapter_section(ch["id"], ch["title"], rows))

    total_sentences = sum(overall.values()) or 1
    grounded_pct = 100.0 * overall.get("grounded", 0) / total_sentences
    unsupported_pct = 100.0 * overall.get("unsupported", 0) / total_sentences

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Grounding Audit</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1080px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ margin-bottom: 0; }}
.gen {{ color: #888; font-size: 0.85rem; }}
.legend {{ margin: 6px 0 14px; font-size: 0.85rem; color: #333; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
.legend span {{ margin-right: 1rem; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }}
.kpi {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; }}
.kpi .v {{ font-size: 1.5rem; font-weight: 600; }}
.kpi .l {{ color: #666; font-size: 0.85rem; }}
section {{ margin: 2rem 0; padding-top: 1rem; border-top: 1px solid #eee; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 0.5rem; }}
th, td {{ border-bottom: 1px solid #eee; padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: #fafafa; }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 10px; color: white; font-size: 0.75rem; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 0.85em; }}
</style></head>
<body>
<h1>Grounding Audit</h1>
<p class="gen">Generated {datetime.now().isoformat(timespec='seconds')}</p>

<div class="kpis">
  <div class="kpi"><div class="v">{total_sentences}</div><div class="l">Sentences audited</div></div>
  <div class="kpi"><div class="v">{grounded_pct:.1f}%</div><div class="l">Grounded</div></div>
  <div class="kpi"><div class="v">{unsupported_pct:.1f}%</div><div class="l">Unsupported (stripped)</div></div>
  <div class="kpi"><div class="v">{len(all_chapters)}</div><div class="l">Chapters + appendices</div></div>
</div>

<h2>Overall</h2>
{overall_bar}

{''.join(sections)}

</body></html>
"""


def run(cfg: Config) -> None:
    store = ClaimStore(cfg.path("claims_db"))
    out = cfg.path("reports") / "grounding_audit.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_html(cfg, store))
    log.info("Wrote %s", out)
