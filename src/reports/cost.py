"""Generate reports/cost_report.html from the budget tracker's JSONL log."""
from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _load_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _format_table(rows: list[tuple], headers: list[str]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _build_html(entries: list[dict]) -> str:
    total = sum(e["usd"] for e in entries)
    by_stage: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    tokens_in = sum(e["input_tokens"] for e in entries)
    tokens_out = sum(e["output_tokens"] for e in entries)
    cache_read = sum(e["cache_read_tokens"] for e in entries)
    cache_write = sum(e["cache_write_tokens"] for e in entries)
    for e in entries:
        by_stage[e["stage"]] += e["usd"]
        by_model[e["model"]] += e["usd"]

    stage_rows = sorted(by_stage.items(), key=lambda x: -x[1])
    model_rows = sorted(by_model.items(), key=lambda x: -x[1])
    stage_table = _format_table(
        [(s, f"${v:.4f}") for s, v in stage_rows], ["Stage", "USD"]
    )
    model_table = _format_table(
        [(m, f"${v:.4f}") for m, v in model_rows], ["Model", "USD"]
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cost Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.6rem; margin-bottom: 0; }}
.gen {{ color: #888; font-size: 0.85rem; }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0; }}
.kpi {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; }}
.kpi .v {{ font-size: 1.5rem; font-weight: 600; }}
.kpi .l {{ color: #666; font-size: 0.85rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 2rem; }}
th, td {{ border-bottom: 1px solid #eee; padding: 6px 10px; text-align: left; }}
th {{ background: #fafafa; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
</style></head>
<body>
<h1>Cost Report</h1>
<p class="gen">Generated {datetime.now().isoformat(timespec='seconds')}</p>

<div class="kpis">
  <div class="kpi"><div class="v">${total:.2f}</div><div class="l">Total spend</div></div>
  <div class="kpi"><div class="v">{tokens_in/1e6:.2f}M</div><div class="l">Input tokens</div></div>
  <div class="kpi"><div class="v">{tokens_out/1e6:.2f}M</div><div class="l">Output tokens</div></div>
  <div class="kpi"><div class="v">{cache_read/1e6:.2f}M</div><div class="l">Cache read tokens</div></div>
  <div class="kpi"><div class="v">{cache_write/1e6:.2f}M</div><div class="l">Cache write tokens</div></div>
  <div class="kpi"><div class="v">{len(entries)}</div><div class="l">API calls</div></div>
</div>

<h2>By stage</h2>
{stage_table}

<h2>By model</h2>
{model_table}

</body></html>
"""


def run(cfg: Config) -> None:
    log_path = cfg.path("reports") / "cost_log.jsonl"
    entries = _load_entries(log_path)
    out = cfg.path("reports") / "cost_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_html(entries))
    log.info("Wrote %s (entries=%d, total=$%.2f)", out, len(entries),
             sum(e["usd"] for e in entries))
