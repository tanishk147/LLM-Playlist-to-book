"""Stage 10 - figures.

Walks each polished chapter, finds `[FIGURE: ...]` directives, locates the
source diagram(s) by mining the claims/visuals that mention diagrams, and
regenerates the figure as Mermaid/TikZ where feasible, or embeds the original
frame otherwise.

Outputs:
  book/figures/generated/<chapter_id>_fig<NN>.{md,tex}  (regen)
  book/figures/embedded/<chapter_id>_fig<NN>.jpg        (embed fallback)

The figure markers in polished.md are replaced with proper figure references
in stage 11 (typeset).
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..grounding import ClaimStore
from ..llm import BudgetTracker, LLMClient
from ..llm.client import Block, Message
from ..llm.prompts import load_prompt
from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)

_FIG_DIRECTIVE_RE = re.compile(r"\[FIGURE:\s*([^\]]+)\]")


@dataclass
class FigureSpec:
    chapter_id: str
    index: int
    description: str
    candidate_frame_paths: list[Path] = field(default_factory=list)
    candidate_diagrams: list[dict] = field(default_factory=list)


def _find_directives(md: str) -> list[str]:
    return [m.group(1).strip() for m in _FIG_DIRECTIVE_RE.finditer(md)]


def _collect_diagram_visuals(cfg: Config, chapter_claims: list) -> list[dict]:
    """Pull vision JSONs for any frame referenced by claims of this chapter
    where the vision category is diagram, math, or slide."""
    out = []
    vision_root = cfg.path("vision")
    seen = set()
    for c in chapter_claims:
        for fid in c.frame_ids:
            if fid in seen:
                continue
            video_id = fid.split("_f")[0] if "_f" in fid else fid.rsplit("_", 1)[0]
            vpath = vision_root / video_id / f"{fid}.json"
            if not vpath.exists():
                continue
            data = json.loads(vpath.read_text())
            if data.get("category") in ("diagram", "math", "slide"):
                out.append(data)
                seen.add(fid)
    return out


def _frame_path_for(cfg: Config, frame_id: str) -> Path | None:
    """Reconstruct the path to the original frame JPEG."""
    parts = frame_id.split("_f")
    if len(parts) != 2:
        return None
    video_id = parts[0]
    p = cfg.path("frames") / video_id / f"{frame_id}.jpg"
    return p if p.exists() else None


def _process_figure(
    client: LLMClient,
    cfg: Config,
    chapter_id: str,
    fig_idx: int,
    description: str,
    diagrams: list[dict],
) -> dict:
    """Call the figure-regen LLM. Returns dict per figure prompt schema."""
    sys_prompt = load_prompt("figure_regenerate")
    blocks: list[Block] = [
        Block(type="text", text=f"## FIGURE directive\n{description}\n"),
    ]
    if diagrams:
        blocks.append(
            Block(
                type="text",
                text="## Diagram extractions\n" + json.dumps(diagrams[:3], indent=2),
            )
        )
        # Attach up to 2 source images for visual reference
        for d in diagrams[:2]:
            fid = d.get("frame_id")
            if not fid:
                continue
            p = _frame_path_for(cfg, fid)
            if p:
                blocks.append(
                    Block(type="image", image_path=p, image_media_type="image/jpeg")
                )
                blocks.append(Block(type="text", text=f"Above: source frame {fid}"))

    resp = client.call(
        stage="figures",
        model=cfg.model("figures"),
        system=sys_prompt,
        messages=[Message(role="user", blocks=blocks)],
        max_tokens=4096,
        temperature=0.0,
    )

    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*\n", "", raw)
        raw = re.sub(r"\n```\s*$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Bad JSON from figure regen for %s fig%d; defaulting to embed",
                    chapter_id, fig_idx)
        result = {"mode": "embed", "code": None, "caption": description, "rationale": "parse_error"}

    # Persist
    fig_base = f"{chapter_id}_fig{fig_idx:02d}"
    if result.get("mode") == "mermaid" and result.get("code"):
        out = cfg.path("figures_generated") / f"{fig_base}.mmd"
        out.write_text(result["code"])
        result["artifact_path"] = str(out.relative_to(cfg.root))
    elif result.get("mode") == "tikz" and result.get("code"):
        out = cfg.path("figures_generated") / f"{fig_base}.tex"
        out.write_text(result["code"])
        result["artifact_path"] = str(out.relative_to(cfg.root))
    else:
        # embed: copy the first source frame, if any
        if diagrams:
            fid = diagrams[0].get("frame_id")
            src = _frame_path_for(cfg, fid) if fid else None
            if src:
                dst = cfg.path("figures_embedded") / f"{fig_base}.jpg"
                shutil.copy2(src, dst)
                result["artifact_path"] = str(dst.relative_to(cfg.root))
            else:
                result["artifact_path"] = None
        else:
            result["artifact_path"] = None

    result["chapter_id"] = chapter_id
    result["index"] = fig_idx
    result["description"] = description
    return result


def run(cfg: Config) -> None:
    outline = read_json(cfg.path("outline"))
    store = ClaimStore(cfg.path("claims_db"))
    budget = BudgetTracker(
        log_path=cfg.path("reports") / "cost_log.jsonl",
        total_max_usd=cfg.budget("total_max"),
        per_chapter_max_usd=cfg.budget("per_chapter_max"),
        abort_on_exceed=cfg.pipeline["budget_usd"]["abort_on_exceed"],
    )
    client = LLMClient(cfg, budget)

    all_chapters = (outline.get("chapters") or []) + (outline.get("appendices") or [])
    all_figures: list[dict] = []
    for ch in all_chapters:
        chap_dir = cfg.path("chapters") / f"{ch['number']:02d}_{ch['id']}"
        polished = chap_dir / "polished.md"
        if not polished.exists():
            log.warning("No polished.md for chapter %s; skipping figures", ch["id"])
            continue
        directives = _find_directives(polished.read_text())
        if not directives:
            continue
        claims = store.claims_for_chapter(ch["id"])
        diagrams = _collect_diagram_visuals(cfg, claims)
        log.info("Chapter %s: %d figure directives, %d candidate diagrams",
                 ch["id"], len(directives), len(diagrams))
        for i, desc in enumerate(directives, start=1):
            spec = _process_figure(client, cfg, ch["id"], i, desc, diagrams)
            all_figures.append(spec)

    write_json(cfg.path("data") / "figures_index.json", all_figures)
    log.info("Figures generated/embedded: %d total", len(all_figures))
