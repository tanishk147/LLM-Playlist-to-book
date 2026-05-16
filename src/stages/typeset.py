"""Stage 11 - typeset.

Concatenates polished chapter markdown into book/manuscript.md, replaces
[FIGURE: ...] directives with proper figure references using the figures
index, generates refs.bib from canonical_refs, and shells out to Pandoc +
XeLaTeX for the final PDF.

Mermaid figures are pre-rendered to PNG via mermaid-cli if available;
otherwise the .mmd code is embedded as a fenced code block (still rendered
in the PDF as inline code, which is functional if not pretty).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from ..utils.cache import read_json
from ..utils.config import Config
from ..utils.logging import get_logger
from ..verification.canonical_refs import CanonicalRefIndex

log = get_logger(__name__)

_FIG_DIRECTIVE_RE = re.compile(r"\[FIGURE:\s*([^\]]+)\]")

_LATEX_SPECIAL = [
    ("\\", "\\textbackslash{}"),
    ("&",  "\\&"),
    ("%",  "\\%"),
    ("$",  "\\$"),
    ("#",  "\\#"),
    ("_",  "\\_"),
    ("^",  "\\^{}"),
    ("~",  "\\textasciitilde{}"),
    ("{",  "\\{"),
    ("}",  "\\}"),
]


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text (e.g. captions)."""
    # Handle backslash first to avoid double-escaping
    result = text.replace("\\", "\\textbackslash{}")
    for char, replacement in _LATEX_SPECIAL[1:]:
        result = result.replace(char, replacement)
    return result


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _get_mmdc_path() -> str | None:
    if shutil.which("mmdc"):
        return "mmdc"
    if Path("/opt/homebrew/bin/mmdc").exists():
        return "/opt/homebrew/bin/mmdc"
    if Path("/usr/local/bin/mmdc").exists():
        return "/usr/local/bin/mmdc"
    return None

def _render_mermaid_to_png(mmd_path: Path, out_png: Path) -> bool:
    """Use mermaid-cli (mmdc) if available. Returns True on success."""
    mmdc_cmd = _get_mmdc_path()
    if not mmdc_cmd:
        return False
    try:
        env = os.environ.copy()
        env["PUPPETEER_EXECUTABLE_PATH"] = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        res = subprocess.run(
            [mmdc_cmd, "-i", str(mmd_path), "-o", str(out_png), "-b", "white"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if res.returncode != 0:
            log.warning("mmdc failed for %s:\n%s\n%s", mmd_path, res.stdout, res.stderr)
        return res.returncode == 0 and out_png.exists()
    except Exception as e:
        log.warning("mermaid-cli failed for %s: %s", mmd_path, e)
        return False


def _build_figure_replacement(
    fig: dict,
    cfg: Config,
    chap_id: str,
    fig_idx: int,
) -> str:
    """Return the markdown snippet that replaces [FIGURE: ...]."""
    mode = fig.get("mode", "embed")
    caption = fig.get("caption") or fig.get("description", "")
    artifact = fig.get("artifact_path")
    label = f"fig:{chap_id}_{fig_idx:02d}"

    if mode == "mermaid" and artifact:
        # Try to pre-render to PNG
        mmd_path = cfg.root / artifact
        png_path = mmd_path.with_suffix(".png")
        if _render_mermaid_to_png(mmd_path, png_path):
            rel = os.path.relpath(png_path, cfg.root)
            return f"\n![{caption}]({rel}){{#{label}}}\n"
        # Fallback: embed mermaid code block (won't render in PDF but preserves content)
        code = mmd_path.read_text()
        return f"\n```mermaid\n{code}\n```\n*Figure: {caption}*\n"

    if mode == "tikz" and artifact:
        # AI-generated TikZ often has syntax issues; show a safe placeholder instead
        # of crashing the entire PDF build with raw LaTeX errors.
        safe_caption = _latex_escape(caption)
        return (
            f"\n\\begin{{center}}"
            f"\\textit{{[Diagram: {safe_caption}]}}"
            f"\\end{{center}}\n"
        )

    if artifact:
        # embed: image
        return f"\n![{caption}]({artifact}){{#{label}}}\n"

    # Nothing available; leave a visible note so the gap is obvious in the PDF
    return f"\n*[Figure missing: {caption}]*\n"


def _replace_figures(md: str, chap_id: str, figures: list[dict], cfg: Config) -> str:
    """Walk markdown, replacing [FIGURE: ...] with built replacements in order."""
    chap_figs = [f for f in figures if f.get("chapter_id") == chap_id]
    chap_figs.sort(key=lambda f: f.get("index", 0))

    out_parts: list[str] = []
    last = 0
    fig_iter = iter(chap_figs)
    for m in _FIG_DIRECTIVE_RE.finditer(md):
        out_parts.append(md[last:m.start()])
        fig = next(fig_iter, None)
        if fig:
            out_parts.append(_build_figure_replacement(fig, cfg, chap_id, fig["index"]))
        else:
            out_parts.append(f"*[Figure missing: {m.group(1).strip()}]*")
        last = m.end()
    out_parts.append(md[last:])
    return "".join(out_parts)


def _build_manuscript_md(cfg: Config) -> Path:
    outline = read_json(cfg.path("outline"))
    figures_idx_path = cfg.path("data") / "figures_index.json"
    figures = read_json(figures_idx_path) if figures_idx_path.exists() else []

    parts: list[str] = []

    # Title metadata block (YAML for Pandoc)
    title = cfg.playlist["book"]["title"]
    subtitle = cfg.playlist["book"]["subtitle"]
    author = cfg.playlist["book"]["author"]
    date = datetime.now().strftime(cfg.playlist["book"].get("date_format", "%B %Y"))
    parts.append("---")
    parts.append(f'title: "{title}"')
    parts.append(f'subtitle: "{subtitle}"')
    parts.append(f'author: "{author}"')
    parts.append(f'date: "{date}"')
    parts.append('lang: en-US')
    parts.append('documentclass: book')
    # oneside = no alternating margins/page numbers. openany = no blank pages before chapters.
    parts.append('classoption: [11pt, oneside, openany]')
    parts.append('header-includes:')
    parts.append('  - \\pagestyle{plain}')
    parts.append('geometry: margin=1in')
    parts.append('linkcolor: blue')
    parts.append('toc: true')
    parts.append('toc-depth: 2')
    parts.append('numbersections: true')
    # sectionnumdepth is Pandoc's own template variable for \setcounter{secnumdepth}{N}.
    # 1 = number chapters + sections only; subsections (###) never get numbered.
    # This prevents X.0.Y entries when chapters jump # → ### without a ## in between.
    parts.append('sectionnumdepth: 1')
    parts.append('bibliography: refs.bib')
    parts.append('---')
    parts.append("")

    # Strip claim citations: single [c23] or multi [c23, c1593] or [@c23]
    claim_ref_re = re.compile(r'\s*\[@?c\d+(?:,\s*@?c\d+)*\]')
    # Strip embedded video section numbers from headings (e.g. "### 12.1 Title" → "### Title")
    heading_num_re = re.compile(r'^(#{1,6}\s+)\d+\.\d+\s+', flags=re.MULTILINE)
    h2_re = re.compile(r"^##\s+", flags=re.MULTILINE)
    deep_heading_re = re.compile(r"^(#{3,6})(\s+)", flags=re.MULTILINE)

    def _shift_headings_up(md: str) -> str:
        """If chapter has only one ## (the title), promote ### → ## and #### → ###.
        Without this, ### renders as \\subsection under a phantom section 0,
        producing X.0.Y numbering in the TOC."""
        if len(h2_re.findall(md)) > 1:
            return md  # Real ## sections exist; structure is already correct
        return deep_heading_re.sub(lambda m: m.group(1)[1:] + m.group(2), md)

    # Chapters
    for ch in outline.get("chapters") or []:
        chap_dir = cfg.path("chapters") / f"{ch['number']:02d}_{ch['id']}"
        polished = chap_dir / "polished.md"
        if not polished.exists():
            log.warning("Skipping chapter %s in manuscript: no polished.md", ch["id"])
            continue
        md = polished.read_text()
        md = _replace_figures(md, ch["id"], figures, cfg)
        # Normalize chapter heading to top-level (Pandoc book class will paginate)
        # The draft prompt instructed `## <title>`; lift to `# <title>` for book class.
        md = _shift_headings_up(md)
        md = re.sub(r"^##\s+", "# ", md, count=1, flags=re.MULTILINE)
        md = heading_num_re.sub(r'\1', md)
        md = claim_ref_re.sub("", md)
        parts.append(md.rstrip())
        parts.append("\n\\clearpage\n")

    # Appendices
    appendices = outline.get("appendices") or []
    if appendices:
        parts.append("\n\\appendix\n")
        for ap in appendices:
            chap_dir = cfg.path("chapters") / f"{ap['number']:02d}_{ap['id']}"
            polished = chap_dir / "polished.md"
            if not polished.exists():
                continue
            md = polished.read_text()
            md = _replace_figures(md, ap["id"], figures, cfg)
            md = re.sub(r"^##\s+", "# ", md, count=1, flags=re.MULTILINE)
            md = heading_num_re.sub(r'\1', md)
            md = claim_ref_re.sub("", md)
            parts.append(md.rstrip())
            parts.append("\n\\clearpage\n")

    manuscript_md = cfg.path("book_md")
    manuscript_md.parent.mkdir(parents=True, exist_ok=True)
    manuscript_md.write_text("\n".join(parts))
    return manuscript_md


def _compute_pipeline_stats(cfg: Config) -> dict:
    stats = {"total_claims": 0, "superseded": 0, "grounding_pct": 0.0, "video_count": 0}
    db = cfg.path("claims_db")
    if db.exists():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        stats["total_claims"] = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        stats["superseded"] = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE extra LIKE '%superseded_by%'"
        ).fetchone()[0]
        conn.close()
    manifest_path = cfg.path("raw") / "manifest.json"
    if manifest_path.exists():
        stats["video_count"] = len(read_json(manifest_path))
    grounded = narrative = unsupported = 0
    for f in cfg.path("chapters").glob("*/verify.json"):
        try:
            for s in json.loads(f.read_text()).get("sentences", []):
                st = s.get("status", "")
                if st == "grounded":
                    grounded += 1
                elif st == "narrative_ok":
                    narrative += 1
                elif st == "unsupported":
                    unsupported += 1
        except Exception:
            pass
    total_audited = grounded + narrative + unsupported
    if total_audited:
        stats["grounding_pct"] = round((grounded + narrative) / total_audited * 100, 1)
    return stats


def _write_latex_titlepage(cfg: Config, stats: dict) -> Path:
    playlist_url = cfg.playlist.get("playlist_url", "")
    book = cfg.playlist.get("book", {})
    title = _latex_escape(book.get("title", ""))
    subtitle = _latex_escape(book.get("subtitle", ""))
    author = _latex_escape(book.get("author", ""))
    date = datetime.now().strftime(book.get("date_format", "%B %Y"))
    total = f"{stats['total_claims']:,}".replace(",", "{,}")
    superseded = f"{stats['superseded']:,}".replace(",", "{,}")
    grounding = f"{stats['grounding_pct']:.1f}\\%"
    videos = str(stats["video_count"])
    tex = (
        "\\renewcommand{\\maketitle}{%\n"
        "  \\begin{titlepage}\n"
        "  \\centering\n"
        "  \\vspace*{2.5cm}\n"
        f"  {{\\Huge\\bfseries {title}\\par}}\n"
        "  \\vspace{0.6cm}\n"
        f"  {{\\large\\itshape {subtitle}\\par}}\n"
        "  \\vspace{2.5cm}\n"
        "  \\begin{tabular}{@{}lr@{}}\n"
        "    \\toprule\n"
        "    \\multicolumn{2}{c}{{\\large\\bfseries Pipeline Statistics}} \\\\\n"
        "    \\midrule\n"
        f"    Videos processed   & {videos} \\\\\n"
        f"    Claims extracted   & {total} \\\\\n"
        f"    Superseded claims  & {superseded} \\\\\n"
        f"    Grounded sentences & {grounding} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\par\\vspace{1.8cm}\n"
        "  {\\small Source playlist:}\\par\\smallskip\n"
        f"  {{\\small\\url{{{playlist_url}}}}}\\par\n"
        "  \\vfill\n"
        f"  {{\\large {author}\\par}}\n"
        "  \\vspace{0.5cm}\n"
        f"  {{\\large {date}\\par}}\n"
        "  \\end{titlepage}\n"
        "}\n"
    )
    out = cfg.root / "book" / "latex_titlepage.tex"
    out.write_text(tex)
    return out


def _write_bibtex(cfg: Config) -> Path:
    refs = CanonicalRefIndex(cfg.pipeline.get("canonical_refs") or [])
    bib = cfg.path("refs_bib")
    bib.parent.mkdir(parents=True, exist_ok=True)
    bib.write_text(refs.bib_entries())
    return bib


def _run_pandoc(cfg: Config, manuscript_md: Path) -> Path:
    out_pdf = cfg.path("book_pdf")
    out_tex = cfg.path("book_tex")
    book_dir = manuscript_md.parent

    # Build relative path to bib from manuscript dir (os.path.relpath handles
    # the case where the bib lives outside the book dir)
    refs = cfg.path("refs_bib")
    try:
        bib_rel = Path(os.path.relpath(refs, book_dir))
    except ValueError:
        # Different drive on Windows etc; fall back to absolute
        bib_rel = refs

    # LaTeX header with tikz and other packages needed for figures
    latex_header = cfg.root / "book" / "latex_header.tex"

    common_args = [
        "--from", "markdown+raw_tex+tex_math_dollars+fenced_code_attributes+pipe_tables",
        "--pdf-engine", "xelatex",
        "--syntax-highlighting=idiomatic",
        "--standalone",
        "--toc",
        f"--resource-path={cfg.root}:{book_dir}",
        f"--bibliography={bib_rel}",
        "--metadata=link-citations:true",
    ]
    if latex_header.exists():
        common_args.append(f"--include-in-header={latex_header}")
    latex_titlepage = cfg.root / "book" / "latex_titlepage.tex"
    if latex_titlepage.exists():
        common_args.append(f"--include-in-header={latex_titlepage}")

    # First produce .tex (handy for debugging)
    log.info("Pandoc: building manuscript.tex")
    res_tex = subprocess.run(
        ["pandoc", str(manuscript_md), "-o", str(out_tex), *common_args],
        capture_output=True,
        text=True,
        cwd=book_dir,
    )
    if res_tex.returncode != 0:
        log.error("Pandoc .tex stderr:\n%s", res_tex.stderr)
        raise RuntimeError("Pandoc failed building .tex")

    # Then PDF
    log.info("Pandoc + XeLaTeX: building manuscript.pdf")
    res_pdf = subprocess.run(
        ["pandoc", str(manuscript_md), "-o", str(out_pdf), *common_args],
        capture_output=True,
        text=True,
        cwd=book_dir,
    )
    if res_pdf.returncode != 0:
        log.error("Pandoc PDF stderr (tail):\n%s", res_pdf.stderr[-3000:])
        raise RuntimeError("Pandoc failed building PDF")
    return out_pdf


def run(cfg: Config) -> None:
    if not _have_tool("pandoc"):
        raise RuntimeError("pandoc not found. Install pandoc + texlive-xetex.")
    if not _have_tool("xelatex"):
        raise RuntimeError("xelatex not found. Install texlive-xetex.")

    log.info("Building manuscript.md")
    manuscript_md = _build_manuscript_md(cfg)
    log.info("Writing refs.bib")
    _write_bibtex(cfg)
    stats = _compute_pipeline_stats(cfg)
    log.info("Stats: %d claims, %d superseded, %.1f%% grounded",
             stats["total_claims"], stats["superseded"], stats["grounding_pct"])
    _write_latex_titlepage(cfg, stats)
    log.info("Running Pandoc")
    pdf = _run_pandoc(cfg, manuscript_md)
    log.info("Built: %s", pdf)
