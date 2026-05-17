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

    # Title metadata block (YAML for Pandoc).
    # `title:` must be set so Pandoc emits \maketitle, which is overridden by
    # the renewcommand in book/latex_titlepage.tex (written by _write_latex_titlepage).
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
    # NOTE: YAML `header-includes` is intentionally NOT used here. Pandoc's
    # `--include-in-header=...` CLI flag (passed in _run_pandoc) replaces
    # the YAML key, so all our overrides live in latex_titlepage.tex /
    # latex_header.tex instead.
    parts.append('geometry: margin=1in')
    parts.append('linkcolor: blue')
    parts.append('toc: true')
    # Chapter + section only in TOC; deeper subsections clutter the listing.
    parts.append('toc-depth: 1')
    parts.append('numbersections: true')
    parts.append('bibliography: refs.bib')
    parts.append('---')
    parts.append("")

    # Strip claim citations: single [c23] or multi [c23, c1593] or [@c23]
    claim_ref_re = re.compile(r'\s*\[@?c\d+(?:,\s*@?c\d+)*\]')
    # Strip embedded video section numbers from headings (e.g. "### 12.1 Title" → "### Title")
    heading_num_re = re.compile(r'^(#{1,6}\s+)\d+\.\d+\s+', flags=re.MULTILINE)
    h2_re = re.compile(r"^##\s+", flags=re.MULTILINE)
    deep_heading_re = re.compile(r"^(#{3,6})(\s+)", flags=re.MULTILINE)

    # Regex to match fenced code blocks (``` or ~~~) so we can protect them
    _code_block_re = re.compile(r"^(`{3,}|~{3,}).*?\n.*?\1\s*$", flags=re.MULTILINE | re.DOTALL)

    def _mask_code_blocks(md: str) -> tuple[str, list[str]]:
        """Replace code blocks with placeholders so heading regexes don't touch them."""
        blocks: list[str] = []
        def _save(m: re.Match) -> str:
            blocks.append(m.group(0))
            return f"\n<!--CODE_BLOCK_{len(blocks) - 1}-->\n"
        masked = _code_block_re.sub(_save, md)
        return masked, blocks

    def _unmask_code_blocks(md: str, blocks: list[str]) -> str:
        """Restore code blocks from placeholders."""
        for i, block in enumerate(blocks):
            md = md.replace(f"<!--CODE_BLOCK_{i}-->", block)
        return md

    def _shift_headings_up(md: str) -> str:
        """If chapter has only one ## (the title), promote ### → ## and #### → ###.
        Without this, ### renders as \\subsection under a phantom section 0,
        producing X.0.Y numbering in the TOC."""
        if len(h2_re.findall(md)) > 1:
            return md  # Real ## sections exist; structure is already correct
        return deep_heading_re.sub(lambda m: m.group(1)[1:] + m.group(2), md)

    def _normalize_chapter_headings(md: str) -> str:
        """Ensure chapter has exactly one top-level `# Title` heading.

        Some polished chapters already start with `# Title`; others use
        `## Title`.  Only promote the first `##` → `#` when the chapter
        does NOT already begin with a `#` heading.
        """
        # Mask code blocks so `# comments` inside code aren't mangled
        md, blocks = _mask_code_blocks(md)

        # Check if the document already starts with a single # heading
        first_line = md.lstrip().split("\n", 1)[0]
        already_has_h1 = first_line.startswith("# ") and not first_line.startswith("## ")

        md = _shift_headings_up(md)

        if not already_has_h1:
            # Promote the first ## to # (chapter title)
            md = re.sub(r"^##\s+", "# ", md, count=1, flags=re.MULTILINE)

        md = heading_num_re.sub(r'\1', md)

        # Restore code blocks
        md = _unmask_code_blocks(md, blocks)
        return md

    # Chapters
    for ch in outline.get("chapters") or []:
        chap_dir = cfg.path("chapters") / f"{ch['number']:02d}_{ch['id']}"
        polished = chap_dir / "polished.md"
        if not polished.exists():
            log.warning("Skipping chapter %s in manuscript: no polished.md", ch["id"])
            continue
        md = polished.read_text()
        md = _replace_figures(md, ch["id"], figures, cfg)
        md = _normalize_chapter_headings(md)
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
            md = _normalize_chapter_headings(md)
            md = claim_ref_re.sub("", md)
            parts.append(md.rstrip())
            parts.append("\n\\clearpage\n")

    manuscript_md = cfg.path("book_md")
    manuscript_md.parent.mkdir(parents=True, exist_ok=True)
    
    # Replace Unicode minus sign with ASCII hyphen to prevent LaTeX errors in code blocks
    final_text = "\n".join(parts).replace("−", "-")
    manuscript_md.write_text(final_text)
    
    return manuscript_md


def _compute_pipeline_stats(cfg: Config) -> dict:
    """Stats shown on the title page.

    Reports two grounding signals:
      * citation_pct   — share of claim-bearing sentences that carry a verified
                         citation. The strictest measure of grounding.
      * grounding_pct  — share of ALL sentences kept after verification
                         (grounded + narrative_ok). Unsupported sentences are
                         stripped before typeset, so this is what survives.
    """
    stats = {
        "total_claims": 0,
        "superseded": 0,
        "grounding_pct": 0.0,
        "citation_pct": 0.0,
        "video_count": 0,
        "chapter_count": 0,
        "book_sentences": 0,
    }
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
    outline_path = cfg.path("outline")
    if outline_path.exists():
        try:
            stats["chapter_count"] = len(read_json(outline_path).get("chapters") or [])
        except Exception:
            pass
    grounded = narrative = unsupported = external = needs_check = 0
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
                elif st == "external_ok":
                    external += 1
                elif st == "needs_external_check":
                    needs_check += 1
        except Exception:
            pass
    total_audited = grounded + narrative + unsupported + external + needs_check
    claim_bearing = total_audited - narrative
    stats["book_sentences"] = grounded + narrative + external
    if total_audited:
        stats["grounding_pct"] = round(
            (grounded + narrative + external) / total_audited * 100, 1
        )
    if claim_bearing:
        stats["citation_pct"] = round((grounded + external) / claim_bearing * 100, 1)
    return stats


def _write_latex_titlepage(cfg: Config, stats: dict) -> Path:
    playlist_url = cfg.playlist.get("playlist_url", "")
    book = cfg.playlist.get("book", {})
    title = _latex_escape(book.get("title", ""))
    subtitle = _latex_escape(book.get("subtitle", ""))
    author = _latex_escape(book.get("author", ""))
    date = datetime.now().strftime(book.get("date_format", "%B %Y"))
    # `{,}` keeps LaTeX from inserting math-mode spacing around the thousands
    # separator (a `,` token alone would be parsed as a math binary operator
    # if the table cell entered math mode for any reason).
    total = f"{stats['total_claims']:,}".replace(",", "{,}")
    superseded = f"{stats['superseded']:,}".replace(",", "{,}")
    book_sentences = f"{stats['book_sentences']:,}".replace(",", "{,}")
    citation = f"{stats['citation_pct']:.1f}\\%"
    grounding = f"{stats['grounding_pct']:.1f}\\%"
    videos = str(stats["video_count"])
    chapters = str(stats["chapter_count"])
    # Header overrides: Pandoc's `--include-in-header` CLI flag shadows YAML
    # `header-includes`, so override the section numbering depth here.
    # Default Pandoc emits \setcounter{secnumdepth}{5} when numbersections=true;
    # cap at 1 (chapter+section) so `### foo` never becomes `6.0.1 foo`.
    tex = (
        "\\setcounter{secnumdepth}{1}\n"
        "\\setcounter{tocdepth}{1}\n"
        "\\pagestyle{plain}\n"
        "\\renewcommand{\\maketitle}{%\n"
        "  \\begin{titlepage}\n"
        "  \\centering\n"
        "  \\vspace*{2cm}\n"
        f"  {{\\Huge\\bfseries {title}\\par}}\n"
        "  \\vspace{0.6cm}\n"
        f"  {{\\large\\itshape {subtitle}\\par}}\n"
        "  \\vspace{2cm}\n"
        "  \\begin{tabular}{@{}lr@{}}\n"
        "    \\toprule\n"
        "    \\multicolumn{2}{c}{{\\large\\bfseries Build statistics}} \\\\\n"
        "    \\midrule\n"
        f"    Videos processed        & {videos} \\\\\n"
        f"    Chapters                & {chapters} \\\\\n"
        f"    Atomic claims extracted & {total} \\\\\n"
        f"    Superseded claims       & {superseded} \\\\\n"
        f"    Sentences in book       & {book_sentences} \\\\\n"
        f"    Citation rate           & {citation} \\\\\n"
        f"    Verification pass rate  & {grounding} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\par\\vspace{1.2cm}\n"
        "  \\begin{minipage}{0.82\\textwidth}\\centering\\footnotesize\n"
        "  \\textit{Every sentence maps back to a transcript span, slide frame, "
        "or canonical reference. Citation rate measures claim-bearing sentences "
        "with a verified citation; verification pass rate counts all sentences "
        "that survived audit.}\n"
        "  \\end{minipage}\n"
        "  \\par\\vspace{1.4cm}\n"
        "  {\\small\\textbf{Source playlist}\\par}\\smallskip\n"
        f"  {{\\small\\url{{{playlist_url}}}}}\\par\n"
        "  \\vfill\n"
        f"  {{\\large {author}\\par}}\n"
        "  \\vspace{0.4cm}\n"
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
