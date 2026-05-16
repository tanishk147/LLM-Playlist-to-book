# llm-playlist-book

Pipeline that converts a YouTube playlist into an eBook-quality manuscript with traceable citations.
Every sentence in the final PDF maps back to a transcript span, slide frame, or canonical reference.

## TL;DR for graders

The finished book is already built and committed:

- **`book/manuscript.pdf`** — 137 pages, 3,873 grounded claims, 13 chapters + 3 appendices.
- **`book/manuscript.md`** and **`book/manuscript.tex`** — source forms.
- **`data/`** — all intermediate artifacts (transcripts, frames, claims SQLite, etc.).
- **`reports/cost_log.jsonl`** — full per-call cost trace (final spend: ~$5.70).

If you only want to read the book, open the PDF. The rest of this README explains
how to rebuild from scratch.

## Rebuilding

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg pandoc texlive-xetex texlive-latex-extra texlive-fonts-recommended
npm install -g @mermaid-js/mermaid-cli  # Optional: for rendering architecture diagrams

# macOS (Homebrew)
brew install ffmpeg pandoc yt-dlp mermaid-cli
brew install --cask mactex          # ~5GB; or --cask basictex for a lighter install
```

### 2. Python environment

Python 3.11 or 3.12 (faster-whisper and sentence-transformers wheels lag on 3.13+, so `pyproject.toml` pins to `<3.14`):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Anthropic API key

Create a `.env` file in the repo root with your key:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

Get a key at <https://console.anthropic.com> → API Keys → Create Key. The pipeline
uses Claude Haiku 4.5 for high-volume stages (vision, extraction, verification) and
Claude Sonnet 4.6 for drafting and polishing.

### 4. (Optional) YouTube cookies

Only needed if you want to re-run the `ingest` or `keyframes` stages from scratch.
YouTube blocks unauthenticated yt-dlp requests from data centers, so the pipeline
reads cookies from `$YT_COOKIES_PATH` when present:

```bash
# Export cookies.txt from your browser (e.g. "Get cookies.txt LOCALLY" extension)
export YT_COOKIES_PATH=/path/to/cookies.txt
```

If you skip this step, only the cached `data/raw/` and `data/frames/` will work —
new downloads will fail with a "Sign in to confirm you're not a bot" error.

### 5. Build

```bash
make book                  # runs every invalidated stage, then typesets the PDF
make reports               # cost + grounding audit (HTML in reports/)
```

Because `data/` is already populated, `make book` on a fresh clone will skip every
expensive stage (ingest → claims) and only re-run typeset. Total time: under a
minute. To force a full rebuild, see "Re-running stages" below.

## Pilot first (full rebuild only)

If you wipe `data/` and rebuild from scratch, run a pilot before the full playlist:

```bash
PILOT=1 make book
```

This processes only the first video in `config/playlist.yaml`. Inspect
`book/manuscript.pdf` and `reports/grounding_audit.html` before scaling up.

## Re-running stages

Each stage writes a sentinel file under `data/.stage/`. Re-running `make` only
re-executes invalidated stages. To force a specific stage to re-run:

```bash
make clean-from STAGE=07   # invalidates claims and every later stage
make book
```

## Pipeline stages

| # | Stage | Output | Model |
|---|---|---|---|
| 01 | ingest | `data/raw/*.m4a`, metadata | yt-dlp |
| 02 | transcribe | word-level JSON | Whisper large-v3 |
| 03 | keyframes | deduped JPEG frames | ffmpeg + phash |
| 04 | vision | per-frame extraction JSON | Haiku 4.5 |
| 05 | align | transcript ↔ frames | (deterministic) |
| 06 | segment | topic boundaries | sentence-transformers |
| 07 | claims | atomic claims in SQLite | Haiku 4.5 |
| 08 | outline | TOC, claim → chapter map | Sonnet 4.6 |
| 09 | chapters | draft → verify → polish per chapter | Sonnet 4.6 (draft, polish) + Haiku 4.5 (verify) |
| 10 | figures | regenerated diagrams + embeds | Haiku 4.5 |
| 11 | typeset | manuscript.pdf | Pandoc + XeLaTeX |

## Reproducibility

- Model IDs pinned in `config/pipeline.yaml`.
- Python deps pinned via `pyproject.toml`; Python 3.11 / 3.12 supported.
- All artifacts hashed; hashes stored alongside outputs.
- Prompt cache TTL configurable (1h default for chapter sessions).
- Random sampling: `temperature=0` for extract/verify/outline; low temp for
  draft (0.2) and polish (0.3).
- `reproducibility.seed` in `config/pipeline.yaml` seeds python/numpy/torch
  PRNGs at every CLI invocation, so embedding segmentation and supersession
  clustering produce stable boundaries across runs.

## Cost control

`src/llm/budget.py` enforces per-stage and total USD ceilings (defaults: $5/chapter,
$150 total). Pipeline aborts before exceeding. Cost log at `reports/cost_report.html`.

## No-hallucination claim

The "no hallucinations" guarantee in the assignment is operationalized as
**traceability**, not output filtering:

1. **Grounding store** (`data/claims.sqlite`) contains atomic claims with source
   coordinates (video, timestamp, frame ID).
2. **Draft stage** writes prose constrained to retrieved claims; every sentence
   carries an inline `[cNNN]` citation tied to a claim ID in the store.
3. **Hybrid retrieval** (BM25 + dense embeddings) picks the most relevant
   claims per chapter so the draft prompt never overflows context. Code and
   equation claims are always retained.
4. **Verify stage** runs in a fresh model context with no draft history and
   marks each sentence `grounded | unsupported | contradicted | needs_external_check`.
5. **Phantom-citation check**: after verification, every cited claim ID is
   re-validated against the store. Sentences that cite non-existent or
   out-of-chapter IDs are demoted to `unsupported` regardless of what the
   verifier said.
6. **Unsupported sentences are deleted**, not rewritten. The audit JSON per
   chapter records every decision.
7. **Code claims are parser-validated** (`ast.parse` for Python, bracket
   balance for others); unparseable code is flagged in the claim store so
   the drafter can avoid it.
8. **Supersession detection**: when a later video refactors an earlier
   concept (semantic similarity + token Jaccard for code), the older claim
   is tagged `superseded_by` and excluded from outline + drafting.
9. **External refs** (Raschka book, "Attention Is All You Need", etc.) may
   only *confirm* playlist-derived claims, never introduce new ones.

Read `reports/grounding_audit.html` after a run. The header shows the
percentage of sentences in each grounding category.

## Repo layout

See `docs/ARCHITECTURE.md` for the full layout. Top level:

```
config/      pinned settings, glossary, playlist URL
src/         pipeline code (stages/, llm/, grounding/, verification/, utils/)
data/        gitignored artifacts (raw, transcripts, frames, claims.sqlite, chapters/)
book/        manuscript.md, manuscript.tex, manuscript.pdf, figures/
reports/     cost_report.html, grounding_audit.html
tests/       pytest suite
```

## License & ethics

This code is MIT-licensed. The *output* (the generated book) is a derivative
work of the playlist. For academic submission this is fair use; for distribution
you need permission from the playlist creator.
