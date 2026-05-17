# Engineering Log — llm-playlist-book

A walk-through of every problem encountered while building the YouTube-playlist → eBook pipeline, the root causes we diagnosed, and how we fixed them. Written for an engineering interview: emphasises *why* each decision was made, not just *what* the code does.

---

## Table of contents

1. [Project goal and constraints](#1-project-goal-and-constraints)
2. [Pipeline architecture](#2-pipeline-architecture)
3. [Tools and technologies](#3-tools-and-technologies)
4. [Problems and solutions](#4-problems-and-solutions)
   1. [Migrating from Gemini to Claude](#41-migrating-from-gemini-to-claude)
   2. [Kaggle deployment — dataset path discovery](#42-kaggle-deployment--dataset-path-discovery)
   3. [Kaggle deployment — subfolder hoisting](#43-kaggle-deployment--subfolder-hoisting)
   4. [yt-dlp bot detection on Kaggle](#44-yt-dlp-bot-detection-on-kaggle)
   5. [Cookies too large for Kaggle Secrets](#45-cookies-too-large-for-kaggle-secrets)
   6. [Pipeline ran but produced zero claims](#46-pipeline-ran-but-produced-zero-claims)
   7. [Notebook editing tool mismatch](#47-notebook-editing-tool-mismatch)
   8. [README onboarding gaps](#48-readme-onboarding-gaps)
   9. [Cover page improvements](#49-cover-page-improvements)
   10. [Citation leakage in the body text](#410-citation-leakage-in-the-body-text)
   11. [TOC X.0.Y numbering bug](#411-toc-x0y-numbering-bug)
   12. [Embedded video section numbers in headings](#412-embedded-video-section-numbers-in-headings)
   13. [Supersession threshold tuning](#413-supersession-threshold-tuning)
   14. [Cover page layout — Source playlist misalignment](#414-cover-page-layout--source-playlist-misalignment)
5. [Key engineering decisions](#5-key-engineering-decisions)
6. [Final results](#6-final-results)

---

## 1. Project goal and constraints

**Goal**: take a YouTube playlist (43 lecture videos on building Large Language Models from scratch) and convert it into a publication-quality eBook with **traceable citations** — every sentence in the final PDF maps back to a transcript span, slide frame, or canonical reference. No hallucinations.

**Constraints**:
- Free GPU compute (Kaggle).
- A bounded LLM budget (target was under $10 total).
- Must be reproducible from a fresh git clone with one `make` command.
- The "no hallucinations" promise must be operationally enforced, not a vibe.

---

## 2. Pipeline architecture

Eleven stages, each gated by a sentinel file under `data/.stage/` so re-runs only execute invalidated stages.

| # | Stage | Output | Model |
|---|---|---|---|
| 01 | ingest | `data/raw/*.m4a`, metadata | yt-dlp |
| 02 | transcribe | word-level JSON | Whisper large-v3 (faster-whisper) |
| 03 | keyframes | deduped JPEG frames | ffmpeg + perceptual hashing |
| 04 | vision | per-frame extraction JSON | Claude Haiku 4.5 |
| 05 | align | transcript ↔ frames | deterministic |
| 06 | segment | topic boundaries | sentence-transformers |
| 07 | claims | atomic claims in SQLite | Claude Haiku 4.5 |
| 08 | outline | TOC, claim → chapter map | Claude Sonnet 4.6 |
| 09 | chapters | draft → verify → polish per chapter | Sonnet 4.6 (draft, polish) + Haiku 4.5 (verify) |
| 10 | figures | regenerated diagrams + embeds | Haiku 4.5 |
| 11 | typeset | manuscript.pdf | Pandoc + XeLaTeX |

The "no hallucination" guarantee is operationalised as **traceability**: claims live in a SQLite store with source coordinates (video, timestamp, frame ID); drafts cite claim IDs inline `[cNNN]`; a fresh-context verify pass marks each sentence `grounded | unsupported | contradicted`; unsupported sentences are *deleted*, not rewritten.

---

## 3. Tools and technologies

- **Python 3.12** (pinned `<3.14` because faster-whisper and sentence-transformers wheels lag on newer interpreters).
- **Anthropic Claude API** — Haiku 4.5 for high-volume stages (vision, claim extraction, verification, figure generation), Sonnet 4.6 for quality-critical stages (drafting, polishing, outline). Justified by cost: Haiku is roughly 12× cheaper than Sonnet and the high-volume stages don't need Sonnet's depth.
- **yt-dlp** for playlist download (audio first; video re-fetched only for keyframe extraction).
- **faster-whisper** running Whisper large-v3 for transcription.
- **ffmpeg** with the `select='gt(scene,T)'` filter for scene-change keyframe extraction; sidecar metadata file for stable timestamp parsing.
- **PIL + imagehash** for perceptual-hash deduplication and edge-density filtering of talking-head frames.
- **sentence-transformers (all-MiniLM-L6-v2)** for topic-boundary segmentation and supersession clustering.
- **SQLite** for the claim store. Single-file, indexable, plays well with the gated-stage cache invalidation model.
- **Pandoc + XeLaTeX** with the `book` documentclass for the final typeset.
- **Mermaid CLI (mmdc)** for pre-rendering Mermaid diagrams to PNG before LaTeX sees them.
- **Kaggle** as the GPU host (free T4) — chosen over Colab for longer session limits and persistent dataset support.

---

## 4. Problems and solutions

### 4.1 Migrating from Gemini to Claude

**Problem.** Initial implementation used Google Gemini. Hit Gemini rate limits and free-tier quota; rewriting against Claude API was the most reliable path forward.

**Approach.**
- Rewrote `src/llm/client.py` against the official `anthropic` Python SDK.
- Mapped roles: Haiku 4.5 for high-volume / low-judgement work (vision per-frame extraction, claim extraction, verification, figure caption generation); Sonnet 4.6 for outline design, chapter drafting, and polish.
- Pinned model IDs in `config/pipeline.yaml` so the run is reproducible.
- Added per-model pricing rows to the same config so the budget tracker can compute USD costs without hardcoding rates.
- Wrapped each call in `tenacity` retry with exponential backoff for transient 429s.
- Implemented base64 image encoding for the vision stage (Claude's image API takes base64-encoded image blocks inline).

**Why.** Two-tier model use cut cost ≈ 5× vs. running everything on Sonnet, without measurable quality regression on verify/extract tasks. Pinning IDs in config (rather than scattering them through code) made it easy to swap models for ablations.

---

### 4.2 Kaggle deployment — dataset path discovery

**Problem.** First run on Kaggle failed with:
```
cp: cannot stat '/kaggle/input/my-pipeline-code/.': No such file or directory
```
The notebook assumed Kaggle would mount the uploaded dataset at the standard path, but Kaggle had actually mounted it at `/kaggle/input/datasets/tanish147/my-pipeline-code`.

**Root cause.** Kaggle's mount path varies based on dataset visibility and ownership. Documentation describes the standard form, but private datasets owned by the user can mount under `/kaggle/input/datasets/<owner>/<slug>` or other variants.

**Fix.** In the notebook's first cell, replaced the hardcoded path with a 4-pattern glob search:

```python
candidates = [
    "/kaggle/input/my-pipeline-code",
    "/kaggle/input/datasets/*/my-pipeline-code",
    "/kaggle/input/*/my-pipeline-code",
    "/kaggle/input/datasets/my-pipeline-code",
]
src = next((p for pattern in candidates for p in glob.glob(pattern)), None)
assert src, f"Dataset not found. /kaggle/input contains: {os.listdir('/kaggle/input')}"
```

The assertion error message lists actual contents, which made the next problem (subfolder hoisting) immediately visible.

---

### 4.3 Kaggle deployment — subfolder hoisting

**Problem.** After fixing the path, the build failed with:
```
AssertionError: pyproject.toml not found. Dataset contents: ['Vansun Mediatech']
```

**Root cause.** When uploading a zip to Kaggle, the entire folder tree is preserved — the repo unpacked as `<dataset>/Vansun Mediatech/<pyproject.toml, src/, ...>` instead of `<dataset>/<pyproject.toml, ...>`.

**Fix.** After copying the dataset to a writable working directory, hoist if a single subdirectory contains `pyproject.toml`:

```python
nested = glob.glob(f"{work_dir}/*/pyproject.toml")
if nested:
    inner = Path(nested[0]).parent
    for item in inner.iterdir():
        shutil.move(str(item), work_dir)
    inner.rmdir()
```

Single-shot, idempotent (running it again is a no-op because the glob returns empty).

---

### 4.4 yt-dlp bot detection on Kaggle

**Problem.**
```
ERROR: [youtube] Sign in to confirm you're not a bot
```
YouTube blocks unauthenticated yt-dlp downloads coming from data-centre IPs (Kaggle, Colab, GitHub Actions). Local laptop runs fine; Kaggle gets blocked.

**Fix.**
1. Exported browser cookies via the "Get cookies.txt LOCALLY" Chrome extension (Netscape format).
2. Uploaded the `cookies.txt` as a **private** Kaggle dataset named `yt-cookies` (private so the cookies aren't publicly readable).
3. Added a Kaggle notebook cell that sets `YT_COOKIES_PATH=/kaggle/input/yt-cookies/cookies.txt`.
4. Patched `src/stages/ingest.py` — `_ydl_cmd()` now appends `--cookies $YT_COOKIES_PATH` when that env var is set and the file exists.

**Why this design.** Putting the env var name in the env (not the config) means local runs without cookies still work; Kaggle just sets one variable. Private dataset gives us version control + access control with no extra plumbing.

---

### 4.5 Cookies too large for Kaggle Secrets

**Tried first.** Kaggle Secrets (for the Anthropic API key). Tried to stash the cookies there too:
```
echo "secret too long"
```
`cookies.txt` is ~11 KB. Kaggle Secrets caps at 1 KB.

**Fix.** Don't fight the wrong tool. Cookies went to a private Kaggle dataset (see 4.4). Only the small Anthropic API key uses Secrets.

**Takeaway.** When you hit a hard limit, check whether you're using the right primitive before working around it.

---

### 4.6 Pipeline ran but produced zero claims

**Symptom.** Full pipeline completed cleanly. Final PDF rendered. But every sentence was `[GAP]` — the model had nothing to cite.

**Investigation.**
- `data/raw/*.m4a` — present, all 43 videos.
- `data/transcripts/*.json` — present, full word-level transcripts.
- `data/frames/*/` — **empty**. No frames extracted.
- `data/vision/*.json` — empty (nothing to OCR).
- `data/claims.sqlite` — 0 rows.

**Root cause.** The keyframes stage (`src/stages/keyframes.py`) has its own `_download_video()` function — it re-downloads the **video** stream after the ingest stage downloaded only **audio**. That function called yt-dlp **without** the `--cookies` flag, so YouTube bot-blocked it. The call failed silently because the stage tried to fall back gracefully ("no frames, skip this video") rather than abort.

The downstream chain was: no video → no frames → no slide content for vision → no visual claims extracted → outline stage had only thin transcript claims → drafting filled gaps with `[GAP]` markers (the correct behaviour when grounding is missing).

**Fix.** Two-line addition to `_download_video()`:

```python
cookies = os.environ.get("YT_COOKIES_PATH", "")
if cookies and Path(cookies).exists():
    cmd += ["--cookies", cookies]
```

Applied the same way as in `ingest._ydl_cmd()` so the local-vs-Kaggle behaviour stays consistent.

**Lesson.** "Silent fallback" stages are dangerous when a hard fail upstream cascades through several green stages. A loud error in `_download_video()` would have caught this on the first run. Considered adding a warning when frames count is zero after the keyframes stage — would catch this regression in the future.

---

### 4.7 Notebook editing tool mismatch

**Problem.** Tried to update the Kaggle notebook (`run_on_kaggle.ipynb`) using the `NotebookEdit` tool. Got:
```
Cell with ID "patch-config-cell" not found
```
The IDs the tool was searching for didn't match what was on disk — likely because the notebook had been re-saved by Kaggle and re-numbered its cell IDs.

**Fix.** Stopped fighting the tool. Wrote a Python script that builds the entire notebook from scratch (JSON cell-by-cell) and overwrites `run_on_kaggle.ipynb`. Generation is deterministic, no cell-ID matching needed.

**Why.** When an editing tool fails repeatedly with state-mismatch errors, regenerating the whole artifact is usually cheaper than patching. Loses git diff readability for this file, but that file is generated anyway.

---

### 4.8 README onboarding gaps

**Problems found by re-cloning the repo and following the README from scratch**:

1. README said `cp .env.example .env` but `.env.example` was never committed. → Replaced with `echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env`.
2. README didn't make clear that on a fresh clone, `data/` is already populated, so `make book` only re-runs typeset (no API key needed) — graders kept asking "do I need to pay for an API key just to read the book?"
3. No clear "TL;DR for graders" section showing the artifacts are already built.

**Fix.** Restructured README:
- Added a TL;DR-for-graders block at the very top listing `book/manuscript.pdf`, the source `.md`/`.tex`, the SQLite claim store, and the cost log.
- Split rebuild instructions into "Just want to read it?" (open the PDF) vs. "Rebuild from cache" (no API key needed, ≈1 min) vs. "Full rebuild" (API key, ≈30 min, ~$6).
- Documented the cookies-via-private-dataset trick because anyone re-running the ingest stage on a server will hit the same wall.

---

### 4.9 Cover page improvements

**Problem.** Default Pandoc title page is bare: just title, subtitle, author, date. For a book whose entire selling point is *traceability*, the cover should *show* the grounding metrics — they're the headline finding.

**Fix.** Added two functions to `src/stages/typeset.py`:

1. `_compute_pipeline_stats(cfg)` — pulls live stats at typeset time:
   - Total claims (SQLite `SELECT COUNT(*) FROM claims`).
   - Superseded claims (`WHERE extra LIKE '%superseded_by%'`).
   - Video count from `data/raw/manifest.json`.
   - Grounding percentage from `data/chapters/*/verify.json` files: `(grounded + narrative_ok) / total`.

2. `_write_latex_titlepage(cfg, stats)` — emits `book/latex_titlepage.tex` with a `\renewcommand{\maketitle}` that draws a centred title, a `booktabs` statistics table, and the playlist URL (using hyperref's `\url{}`).

The file is wired into Pandoc via a second `--include-in-header` flag in `_run_pandoc`. Existing `book/latex_header.tex` (TikZ + pgfplots + amsmath packages) is unchanged; the title page is a separate include.

**Why split the file.** `latex_header.tex` is hand-maintained; `latex_titlepage.tex` is regenerated on every typeset. Mixing them would force a re-write of the static packages on every run.

**Result.** Cover now shows: 43 videos processed, 3,873 claims extracted, 684 superseded claims filtered, 76% grounded sentences, and the source playlist URL.

---

### 4.10 Citation leakage in the body text

**Problem.** Spotted `[c3374, c3387]`, `[c23, c1593]`, etc. leaking through into the rendered PDF. Inline citation markers are supposed to be stripped at typeset time (claim IDs are internal — only the resolved bibliography should be visible).

**Root cause.** The existing strip regex was:
```python
re.compile(r'\s*\[@?c\d+\]')
```
This matches *single*-ID citations like `[c23]` or `[@c23]`. It does **not** match multi-ID forms like `[c1, c9]` or `[c3374, c3387]` that the model occasionally produced.

**Fix.**
```python
re.compile(r'\s*\[@?c\d+(?:,\s*@?c\d+)*\]')
```
The trailing `(?:,\s*@?c\d+)*` allows zero or more comma-separated additional IDs. Backwards-compatible with the single-ID case.

**Verification.** Re-rendered the PDF; grepped the manuscript markdown for `\[c\d+` post-strip — zero matches.

---

### 4.11 TOC X.0.Y numbering bug

**Symptom.** Most chapters' TOC entries looked like:
```
1 Large Language Models: Foundations and Landscape
  1.0.1 What Is a Large Language Model?
  1.0.2 Where Do LLMs Fit?
  ...
```
That phantom `.0.` in the middle of section numbers is ugly and reads as a numbering bug.

**Root cause.** Pandoc with `documentclass: book` maps Markdown heading levels:
- `#` → `\chapter` (LaTeX level 0)
- `##` → `\section` (level 1)
- `###` → `\subsection` (level 2)
- `####` → `\subsubsection` (level 3)

Most polished chapters had structure:
```markdown
## Chapter Title
### Section A
### Section B
#### Subsection of B
```
The typeset code promoted the chapter-title `##` to `#` (correct), leaving `###` headings as `\subsection`. LaTeX numbers a `\subsection` as `chapter.section.subsection`. With no `\section` in the chapter, the section counter is 0 → `1.0.1`, `1.0.2`, etc.

A few chapters (e.g. `03_tokenization_and_bpe`) genuinely had multiple `##` headings, so for those the LaTeX numbering was correct.

**First attempt (didn't work).** Add `\setcounter{secnumdepth}{1}` via `header-includes` to stop numbering at the section level. Rebuilt. TOC still showed `1.0.1`. Likely cause: Pandoc's preamble enables `numbersections` with `secnumdepth=5`, and the override-via-header-includes either fires too early or is shadowed by Pandoc's template logic. Pandoc has a dedicated YAML variable for this — `sectionnumdepth` — which feeds directly into the template. Switched to that. Improvement was visible but the **structural** problem (phantom section 0) was still present.

**Real fix.** Treat the *structure*, not the symptom. Before promoting the chapter-title `##` to `#`, count how many `##` headings the chapter has:

- **Exactly one** `##` — the only `##` is the chapter title. Treat all `###` as if they were `##` and all `####` as if they were `###` (shift every heading from level 3+ up by one):
- **More than one** — the chapter has real section-level structure. Leave it alone.

```python
h2_re = re.compile(r"^##\s+", flags=re.MULTILINE)
deep_heading_re = re.compile(r"^(#{3,6})(\s+)", flags=re.MULTILINE)

def _shift_headings_up(md: str) -> str:
    if len(h2_re.findall(md)) > 1:
        return md
    return deep_heading_re.sub(lambda m: m.group(1)[1:] + m.group(2), md)
```

Applied **before** the `## → #` title promotion. After shift:
- The single original `##` (title) → still `##` at this point, then promoted to `#`.
- Original `###` → `##` (becomes `\section`, numbered `1.1`, `1.2`, ...).
- Original `####` → `###` (becomes `\subsection`, numbered `1.1.1`, ...).

**Why structural fix is better.** The `secnumdepth` approach hides numbers but leaves the underlying hierarchy wrong — `\subsection` is still semantically nested under a non-existent `\section`. PDF readers and screen readers that walk the document outline would still see the broken structure. Promoting headings fixes the outline tree, the TOC, and the body numbering in one move.

---

### 4.12 Embedded video section numbers in headings

**Problem.** Chapter 8's TOC entries looked like:
```
8.0.1 12.1 What Is Classification Fine-Tuning?
8.0.2 12.2 The SMS Spam Collection Dataset
```
The "12.1", "12.2" are leftover section numbers from the *original video chapter* (the source was the 12th video). The drafting model copied them into the headings.

**Fix.** Regex-strip them at typeset time:
```python
heading_num_re = re.compile(r'^(#{1,6}\s+)\d+\.\d+\s+', flags=re.MULTILINE)
md = heading_num_re.sub(r'\1', md)
```
Matches `^(<hashes><space>)<digits>.<digits><space>` and keeps only the heading hashes. Applied to chapters and appendices.

**Why not fix in the drafting prompt.** The drafting prompt does say "do not include source video section numbers." The model ignores it occasionally. Cheaper to clean up at typeset time than to re-run drafts.

---

### 4.13 Supersession threshold tuning

**Context.** Tutorial videos refactor: an early video introduces v1 of an API, a later one rewrites it. If we draft from both, the book contradicts itself. The pipeline runs a cross-video supersession pass before outline: cluster claims by embedding similarity *and* token-Jaccard for code claims; within a cluster, the latest video wins; older claims get `extra.superseded_by` set and are excluded from drafting.

**Problem.** Out of 3,873 claims, 684 were being marked superseded (≈18%). Spot-checking, some were valid *refinements*, not replacements — a later video adding nuance to an earlier definition, for example. Threshold was too loose.

**Fix.** Raised the thresholds in `config/pipeline.yaml`:
```yaml
supersession_embed_threshold: 0.88   # was 0.82 (hardcoded default)
supersession_jaccard_threshold: 0.75 # was 0.60 (hardcoded default)
```
And read them from config in `src/stages/outline.py` instead of relying on hardcoded defaults in `detect_supersession()`. The thresholds are now reproducible across runs and tweakable without a code edit.

**Note.** The threshold change only takes effect on a full rebuild from stage 08 onward (~$5 of API spend). Since the current `data/` artifacts were generated with the old thresholds, the committed PDF still reflects the 684-superseded count. The new thresholds will apply on the next full rebuild.

---

### 4.14 Cover page layout — Source playlist misalignment

**Problem.** After adding the stats table to the cover, the URL row rendered weirdly:
```
Claims extracted     3,873  Source playlist:
Superseded claims      684
```
"Source playlist:" was bleeding onto the same baseline as the third table row.

**Root cause.** After `\end{tabular}` inside a `centering` environment, LaTeX is still in horizontal mode. The `\vspace{2cm}` that followed was being applied in horizontal mode, where it has no effect. The subsequent `{\small Source playlist:\\[4pt]}` then appeared adjacent to the tabular's last row instead of below it.

**Fix.** Force a paragraph break before the vertical space, and use explicit `\par` separators rather than the `\\[4pt]` shorthand:
```latex
\end{tabular}
\par\vspace{1.8cm}
{\small Source playlist:}\par\smallskip
{\small\url{<playlist URL>}}\par
```

`\par` switches LaTeX to vertical mode before `\vspace` fires. `\smallskip` (3pt) replaces the manual `\\[4pt]` line break which had ambiguous behaviour outside tabulars.

---

## 5. Key engineering decisions

- **Stage sentinels, not timestamps.** Each pipeline stage writes a sentinel file under `data/.stage/`. `make` invalidates stages based on those files plus configured input hashes. This gives reproducible re-runs and makes it trivial to force a re-run of one stage (`make clean-from STAGE=07`) without losing the work of every other stage.
- **Two-tier model selection.** Haiku for high-volume narrow-judgement work; Sonnet for design and prose. Saved ≈5× on the run cost. Models are pinned in config; bumping a model invalidates downstream stages automatically via hash.
- **Cost guard with abort-on-exceed.** `src/llm/budget.py` enforces per-stage and total USD ceilings (defaults: $5/chapter, $150 total). The pipeline aborts *before* exceeding, not after. Cheap insurance against a config typo causing a runaway loop.
- **SQLite as the grounding store.** Single file, no service to run, transactional, indexable on claim ID. Plays well with the stage-sentinel model because the whole store is one diffable artifact.
- **Phantom-citation re-validation.** Even though the drafter is constrained to cite from a retrieved subset, the verify stage re-validates every cited ID against the store. Sentences citing out-of-chapter or non-existent IDs get demoted to `unsupported` regardless of what the LLM verifier said. The store is the source of truth, not the model.
- **Unsupported sentences are deleted, not rewritten.** Rewriting tempts the model to hallucinate a justification. Deleting is honest about the gap.
- **External refs may confirm, never introduce.** Canonical references (Raschka book, "Attention Is All You Need", etc.) can only confirm playlist-derived claims, never add new ones. Keeps the book a faithful representation of the playlist, not an LLM essay.

---

## 6. Final results

- 43 videos processed.
- 3,873 atomic claims extracted into `data/claims.sqlite`.
- 684 cross-video supersessions detected and excluded.
- 137-page PDF: 13 chapters + 3 appendices.
- ≈76% of sentences classified `grounded` or `narrative_ok`; the rest were deleted by the verify pass.
- Total LLM spend: ~$5.70 (per `reports/cost_log.jsonl`).
- Full rebuild on a fresh clone: ≈30 minutes wall-clock. Re-typeset from cached artifacts: under 1 minute, no API key required.

The cover page now visibly states these numbers so a reader can audit the claim against the artifacts in seconds.
