.PHONY: all clean clean-from book setup reports pilot lint test
.DEFAULT_GOAL := book

PY := python -m src.cli
DATA := data
BOOK := book
STAGES := $(DATA)/.stage
CFG := config/pipeline.yaml
PLAYLIST := config/playlist.yaml
GLOSSARY := config/glossary.txt

# ---------- Setup ----------
setup:
	@which yt-dlp ffmpeg pandoc xelatex >/dev/null || \
	  (echo "Missing system deps. Install: yt-dlp ffmpeg pandoc texlive-xetex" && exit 1)
	@test -f .env || (echo "Create .env from .env.example" && exit 1)
	@mkdir -p $(STAGES)

# ---------- Pipeline ----------
$(STAGES)/01_ingest: $(CFG) $(PLAYLIST) | setup
	$(PY) ingest
	@touch $@

$(STAGES)/02_transcribe: $(STAGES)/01_ingest $(GLOSSARY)
	$(PY) transcribe
	@touch $@

$(STAGES)/03_keyframes: $(STAGES)/01_ingest
	$(PY) keyframes
	@touch $@

$(STAGES)/04_vision: $(STAGES)/03_keyframes
	$(PY) vision
	@touch $@

$(STAGES)/05_align: $(STAGES)/02_transcribe $(STAGES)/04_vision
	$(PY) align
	@touch $@

$(STAGES)/06_segment: $(STAGES)/05_align
	$(PY) segment
	@touch $@

$(STAGES)/07_claims: $(STAGES)/06_segment
	$(PY) claims
	@touch $@

$(STAGES)/08_outline: $(STAGES)/07_claims
	$(PY) outline
	@touch $@

$(STAGES)/09_chapters: $(STAGES)/08_outline
	$(PY) chapters
	@touch $@

$(STAGES)/10_figures: $(STAGES)/09_chapters
	$(PY) figures
	@touch $@

$(STAGES)/11_typeset: $(STAGES)/10_figures
	$(PY) typeset
	@touch $@

book: $(STAGES)/11_typeset
	@echo ""
	@echo "Built: $(BOOK)/manuscript.pdf"

pilot:
	PILOT=1 $(MAKE) book

reports: $(STAGES)/11_typeset
	$(PY) report cost
	$(PY) report grounding
	@echo "Reports: reports/cost_report.html reports/grounding_audit.html"

# ---------- Re-entry ----------
clean-from:
	@test -n "$(STAGE)" || (echo "Usage: make clean-from STAGE=07" && exit 1)
	@find $(STAGES) -maxdepth 1 -type f | sort | awk -F/ -v s="$(STAGE)" '$$NF >= s"_" {print}' | xargs -r rm -v

clean:
	rm -rf $(DATA) $(BOOK)/manuscript.pdf $(BOOK)/manuscript.tex $(BOOK)/manuscript.md $(BOOK)/figures/generated/* $(BOOK)/figures/embedded/* reports/*.html

# ---------- Dev ----------
lint:
	ruff check src tests

test:
	pytest -q
