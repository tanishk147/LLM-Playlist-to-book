# Chapter Draft Prompt

You are writing one chapter of a technical eBook titled "Building Large
Language Models from Scratch." The book is grounded in a specific YouTube
playlist; you have access to the extracted claims from that playlist as
your source material.

## Your task

Write a complete chapter draft for the chapter described below, based solely
on the supplied claims. The chapter should read like a published technical
book chapter, not like a transcript or a summary.

## Hard rules

1. **Every factual sentence must carry an inline citation.** Citations are
   in the form `[c123]` where `c123` is a claim ID from the supplied claim
   list. A single sentence may carry multiple citations, e.g. `[c12, c47]`.
2. Sentences that are pure narrative connective tissue (e.g. "We now turn to
   the question of...", "In the next section,...") may omit citations.
3. **Do not introduce facts, code, numbers, equations, or technical
   assertions that are not in the supplied claims.** If you feel a key
   piece is missing, write `[GAP: <what's missing>]` inline rather than
   inventing it. The verification pass will follow up on gaps.
4. **Code blocks must be reproduced verbatim from claims of type `code`.**
   You may add a 1-2 sentence preamble explaining the code, but the code
   itself comes directly from a claim. Cite the claim ID on the line above
   the fence (e.g. `# Source: [c47]`).
5. **Equations and hyperparameters must be cited.** Do not round numbers,
   do not "clean up" formulas. Reproduce as given.
6. **No hedging language without warrant.** Do not write "it is generally
   believed that..." or "many people think...". Either the claim is supported
   by the playlist source (cite it) or it should not appear.
7. **Pedagogy is welcome.** You may explain *why* something works using your
   own words, but the *what* must come from claims. Frame your explanations
   so a verifier can distinguish your inference (educational gloss) from
   asserted fact (claim-derived). Use phrases like "intuitively," "to see
   this," or "the consequence is" to introduce your own connective reasoning.

## Style

- Audience: a CS student or working engineer who knows Python and basic ML
  but is new to LLMs.
- Tone: clear, direct, mildly informal. No marketing language.
- Length target: see `estimated_pages` × 350 words/page as a rough budget.
- Use Markdown. Headings: `##` for chapter title (use the provided title),
  `###` for sections, `####` for subsections.
- Code blocks: triple-backtick fences with language tags.
- Math: use `$...$` for inline, `$$...$$` for display.
- Diagrams: where a claim references a `diagram` frame, write
  `[FIGURE: <one-line description of what to draw>]` inline; the figure
  stage will turn this into a real figure later.

## Output format

Output ONLY the chapter markdown. No preamble. No "Here is the chapter:".
Start with the `## <chapter title>` heading.
