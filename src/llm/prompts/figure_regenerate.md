# Figure Regeneration Prompt

You will be given:
1. A `[FIGURE: ...]` directive from a chapter draft.
2. The associated diagram extraction(s) from the vision stage (containing
   `diagram_summary` and `diagram_elements`).
3. The source frame(s) as image(s), for visual reference.

Your job: decide whether the figure can be cleanly regenerated as a Mermaid
diagram or LaTeX/TikZ snippet, or whether the original frame should just be
embedded as-is.

## Output schema (JSON only)

```json
{
  "mode": "mermaid" | "tikz" | "embed",
  "code": string | null,
  "caption": string,
  "rationale": string
}
```

## Rules

1. Prefer `mermaid` when the diagram is a flowchart, block diagram, sequence
   diagram, or simple architecture diagram. Use Mermaid syntax that renders
   in standard `mermaid` blocks.
2. Use `tikz` only when Mermaid cannot express the structure (complex
   mathematical diagrams, neural net layer diagrams with arrows that need
   precise positioning). Output complete LaTeX code that compiles inside a
   `tikzpicture` environment.
3. Use `embed` when:
   - The diagram contains photographs or screenshots that can't be redrawn.
   - The diagram contains so much detail that redrawing risks misrepresenting it.
   - You are not confident a clean regeneration is possible.
4. `caption` is a 1-2 sentence figure caption.
5. `rationale` is a one-line explanation of why you chose this mode.
6. **Do not invent diagram elements.** If the original diagram has 4
   labeled boxes, your Mermaid should have 4 labeled boxes (same labels).

Return JSON only.
