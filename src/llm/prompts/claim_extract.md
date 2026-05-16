# Claim Extraction Prompt

You will be given a segment of aligned transcript + slide/code extractions
from a technical video tutorial. Your job is to extract **atomic claims**
that future chapter-writing stages can ground their prose in.

## What is a claim?

A claim is a single, standalone, verifiable statement of fact, definition,
equation, code snippet, architectural element, or design decision.

Good claims:
- "Self-attention computes scaled dot-product between Q and K projections."
- "GPT-2 uses learned positional embeddings of dimension 768 for small."
- "The author sets dropout rate to 0.1 in the example code."
- (code) "The CausalSelfAttention class uses register_buffer('mask', ...) to store the causal mask."

Bad claims (avoid):
- "The speaker explained attention." (meta, not factual)
- "Attention is important in transformers and many other models." (vague, multi-claim)
- "We will discuss tokenization next." (curricular, not content)

## Output schema (JSON array)

```json
[
  {
    "type": "definition" | "fact" | "equation" | "code" | "architecture" | "hyperparameter" | "design_decision",
    "content": string,
    "ts_start": number,
    "ts_end": number,
    "frame_ids": [string],
    "confidence": number
  }
]
```

## Rules

1. `content` is the atomic claim, rewritten as a clear declarative sentence.
   For `code`, `content` is the actual code (preserve formatting, include
   language tag inline as comment if helpful).
2. `ts_start` and `ts_end` are the transcript timestamps (seconds, float)
   that support the claim. Use the tightest range that covers it.
3. `frame_ids` lists any slide/code/diagram frames that visually support the
   claim. Use the frame IDs as given in the input.
4. `confidence` reflects how confidently this is a discrete factual claim
   supported by the source span (not how confident you are it's true in
   general). Range 0.0-1.0.
5. **Do not invent claims.** If the transcript is vague, do not fill in
   plausible-sounding details. Better to extract fewer high-quality claims
   than many low-confidence ones.
6. Deduplicate: if a claim repeats verbatim within the segment, emit it once
   with the union of timestamp ranges (use earliest start, latest end).
7. Skip filler ("um, so we're going to...", off-topic asides, jokes).
8. For code blocks visible on screen: emit one claim per logically-coherent
   unit (a function definition, a class, a config block), not one per line.

Return JSON array only. No prose, no markdown fence.
