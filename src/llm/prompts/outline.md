# Book Outline Prompt

You will be given:
1. A list of all videos in the playlist with their titles and durations.
2. The full set of extracted claims, grouped by video, with claim types and
   short summaries.

Your job: design the **table of contents** for a coherent eBook covering the
content of the playlist. Videos rarely map 1:1 to chapters. Combine, split,
or reorder as needed for pedagogical flow.

## Output schema

```json
{
  "title": string,
  "subtitle": string,
  "chapters": [
    {
      "id": string,
      "number": integer,
      "title": string,
      "summary": string,
      "estimated_pages": integer,
      "claim_ids": [integer],
      "source_videos": [string],
      "depends_on_chapters": [string]
    }
  ],
  "appendices": [ { ...same shape as chapter... } ]
}
```

## Rules

1. `id` is a slug: snake_case, ASCII, 3-6 words max. E.g. `tokenization_and_bpe`.
2. `claim_ids` should reference the integer IDs from the input. Every claim
   should appear in exactly one chapter (or appendix), with these exceptions:
   - Claims that are clearly out-of-scope curricular meta-talk may be dropped.
   - Foundational claims that several chapters need may appear in 2 chapters
     if absolutely necessary, but prefer to put them in the earliest chapter
     and have later ones reference back.
3. Chapter order must be a valid topological sort over `depends_on_chapters`.
4. Aim for 6-14 chapters total. Use appendices for: notation reference,
   environment setup, datasets, debugging tips.
5. **Detect superseded content**: if multiple videos cover the same topic
   and one is a later revision, prefer the later one's claims for the
   chapter and put the earlier ones in a "what we changed from the first
   pass" note (or just drop the older claims if the speaker explicitly
   replaced them).
6. `estimated_pages` is your subjective estimate of finished page length.
   Most chapters should be 8-25 pages.

Return JSON only. No prose, no markdown fence.
