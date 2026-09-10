# Plain-language guide for PR digests

Use this guide before writing or revising a digest. The reader is an engineer who may not know the
PR's subsystem. Accuracy still comes first, but unfamiliar vocabulary must not carry the
explanation by itself.

## Translate the mechanism without hiding it

Start with what the code does in familiar words. Put an important source-code term after the
explanation so readers can still search for it.

| Avoid | Prefer |
|---|---|
| "decode-side row selection" | "choose which rows to keep while the compressed data is being decoded" |
| "positional mask" | "a bitmap that marks which original row positions remain visible (the positional mask)" |
| "materialize only rows satisfying both" | "create output rows only when both checks keep them" |
| "confirmed decoder consumption" | "the decoder reports that it already applied the visibility bitmap" |
| "compose visibility into the decode selection" | "apply row visibility together with the query filter during decoding" |
| "writer-free covering snapshot" | "a read-only snapshot that includes every commit present when the bitmap was built" |
| "cache probe/build/publish loop" | "look for saved visibility information, build it when missing, then save a safe result" |
| "admit the zero-copy transfer" | "allow Sirius to reuse the decoded table directly instead of copying it" |
| "key admission" | "decide which join keys can safely use the optimization" |
| "fallback contract" | "the rules for using the older path when the optimization cannot run" |

These are patterns, not replacements to apply blindly. Use the meaning established by the PR's
actual code.

## Introduce abbreviations and identifiers

- On first use, write "multi-version concurrency control (MVCC), which determines which rows a
  query is allowed to see." Adjust the explanation to the PR's actual concern.
- Introduce an identifier by purpose: "The function that checks whether saved visibility data can
  be reused, `mvcc_mask_cache_reusable`, ..."
- Keep identifiers out of the opening summary and diagram unless the identifier itself is the
  user-facing concept.
- Do not repeat both the plain phrase and technical term in every sentence. Define the term once,
  then use whichever version makes the next sentence easiest to understand.

## Make cause and consequence explicit

Prefer a short chain the reader can follow:

1. What condition starts this path?
2. What does the old code do?
3. What does the new code do instead?
4. What work, behavior, or safety property changes as a result?
5. What happens when the new path cannot run?

Avoid sentences that only rename code. "The request gains `keep_mask_words`" is incomplete. Add
the consequence: those words let the decoder exclude invisible rows at the same time as it applies
the query filter.

## Final jargon check

Read the title, summary, before/after table, diagram, key-change titles, and first paragraph of each
detailed section without looking at the code. Revise them if any of these are true:

- a term names an implementation technique but does not say what it does;
- an acronym appears before its meaning;
- a heading is mostly class names, field names, or stacked nouns;
- verbs such as "compose," "materialize," "consume," "admit," or "dispatch" could be replaced by a
  clearer action;
- "fast path" or "fallback" appears without explaining the optimized or older behavior;
- the reader must infer why the change matters from the diff.

Keep technical terms inside quoted PR text and code diffs unchanged. Explain them immediately
before or after the quote when they are necessary to understand it.
