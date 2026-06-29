# kenon — pre-mortem

A fragility map: *assume a future incident has happened — what most likely caused
it?* This is a design-review artifact, not a bug list. Each entry names a failure
mode, how it would surface, and a suggested direction. Nothing here is a
confirmed defect; items marked **[needs human decision]** would change public
behaviour and are deliberately **not** fixed in this pass.

Scope: read-through of `src/kenon/` at the `design-review` branch point. The
codebase is in good shape — strong docstrings with explicit `Contract:` sections,
no bare `except`, all exception handlers already specific, clean `ty`/`ruff`
(ANN/D). The fragilities below are mostly *semantic* and *operational*, not
mechanical.

---

## Applied in this pass (safe, behaviour-preserving)

- Removed a stale `# type: ignore[name-defined]` on `Tokenizer._token_text`
  (mypy-era leftover; `ty` passes without it).
- Tightened `knn_mask: np.ndarray` → `NDArray[np.bool_]` in
  `build_semantic_graph` (annotation only).

Nothing else was safe to change mechanically — the exception handling is already
specific and the types already pass `ty`. The remaining value of this review is
the map below.

---

## High-impact fragilities

### 1. Word similarity from few documents is degenerate  *(most important)*
`cosine_similarity_matrix` builds a *word* vector by transposing the
*document × term* matrix, so each word vector's dimensionality equals the number
of **documents**. On a short corpus (a handful of sentences) word vectors live in
a 2–5 dimensional space and cosine similarities collapse toward 0/1 — many word
pairs come out *exactly* 1.0.

- **Surfaces as:** `build_semantic_graph` returning a misleadingly dense, almost
  fully-connected graph on small inputs. The `news_article_analysis` example
  produces **555 edges among 81 nodes** from ~12 sentences — that density is an
  artifact, not signal.
- **Why it bites:** users treat the semantic graph as meaningful when it is
  mostly noise; downstream centrality/backbone results inherit the artifact.
- **[needs human decision]** Options: require a minimum document count and warn
  below it; document that `build_semantic_graph` needs many documents to be
  meaningful; or steer small-corpus users toward `build_cooccurrence_graph`
  (which does not have this problem). At minimum, a docstring warning.

### 2. sklearn embedders densify the term matrix  *(memory)*
`CountVectorizerEmbedder` / `TfidfEmbedder` call `.toarray()` in both `transform`
and `fit_transform`, materialising a dense `(n_docs × n_vocab)` float64 array.

- **Surfaces as:** `MemoryError` on larger corpora — directly contradicting the
  library's "memory-efficient" positioning (the reason `PMIEmbedder`/chronowords
  exists).
- **[needs human decision]** Keep the sparse matrix and densify lazily, or cap
  vocabulary, or document the dense-materialisation cost. Changing the return
  type to sparse would alter the public contract (`Matrix` is dense `float64`).

### 3. `transform()` before `fit()` raises the *wrong* error
The documented contract is "fit before transform," and `vocabulary` enforces it
with a clear `RuntimeError`. But `transform()` does **not** guard — it calls into
the unfitted sklearn vectorizer and raises sklearn's `NotFittedError`, an
undocumented, inconsistent error type.

- **Surfaces as:** users catching `RuntimeError` (per the contract) miss the real
  failure; the error message points at sklearn internals.
- **[needs human decision]** Add a `_fitted` guard to `transform` raising the
  documented `RuntimeError`. This changes the observed exception type, hence not
  applied here.

### 4. `build_semantic_graph(k_neighbors=...)` re-fits and can silently no-op
When `k_neighbors` is set, the function calls `embedder.fit(corpus)` a **second**
time (it was already fit inside `cosine_similarity_matrix`), then builds the kNN
mask only if `word_vectors.shape[0] == len(vocab)`. If that check fails, the kNN
masking is **silently skipped** and the result falls back to threshold-only.

- **Surfaces as:** (a) wasted work / double-fit side effects on stateful
  embedders; (b) a user who asked for a kNN graph silently getting a
  threshold graph, with no warning.
- **[needs human decision]** Reuse the already-fitted embedder; and either raise
  or `log.warning` when the kNN mask can't be built instead of silently
  degrading.

---

## Operational / lower-likelihood fragilities

### 5. `get_stopwords` does network I/O at runtime
First call triggers `nltk.download("stopwords")` (guarded by a module global
`_NLTK_DATA_DOWNLOADED`). Two issues:
- Offline / firewalled environments fail at *runtime*, not install time.
- An unsupported `lang` reaches `nltk.corpus.stopwords.words(lang)` and raises a
  raw NLTK `OSError`/`LookupError` — unlike the spaCy path, which wraps the error
  with a helpful message.
- **[needs human decision]** Wrap the unsupported-language error with guidance;
  document the one-time network dependency (now noted in Troubleshooting).

### 6. `disparity_integral` is unguarded against its own preconditions
The docstring says `x != 1.0` and `k != 1.0`, but the function does not enforce
them — `x=1.0` or `k=1.0` yields `inf`/`nan` rather than a clear error. In the
internal call path this is safe (a node of degree > 1 always has `norm_weight < 1`,
and `get_disparity_significance` guards `degree <= 1`), so it only bites callers
who use the public helper directly. Low likelihood; worth either a guard or a
"this is an internal helper" note.

### 7. `apply_disparity_filter` mutates its input in place
Documented, and `extract_backbone` deep-copies before calling it — but invoking
`apply_disparity_filter` directly rewrites the caller's edge/node attributes. A
footgun for anyone composing it themselves. Consider a `copy: bool = False` knob
or a louder docstring note.

### 8. `load_graph(fmt="pickle")` executes arbitrary code
`pickle.load` on an untrusted file is RCE (the `# noqa: S301` acknowledges it).
Documented and now flagged with a `Warning` admonition in the docs. Consider
refusing pickle unless an explicit `trust=True` is passed.

### 9. `alpha_ptile` percentile is a lower bound under ties
`searchsorted(sorted_alphas, alpha, "left") / N` gives the fraction *strictly*
less than `alpha`. On tie-heavy small graphs many edges share an alpha, so the
percentile (and therefore `extract_backbone`'s `min_alpha_ptile` cut) is biased
low. Minor; affects only small/degenerate graphs.

### 10. Chunk boundaries can split mid-token
`_chunk_text` hard-splits at `_CHUNK_TARGET` (200K chars) when no separator is
found within the window. For pathological inputs (very long whitespace-free
runs) a token straddling the boundary is split in two. Extremely unlikely on
natural text; noted for completeness.

---

## Summary for triage

| # | Fragility | Likelihood | Impact | Disposition |
|---|-----------|-----------|--------|-------------|
| 1 | Degenerate word similarity on few docs | High on small corpora | High (silent wrong results) | **needs human decision** |
| 2 | Dense embedder matrices | Medium on large corpora | High (OOM) | **needs human decision** |
| 3 | `transform` pre-fit raises wrong error | Low | Low | **needs human decision** |
| 4 | kNN re-fit / silent no-op | Medium | Medium | **needs human decision** |
| 5 | Runtime NLTK download + unwrapped lang error | Medium | Medium | **needs human decision** |
| 6 | `disparity_integral` unguarded | Low | Low | optional guard |
| 7 | In-place mutation footgun | Low | Low | optional knob |
| 8 | pickle RCE | Low | High if shared | documented; optional `trust` flag |
| 9 | percentile tie bias | Low | Low | optional |
| 10 | chunk mid-token split | Very low | Low | noted |
