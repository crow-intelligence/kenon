# Changes summary — kenon hardening pass

A review queue, not merged changes. Three phase PRs (docs / design / tests) plus
this summary, all opened against `main`, **none merged**. `main` was never
touched. Triage at your leisure.

## The PRs

| PR | Branch | What it does | Risk |
|----|--------|--------------|------|
| [#2](https://github.com/crow-intelligence/kenon/pull/2) | `docs-overhaul` | Docs overhaul: fixes a package-wide example-rendering bug, rebuilds quickstart/tutorial/troubleshooting, enables strict docs builds | Low — docstring + docs only, no behaviour change |
| [#3](https://github.com/crow-intelligence/kenon/pull/3) | `design-review` | `PRE-MORTEM.md` fragility map + two safe type/exception tightenings | Low — annotation/comment only |
| [#4](https://github.com/crow-intelligence/kenon/pull/4) | `test-hardening` | Property-based tests on the math core + mutation testing (~82%) | Low — tests + dev tooling only |

All three are green: `make ci` passes (118 → 132 tests), and #2 builds docs with
`mkdocs build --strict` warning-free.

### Safe to merge as-is
- **#2 (docs)** — highest priority, this is what users complained about. The
  root cause was real: every `Example:` docstring section rendered as nested
  blockquotes (griffe only treats the plural `Examples:` as code), which also
  broke strict builds. Fixed package-wide. The quickstart and tutorial were
  executed in a clean venv before committing.
- **#3 (design)** — the only code changes are a stale `# type: ignore` removal
  and one annotation tightening. The substance is `PRE-MORTEM.md`.
- **#4 (tests)** — pure additions. Adds `mutmut` to the `dev` extra and a
  `[tool.mutmut]` config; mutation is dev-only and **not** wired into CI.

## Needs your decision (behaviour-changing — deliberately not applied)

From `PRE-MORTEM.md` (full detail there) and the test pass:

1. **Degenerate word similarity on few-document corpora** *(highest impact)* —
   `cosine_similarity_matrix` builds word vectors whose dimensionality equals the
   document count, so `build_semantic_graph` over a short corpus returns an
   artifactually dense graph. Decide: warn below a minimum doc count, document
   the limitation, or steer small-corpus users to `build_cooccurrence_graph`.
2. **sklearn embedders densify** (`.toarray()`) — OOM risk on large corpora,
   contradicting the memory-efficient positioning. Decide: keep sparse / cap
   vocab / document.
3. **`transform()` before `fit()`** raises sklearn's `NotFittedError`, not the
   documented `RuntimeError`. One-line guard would fix it, but it changes the
   observed exception type.
4. **`build_semantic_graph(k_neighbors=...)`** re-fits the embedder and silently
   degrades to a threshold graph if a shape check fails. Decide: reuse the fitted
   embedder; warn/raise instead of silently degrading.
5. **`detect_collocations` propagates NLTK crashes on degenerate corpora**
   *(surfaced by the new property tests)* — `metric="chi_sq"` raises
   `ZeroDivisionError` and `metric="likelihood"` raises a math-domain
   `ValueError` on all-identical tokens (e.g. `["a"] * 12`). Only `"pmi"` is
   robust. Decide: guard and raise a kenon-level error, or document the
   constraint. (The property test sidesteps it by using `pmi`.)
6. **`get_stopwords` runtime NLTK download + unwrapped language error** — fails
   at runtime offline, and an unsupported `lang` surfaces a raw NLTK error
   instead of a helpful one like the spaCy path. Decide: wrap the error.

Lower priority (also in `PRE-MORTEM.md`): unguarded `disparity_integral` public
helper, in-place mutation in `apply_disparity_filter`, pickle-load RCE surface,
percentile tie-bias.

## Proposed features (not built — your call)

From the kenon tutorial brief in the runbook:

- **`kenon.paths` helper** — the docs tutorial demonstrates concept-to-concept
  pathfinding using networkx directly on the `SemanticGraph` (no new API). If you
  want this as first-class kenon surface, a small module wrapping
  `nx.shortest_path` / `nx.all_simple_paths` with co-occurrence-weight→distance
  inversion and weight-threshold filtering would be a clean, well-scoped addition.
- **Human-norm comparison (Nelson norms / Small World of Words)** — comparing a
  text-derived network against human free-association norms needs external,
  licensed datasets and a graph-comparison helper. Too large/risky for an
  unattended pass; the tutorial describes the methodology conceptually. Worth a
  dedicated feature branch if you want it.

## What I deliberately did not touch, and why

- **`main`** — never checked out, committed to, or merged into. Untouched.
- **The Honnibal skills** — the runbook's Part A (download → review the raw
  `.md.txt` → activate) is the one step it says a human must not skip under
  bypass mode, because a skill runs with full shell/git access. Only
  `hypothesis.md` was installed. Rather than auto-download and activate
  third-party skills unattended, I replicated each skill's *intent* by hand. To
  use the genuine commands, run Part A yourself and re-run.
- **`uv.lock` version drift** — the repo had a pre-existing stale lock
  (`0.1.0` vs `pyproject`'s `0.1.1`). I left it alone on the docs and design
  branches; #4 regenerates the lock (for the `mutmut` dev dep), which also
  corrects that drift.
- **Any behaviour-changing fix** — per the guardrails, everything that would
  alter the public API or runtime behaviour is written up above rather than
  applied.

## Artifacts

- `PRE-MORTEM.md` (on `design-review`) — the fragility map.
- `MUTATION-TESTING.md` (on `test-hardening`) — score (~82%, 362/442) + survivor
  triage.
- This file.
