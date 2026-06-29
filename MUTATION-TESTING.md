# Mutation testing summary

Tool: [`mutmut`](https://github.com/boxed/mutmut) 3.6 (dev-only; not run in CI).
Run with `KENON_MUTATION=1 uv run mutmut run` (the env var loads a Hypothesis
profile in `tests/conftest.py` that suppresses the `differing_executors` health
check and disables the example database during parallel mutation runs).

## Scope

Mutation is scoped to the **math-heavy core** and its fast, spaCy-free tests
(see `[tool.mutmut]` in `pyproject.toml`):

- mutated: `backbone.py`, `cooccurrence.py`, `graphs.py`
- test selection: `test_backbone.py`, `test_cooccurrence.py`, `test_graphs.py`

`tokenizer.py`, `embeddings.py`, and `stopwords.py` are **not** mutated — their
tests load spaCy / sklearn and are too slow to run per-mutant. This is a
deliberate cap, not full coverage.

## Score

| metric | value |
|--------|------:|
| total mutants | 442 |
| killed | 361 |
| killed by timeout | 1 |
| **survived** | **80** |
| **mutation score** | **~82%** (362 / 442) |

> Note on reproducibility: the new property-based tests run with the Hypothesis
> example database disabled during mutation (so parallel workers don't collide),
> which means a *different* random example set runs each time. As a result the
> exact survivor count drifts by ~±10 between runs in the property-tested
> functions (`detect_collocations`, `apply_disparity_filter`). The
> deterministic exact-value / exact-structure tests added below are stable kills.

## Gaps closed this pass

Tests added specifically to kill surviving mutants:

- **`disparity_integral` / `get_disparity_significance`** — exact-value unit
  tests pinning the formula (`disparity_integral(0.5, 3) == -0.125`,
  `get_disparity_significance(0.5, 3) == 0.25`). The range/monotonicity property
  tests left arithmetic-operator mutants alive; these kill them. The disparity
  formula now has **zero survivors**.
- **`build_cooccurrence_graph`** — an exact-structure test on `["a","b","c"]`
  (pins the skip-gram window arithmetic and the 0.5 normalised weights), an
  inclusive-`min_weight` test (`>=` vs `>`), and a default-`window`-is-2 test.
- **`apply_disparity_filter`** — the `norm_weight` value is now asserted in
  `[0, 1]` (previously only its key presence was checked, so `norm_weight = None`
  survived).

## Remaining survivors (80) — triage

Most survivors are **equivalent mutants** (no input can distinguish them) or
require contrived inputs for negligible value:

- **`cosine_similarity_matrix` (10)** — almost all equivalent. The
  `np.clip(sim, -1, 1)` is defensive: cosine values are already in `[-1, 1]`, so
  mutating the clip bounds changes nothing. sklearn assigns vocabulary indices
  alphabetically, so the `key=lambda w: vocab_dict[w]` sort mutating to `key=None`
  yields identical order. `.astype(None)` defaults to float64.
- **`apply_disparity_filter` (12) / `extract_backbone` (13)** — largely
  unreachable branches: `data.get("weight", <default>)` fallbacks never fire
  (edges always carry weights), and `strength <= 0` guards never fire (strengths
  are positive for connected nodes).
- **`build_semantic_graph` (24)** — concentrated in the optional `k_neighbors`
  path and threshold bookkeeping; many are behaviour-equivalent on realistic
  inputs. (This function is also flagged in `PRE-MORTEM.md` for a silent-no-op
  fragility.)
- **`build_cooccurrence_graph` (6)** — `total_pairs == 1` is unreachable
  (co-occurrence events are always counted symmetrically, so the total is even);
  the remaining window-arithmetic mutants produce identical output on the
  symmetric test input.
- **`detect_collocations` (11)** — mutations inside the NLTK metric dispatch;
  killed inconsistently by the random property examples (see reproducibility
  note).
- **`save_graph` / `load_graph` (2 each)** — format-string dispatch details.

## Finding surfaced by this pass

While hardening `detect_collocations`, the property tests revealed that it
**propagates NLTK exceptions on degenerate corpora**: `metric="chi_sq"` raises
`ZeroDivisionError` and `metric="likelihood"` raises a math-domain `ValueError`
on all-identical token input (e.g. `["a"] * 12`). Only `"pmi"` is robust. The
property test uses `"pmi"` to stay green; the crash is left unfixed (it changes
public behaviour) and recorded in `CHANGES_SUMMARY.md` for a human decision.
