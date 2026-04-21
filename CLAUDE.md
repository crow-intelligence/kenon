# kenon — project notes for Claude

Semantic and co-occurrence graph library for midsized texts. Corpus-internal statistics only — no neural models, no external training data. By Crow Intelligence (https://crowintelligence.org/).

## Tech stack

- **Python ≥3.11**, packaged and run with **uv** (uv_build backend)
- **ruff** for format + lint, **ty** (Astral's type checker) for typing — no mypy
- **pytest** + **hypothesis** for testing; `strategies.py` holds shared Hypothesis strategies
- **mkdocs-material** + **mkdocstrings** for docs, hosted on **ReadTheDocs**
- **PyPI trusted publishing** (OIDC) via `.github/workflows/publish.yml`

## Module layout (`src/kenon/`)

- `tokenizer.py` — spaCy-backed `Tokenizer` (sentences, tokens, lemmas)
- `stopwords.py` — merged NLTK + sklearn stopwords with extensions
- `embeddings.py` — `CountVectorizerEmbedder`, `TfidfEmbedder`, `PMIEmbedder` (via `chronowords`)
- `cooccurrence.py` — `build_cooccurrence_graph`, `detect_collocations` (NLTK)
- `graphs.py` — `build_semantic_graph`, `cosine_similarity_matrix`, `save_graph`/`load_graph` (graphml/gml/pickle)
- `backbone.py` — disparity filter (Serrano et al. 2009) for extracting statistically significant edges
- `_types.py` — type aliases (`Token`, `SemanticGraph`, `Matrix`, etc.)

## Dev commands

The `Makefile` wraps `uv run`:

- `make ci` — format + lint + typecheck + test
- `make test` — `pytest --doctest-modules --cov=kenon --cov-report=term-missing`
- `make format` / `make lint` — ruff format check / lint check
- `make typecheck` — `ty check src`

Docs:

- `uv run mkdocs serve` — live preview at :8000
- `uv run mkdocs build` — build to `site/` (gitignored; add if not yet)

## Local dev setup

The transitive dep `chronowords` compiles a Cython extension, so the Python used for the venv must have dev headers (`Python.h`). System Pythons on most Linux distros don't ship headers by default — you'll get `fatal error: Python.h: No such file or directory` during `uv sync`.

Two ways to satisfy this:

1. **Install OS dev headers** matching your system Python, e.g.:
   ```bash
   sudo apt install python3.12-dev   # Debian/Ubuntu; adjust the version
   ```

2. **Use a uv-managed Python** (ships with headers):
   ```bash
   uv python install 3.11
   uv sync --all-extras --python 3.11
   ```
   If `uv sync` still grabs the system interpreter (check `.venv/pyvenv.cfg` — `home` should point inside `~/.local/share/uv/python/...`, not `/usr/bin`), pass the path explicitly:
   ```bash
   uv sync --all-extras --python "$(uv python find 3.11)"
   ```

Then install the spaCy model used in tests:

```bash
uv pip install --python .venv \
  en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

If `uv run` re-resolves and keeps trying to build against the wrong Python, invoke tools directly from `.venv/bin/` (e.g. `.venv/bin/pytest`) to pin to the venv interpreter.

CI doesn't hit this — the Ubuntu runner image includes dev headers.

## Release process

1. Bump `version` in `pyproject.toml`
2. Add an entry at the top of `CHANGELOG.md`
3. Commit with subject `Release <version>`, push to `main`
4. Tag `v<version>`, push the tag
5. `gh release create v<version> --title v<version> --notes "..."` — triggers `.github/workflows/publish.yml` which builds with uv, publishes to PyPI via OIDC, and pings the ReadTheDocs webhook

## Conventions

- `ruff` config in `pyproject.toml`: line-length 88, Google docstring style, selects `E,F,I,N,UP,ANN,D`
- Tests skip `D100/D102/D103/D104/ANN` (see `[tool.ruff.lint.per-file-ignores]`)
- Docstrings follow Google style and include an `Example` block with doctests (run via `--doctest-modules`)
- Many public functions document a `Contract:` section — these are the invariants that property-based tests should cover
- Commit style: short imperative subject, no conventional-commits prefix (look at `git log` for examples)

## Logo

- `img/kenon.svg` — canonical logo. Text is dark (`#2C3E50`) to render on light backgrounds (GitHub/PyPI/mkdocs default).
- `docs/assets/kenon.svg` — copy used by mkdocs (`theme.logo` path is relative to `docs_dir`).
- README references the logo by absolute `raw.githubusercontent.com` URL so PyPI renders it.

## Known follow-ups

- GitHub issue: bump CI actions for Node.js 24 compat (deprecation warnings from `actions/checkout@v4`, `astral-sh/setup-uv@v4`)
