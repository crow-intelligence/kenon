<p align="center">
  <img src="https://raw.githubusercontent.com/crow-intelligence/kenon/main/img/kenon.svg" alt="kenon logo" width="400">
</p>

# kenon

Semantic and co-occurrence graphs for midsized texts. Kenon builds weighted
graphs from text using corpus-internal statistics — no neural models or external
training data required. Supports co-occurrence windows, TF-IDF similarity,
PMI embeddings, and disparity filter backbone extraction.

## Installation

```bash
uv add kenon
python -m spacy download en_core_web_sm
```

## Quickstart

```python
from kenon import (
    Tokenizer,
    get_stopwords,
    build_cooccurrence_graph,
    extract_backbone,
)

# 1. Tokenize
tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
tokens = tokenizer.flat_tokens("The cat sat on the mat. The dog ran in the park.")

# 2. Build graph
stopwords = get_stopwords("english")
graph = build_cooccurrence_graph(tokens, window=2, stopwords=stopwords)

# 3. Extract backbone
backbone = extract_backbone(graph, min_alpha_ptile=0.3, min_degree=2)
print(f"Backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")
```

## Features

- **Tokenization**: spaCy-backed sentence splitting, tokenization, and lemmatization
- **Stopwords**: Merged NLTK + sklearn stopword lists with custom extensions
- **Embeddings**: Count vectors, TF-IDF, and PMI (via chronowords) — all corpus-internal
- **Co-occurrence graphs**: Skip-gram window co-occurrence with collocation detection
- **Semantic graphs**: Cosine similarity graphs from any embedder
- **Backbone extraction**: Disparity filter for statistically significant edges

## Documentation

Full documentation — quickstart, the word-association tutorial, troubleshooting,
and the complete API reference — is at
[kenon.readthedocs.io](https://kenon.readthedocs.io). The sources live in `docs/`.

## Roadmap

Planned for the next iteration. The robustness items are analysed in detail in
[`PRE-MORTEM.md`](PRE-MORTEM.md).

**Robustness / API decisions**

- [ ] Warn or document when `build_semantic_graph` runs on too few documents — word-similarity is degenerate on small corpora (each word vector's dimension is the document count).
- [ ] Avoid the dense `.toarray()` materialisation in the sklearn embedders (out-of-memory risk on large corpora).
- [ ] Make `transform()` raise the documented `RuntimeError` when called before `fit()` (it currently surfaces sklearn's `NotFittedError`).
- [ ] `build_semantic_graph(k_neighbors=...)`: reuse the already-fitted embedder and warn instead of silently degrading to a threshold-only graph.
- [ ] Guard `detect_collocations` against NLTK crashes on degenerate corpora — `chi_sq` raises `ZeroDivisionError` and `likelihood` a math-domain `ValueError` on all-identical tokens; only `pmi` is robust.
- [ ] Wrap the unsupported-language error in `get_stopwords` and document its one-time NLTK download.

**Proposed features**

- [ ] `kenon.paths` — a first-class concept-to-concept pathfinding helper (currently demonstrated via networkx in the tutorial).
- [ ] Compare a text-derived network against human association norms — [Nelson norms](http://w3.usf.edu/FreeAssociation/) / [Small World of Words](https://smallworldofwords.org/) (needs external datasets).

**Maintenance**

- [ ] Refresh dependencies to current releases (chronowords 0.3.x, numpy 2.5, scipy 1.18, scikit-learn 1.9, pandas 3.0.3, spaCy 3.8.14). The chronowords bump is API-compatible; it's blocked on **chronowords adding Python 3.13 support** so a fresh `pip install kenon` resolves on 3.13 (chronowords currently caps `requires-python` at `<3.13`).

## Made by

Kenon is made by [Crow Intelligence](https://crowintelligence.org/).

## License

MIT
