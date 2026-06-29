# Changelog

## 0.1.2

- Documentation overhaul: runnable end-to-end quickstart, a word-association graph tutorial (concept-to-concept pathfinding), a troubleshooting page, and example pages with embedded source; enable strict mkdocs builds
- Fix a package-wide docstring rendering bug — Google `Example:` sections (singular) rendered as blockquotes and broke strict builds; renamed to `Examples:` so they render as highlighted code
- Promote the `load_graph` pickle-safety caveat to a prominent warning; complete the API reference (add `get_stopwords`)
- Expand the test suite with property-based tests for the disparity-filter math, co-occurrence weight normalisation, collocation invariants, and the cosine-similarity contract; add exact-value tests pinning the disparity formula
- Add mutation testing (mutmut, dev-only; ~82% score) and `PRE-MORTEM.md` / `MUTATION-TESTING.md` engineering notes
- Safe internal tightening: remove a stale type-ignore, narrow a numpy annotation
- Add a Roadmap to the README for the next iteration

## 0.1.1

- Add project logo to README, mkdocs site, and PyPI page
- Update Crow Intelligence URL to crowintelligence.org across README, docs, and PyPI metadata
- Expand Hypothesis property-based tests (graph save/load roundtrip, disparity significance bounds, embedder fit_transform consistency, backbone subset invariants, cooccurrence symmetry)

## 0.1.0

- Initial release
- Tokenizer with spaCy backend (sentence splitting, lemmatization)
- Unified stopword lists (NLTK + sklearn)
- Corpus-internal embeddings: CountVectorizer, TF-IDF, PMI
- Co-occurrence graph construction with skip-gram windows
- Collocation detection via NLTK (PMI, chi-squared, likelihood ratio)
- Semantic similarity graphs from embeddings
- Disparity filter backbone extraction
- Graph persistence (GraphML, GML, pickle)
