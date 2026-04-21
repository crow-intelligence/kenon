# Changelog

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
