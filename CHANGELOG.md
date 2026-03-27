# Changelog

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
