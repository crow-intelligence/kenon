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

See the `docs/` directory for full API reference and examples.

## License

MIT
