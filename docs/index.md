# kenon

Semantic and co-occurrence graphs for midsized texts.

Kenon builds weighted graphs from text using corpus-internal statistics only.
No neural models or external training data required.

## Core concepts

- **Co-occurrence graphs**: Tokens as nodes, skip-gram window co-occurrence as edges
- **Semantic graphs**: Tokens as nodes, cosine similarity from embeddings as edges
- **Backbone extraction**: Disparity filter to keep only statistically significant edges

## Quick links

- [Quickstart](quickstart.md)
- [API Reference](api/tokenizer.md)
- [Examples](examples/news_article_analysis.md)

---

Made by [Crow Intelligence](https://crow-intelligence.github.io/)
