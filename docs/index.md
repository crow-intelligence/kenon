<p align="center">
  <img src="assets/kenon.svg" alt="kenon logo" width="400">
</p>

# kenon

Semantic and co-occurrence graphs for midsized texts.

Kenon builds weighted graphs from text using corpus-internal statistics only.
No neural models or external training data required.

## Core concepts

- **Co-occurrence graphs**: Tokens as nodes, skip-gram window co-occurrence as edges
- **Semantic graphs**: Tokens as nodes, cosine similarity from embeddings as edges
- **Backbone extraction**: Disparity filter to keep only statistically significant edges

## Quick links

- [Quickstart](quickstart.md) — clean install to backbone graph
- [Tutorial: word-association graphs](tutorials/word_association_graph.md) — find paths between concepts
- [Examples](examples/news_article_analysis.md) — full, runnable scripts
- [Troubleshooting](troubleshooting.md) — common errors and fixes
- [API Reference](api/tokenizer.md) — every public function and its contract

---

Made by [Crow Intelligence](https://crowintelligence.org/)
