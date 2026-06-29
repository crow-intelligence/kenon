# Quickstart

This page takes you from a clean install to a backbone graph in one sitting.
Every snippet below is runnable as-is.

## Installation

```bash
uv add kenon
python -m spacy download en_core_web_sm
```

kenon uses a spaCy model for tokenisation. `en_core_web_sm` is the small English
model used throughout the docs; the download step is required the first time.

!!! note "Building from source"
    kenon depends on [`chronowords`](https://pypi.org/project/chronowords/),
    which compiles a Cython extension. If `uv add` fails with
    `fatal error: Python.h: No such file or directory`, see the
    [Troubleshooting](troubleshooting.md) page.

## Build a co-occurrence graph

A co-occurrence graph connects tokens that appear near each other within a
sliding window. Nodes are tokens; edge weights are normalised co-occurrence
frequencies.

```python
from kenon import Tokenizer, get_stopwords, build_cooccurrence_graph, extract_backbone
import networkx as nx

text = (
    "The central bank held interest rates steady on Thursday. "
    "Inflation in the euro zone was declining but remained too high. "
    "Financial markets reacted positively as bond yields fell and stocks rose. "
    "Analysts said the rate-hiking cycle had likely peaked."
)

# 1. Tokenise (lemmatise so "rates"/"rate" collapse to one node)
tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
tokens = tokenizer.flat_tokens(text)

# 2. Build the co-occurrence graph, dropping stopwords first
stopwords = get_stopwords("english")
graph = build_cooccurrence_graph(tokens, window=3, stopwords=stopwords)
print(f"co-occurrence graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

## Extract the backbone

Real co-occurrence graphs are dense and noisy. The **disparity filter**
([Serrano et al. 2009](https://arxiv.org/pdf/0904.2389.pdf)) keeps only edges
that are statistically significant relative to each node's other connections.

```python
# 3. Keep the statistically significant edges
backbone = extract_backbone(graph, min_alpha_ptile=0.3, min_degree=1)
print(f"backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")

# 4. A backbone graph is a plain networkx.Graph — use the whole networkx toolkit
top = sorted(nx.degree_centrality(backbone).items(), key=lambda kv: kv[1], reverse=True)[:5]
for word, score in top:
    print(f"  {word}: {score:.2f}")
```

Running the two snippets above prints something like:

```text
co-occurrence graph: 27 nodes, 78 edges
backbone: 6 nodes, 6 edges
  bank: 0.40
  central: 0.40
  hold: 0.40
  cycle: 0.40
  likely: 0.40
```

!!! tip "`min_alpha_ptile` controls aggressiveness"
    Higher values prune more edges. On short texts like this one, start low
    (`0.3`) and `min_degree=1`; on book-length corpora, raise both.

## Where to next

- [Word-association graphs tutorial](tutorials/word_association_graph.md) — find
  paths between two concepts.
- [Examples](examples/news_article_analysis.md) — full, runnable scripts.
- [API Reference](api/tokenizer.md) — every public function and its contract.
