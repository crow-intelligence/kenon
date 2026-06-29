# Troubleshooting

Common errors and how to resolve them. Each entry shows the failure, why it
happens, and the fix.

## `RuntimeError: spaCy model '...' is not installed`

```text
RuntimeError: spaCy model 'en_core_web_sm' is not installed.
Run: python -m spacy download en_core_web_sm
```

`Tokenizer` loads its spaCy model lazily on first use, so this surfaces the
first time you call `flat_tokens`, `tokenize`, or `sentencize` — not at
construction time. Install the model named in the message:

```bash
python -m spacy download en_core_web_sm
```

For other languages, pass the model name to `Tokenizer(...)` and download the
matching model (e.g. `de_core_news_sm` for German).

## `fatal error: Python.h: No such file or directory`

This appears while installing kenon, during the build of its transitive
dependency [`chronowords`](https://pypi.org/project/chronowords/), which
compiles a Cython extension and needs the Python development headers. System
Pythons on most Linux distros don't ship them by default.

Either install the headers for your system Python:

```bash
sudo apt install python3.12-dev   # Debian/Ubuntu; match your version
```

…or use a uv-managed Python, which bundles them:

```bash
uv python install 3.11
uv sync --all-extras --python 3.11
```

## `ImportError: chronowords is required for PMIEmbedder`

`PMIEmbedder` depends on `chronowords` (it ships as a core dependency, but can be
absent in a stripped environment). Install it:

```bash
uv add chronowords
```

`CountVectorizerEmbedder` and `TfidfEmbedder` have no such dependency — use them
if you only need count- or TF-IDF-based embeddings.

## `ValueError: sklearn stopwords are only available for English`

```python
get_stopwords("german")  # ValueError
```

scikit-learn ships an English-only stopword list. For other languages, restrict
the sources to NLTK:

```python
get_stopwords("german", sources=["nltk"])
```

## `ValueError: Unsupported stopword sources`

`get_stopwords(..., sources=[...])` accepts only `"nltk"` and `"sklearn"`. Any
other name (typo, wrong list) raises. Pass one or both of the supported names.

## An empty or tiny backbone

`extract_backbone` returns an **empty graph** when the input has no edges, and a
small backbone is expected for short inputs — the disparity filter is designed to
discard everything that isn't statistically significant. If you get fewer nodes
than you want:

- Lower `min_alpha_ptile` (e.g. `0.3`) to keep more edges.
- Lower `min_degree` to `1` so weakly connected nodes survive.
- Use a larger corpus — the filter needs enough edges per node to judge
  significance.

## `NodeNotFound` when finding paths

`networkx.shortest_path(graph, src, dst)` raises `NodeNotFound` if either word is
absent from the graph — usually because it was removed as a stopword, never
occurred, or fell below `min_weight`. Guard before querying:

```python
if src in graph and dst in graph and nx.has_path(graph, src, dst):
    path = nx.shortest_path(graph, src, dst, weight="distance")
```

## Loading a graph fails or behaves oddly

`save_graph` / `load_graph` must use the **same `fmt`** on both ends. `graphml`
and `gml` round-trip attributes safely and are human-readable;
`pickle` is fastest but **executes arbitrary code on load** — never load a
pickle file from an untrusted source. Prefer `graphml` for anything you share.
