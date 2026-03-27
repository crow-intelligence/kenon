# Quickstart

## Installation

```bash
uv add kenon
python -m spacy download en_core_web_sm
```

## Basic usage

```python
from kenon import Tokenizer, get_stopwords, build_cooccurrence_graph, extract_backbone

# Tokenize
tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
tokens = tokenizer.flat_tokens("Your text here.")

# Build graph
stopwords = get_stopwords("english")
graph = build_cooccurrence_graph(tokens, window=2, stopwords=stopwords)

# Extract backbone
backbone = extract_backbone(graph, min_alpha_ptile=0.3, min_degree=2)
```
