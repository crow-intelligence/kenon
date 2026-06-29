# News article analysis

A complete walk-through of the core kenon workflow on a single news article:
tokenise → remove stopwords → build a co-occurrence graph → extract the backbone
→ rank words by centrality → compare two semantic-graph embedders.

## What it shows

1. Tokenising a news article with lemmatisation
2. Removing stopwords
3. Building a co-occurrence graph (`window=3`)
4. Extracting the backbone with the disparity filter
5. Finding the top nodes by degree centrality
6. Building semantic graphs with `CountVectorizerEmbedder` and `TfidfEmbedder`

## Run it

```bash
python examples/news_article_analysis.py
```

```text
Total tokens: 158
Tokens after stopword removal: 95

Co-occurrence graph: 78 nodes, 274 edges
Backbone: 4 nodes, 6 edges

Top 10 nodes by degree centrality (backbone):
  growth: 1.000
  sluggish: 1.000
  recent: 1.000
  quarter: 1.000

--- CountVectorizer Semantic Graph ---
Nodes: 81, Edges: 555

--- TF-IDF Semantic Graph ---
Nodes: 81, Edges: 555
```

## Source

```python title="examples/news_article_analysis.py"
--8<-- "examples/news_article_analysis.py"
```
