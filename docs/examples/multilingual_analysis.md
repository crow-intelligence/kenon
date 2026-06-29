# Multilingual analysis

kenon is language-agnostic: tokenisation is driven by whichever spaCy model you
load, and stopwords can come from NLTK (many languages) or scikit-learn (English
only). This example processes English and German text and compares graph density.

## What it shows

1. Tokenising English and German texts with different spaCy models
2. Selecting stopword sources per language (`sources=["nltk"]` for German)
3. Building co-occurrence graphs for each and comparing density

## Prerequisites

The German half needs the German spaCy model. If it is missing, the script
prints a clear message and skips that section rather than crashing:

```bash
python -m spacy download de_core_news_sm
```

## Run it

```bash
python examples/multilingual_analysis.py
```

```text
Multilingual Co-occurrence Graph Analysis
==================================================

--- English ---
  Tokens: 37, After filtering: 23
  Graph: 22 nodes, 43 edges
  Density: 0.1861
  Top 5 nodes by degree: [('growth', 8), ('finding', 4), ('prestigious', 4), ('scientific', 4), ('journal', 4)]

German analysis skipped: spaCy model 'de_core_news_sm' is not installed.
```

## Source

```python title="examples/multilingual_analysis.py"
--8<-- "examples/multilingual_analysis.py"
```
