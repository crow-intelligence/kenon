# Comparing two texts

Build a co-occurrence graph for each of two texts on different topics, then
quantify how much vocabulary they share. This is the building block for any
graph-to-graph comparison (including comparing a text-derived network against
human association norms — see the
[word-association tutorial](../tutorials/word_association_graph.md)).

## What it shows

1. Building co-occurrence graphs for two different texts
2. Computing the Jaccard similarity of their node sets
3. Identifying the vocabulary gap (terms unique to each text)

## Run it

```bash
python examples/comparing_two_texts.py
```

```text
Text A graph: 28 nodes, 55 edges
Text B graph: 30 nodes, 59 edges

Jaccard similarity of node sets: 0.018

Shared nodes (1): ['natural']
Only in Text A (27): ['algorithm', 'architecture', ...]
Only in Text B (29): ['biodiversity', 'cap', ...]

Vocabulary gap: 56 unique terms out of 57 total
```

## Source

```python title="examples/comparing_two_texts.py"
--8<-- "examples/comparing_two_texts.py"
```
