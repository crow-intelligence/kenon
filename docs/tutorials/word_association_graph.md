# Tutorial: word-association graphs

A co-occurrence graph built from text is a kind of **word-association network**:
words that habitually appear together become connected. This tutorial builds one
from a small corpus, then walks the network to answer a concrete question:

> How does the corpus "get from" one concept to another?

Everything here uses the graph directly via [networkx](https://networkx.org/) —
a kenon graph *is* a `networkx.Graph`, so the entire networkx toolkit applies.

## 1. Build the association graph

```python
import networkx as nx
from kenon import Tokenizer, get_stopwords, build_cooccurrence_graph

corpus = """
Coffee contains caffeine, a stimulant that increases alertness and focus.
Caffeine blocks adenosine receptors in the brain, reducing the feeling of fatigue.
Many students drink coffee to stay awake while studying for exams.
Sleep deprivation harms memory and concentration, so studying tired is ineffective.
A good night of sleep consolidates memory and restores focus for the next day.
"""

tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
tokens = tokenizer.flat_tokens(corpus)
stopwords = get_stopwords("english")

graph = build_cooccurrence_graph(tokens, window=4, stopwords=stopwords)
print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

```text
graph: 32 nodes, 135 edges
```

## 2. Turn association strength into distance

Co-occurrence edge weights are **association strength** — a *higher* weight means
two words are *more* closely associated. Shortest-path algorithms, on the other
hand, minimise total *distance*. Invert the weight so that strong associations
become short hops:

```python
for _u, _v, data in graph.edges(data=True):
    data["distance"] = 1.0 / data["weight"]
```

!!! warning "Don't pass raw co-occurrence weight to `shortest_path`"
    Using `weight="weight"` would treat the *strongest* associations as the
    *longest* detours. Always convert to a distance first.

## 3. Find the path between two concepts

```python
src, dst = "coffee", "memory"

path = nx.shortest_path(graph, src, dst, weight="distance")
print("shortest association path:", " -> ".join(path))

print("a few alternative paths (<= 4 hops):")
for p in list(nx.all_simple_paths(graph, src, dst, cutoff=4))[:3]:
    print("  ", " -> ".join(p))
```

```text
shortest association path: coffee -> exam -> memory
a few alternative paths (<= 4 hops):
   coffee -> contain -> caffeine -> focus -> memory
   coffee -> contain -> stimulant -> focus -> memory
   coffee -> contain -> increase -> focus -> memory
```

The shortest path is the corpus's most direct line of reasoning from `coffee` to
`memory`; the simple paths expose the alternative routes (via `caffeine` and
`focus`). On a denser corpus, run `extract_backbone` first to strip noise edges
before pathfinding, so the paths follow only statistically significant links.

!!! note "Missing nodes raise `NodeNotFound`"
    `nx.shortest_path` raises if a concept was filtered out as a stopword, never
    occurred, or fell below `min_weight`. Guard with `src in graph and dst in graph`,
    and check `nx.has_path(graph, src, dst)` before asking for a path.

## 4. Comparing against human association norms

A natural next question is whether a *text-derived* association network matches
how *people* associate words — as captured by human free-association norms such
as the [Nelson norms](http://w3.usf.edu/FreeAssociation/) or the
[Small World of Words](https://smallworldofwords.org/) (SWOW) project. The
comparison is conceptually straightforward: build the kenon graph, load the norm
graph, align the shared vocabulary, and compare neighbourhoods (e.g. rank
correlation of edge weights, or overlap of each word's top associates).

kenon does not yet ship a loader for these external datasets or a built-in graph
comparison helper — both are tracked as proposed features (see `CHANGES_SUMMARY.md`
in the repository). For now you can compare two graphs with networkx directly, as
shown in the [Comparing two texts](../examples/comparing_two_texts.md) example.
