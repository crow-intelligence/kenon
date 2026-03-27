"""Co-occurrence graphs from tokenised text using skip-gram windows.

Builds weighted co-occurrence graphs where edge weight equals the normalised
co-occurrence frequency, and detects statistically significant collocations
using NLTK.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from kenon._types import SemanticGraph, Token


def build_cooccurrence_graph(
    tokens: list[Token],
    window: int = 2,
    stopwords: frozenset[str] | None = None,
    min_weight: float = 0.0,
) -> SemanticGraph:
    """Build a weighted co-occurrence graph using skip-gram windows.

    Each node is a token. Each edge weight is the relative co-occurrence
    frequency of the two tokens within the specified window.

    Args:
        tokens: Flat list of tokens (already lowercased / lemmatised as desired).
        window: Half-width of the skip-gram context window. A window of 2 means
            each token is paired with the 2 tokens before and 2 after.
        stopwords: Tokens to exclude from nodes and edges.
        min_weight: Drop edges with weight below this threshold.

    Returns:
        A ``networkx.Graph`` with ``weight`` edge attributes.

    Raises:
        ValueError: If ``window`` is less than 1.

    Contract:
        - No self-loops in the returned graph.
        - All edge weights are positive.
        - Stopword filtering happens before counting.

    Example:
        >>> tokens = ["cat", "sat", "mat", "cat", "mat"]
        >>> g = build_cooccurrence_graph(tokens, window=1)
        >>> g.has_node("cat")
        True
        >>> g["cat"]["sat"]["weight"] > 0
        True
    """
    if window < 1:
        msg = f"window must be >= 1, got {window}"
        raise ValueError(msg)

    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    pair_counts: Counter[tuple[str, str]] = Counter()
    total_pairs = 0

    for i, token_a in enumerate(tokens):
        start = max(0, i - window)
        end = min(len(tokens), i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            token_b = tokens[j]
            if token_a == token_b:
                continue
            pair = (min(token_a, token_b), max(token_a, token_b))
            pair_counts[pair] += 1
            total_pairs += 1

    graph: SemanticGraph = nx.Graph()

    if total_pairs == 0:
        return graph

    for (a, b), count in pair_counts.items():
        weight = count / total_pairs
        if weight >= min_weight:
            graph.add_edge(a, b, weight=weight)

    return graph


def detect_collocations(
    tokens: list[Token],
    n: int = 2,
    metric: str = "pmi",
    top_n: int = 20,
    min_freq: int = 2,
) -> list[tuple[str, ...]]:
    """Detect statistically significant n-grams using NLTK collocation finders.

    Args:
        tokens: Flat token list.
        n: N-gram size. Supports 2 (bigrams) and 3 (trigrams).
        metric: Scoring metric. One of ``"pmi"``, ``"chi_sq"``, ``"likelihood"``.
        top_n: Number of top collocations to return.
        min_freq: Minimum frequency filter applied before scoring.

    Returns:
        List of token tuples sorted by score descending.

    Raises:
        ValueError: If ``n`` is not 2 or 3.
        ValueError: If ``metric`` is not one of the supported values.

    Contract:
        - Returns at most ``top_n`` tuples.
        - Each tuple has length ``n``.

    Example:
        >>> tokens = ["new", "york", "city", "new", "york", "times"] * 10
        >>> colls = detect_collocations(tokens, n=2, top_n=5)
        >>> ("new", "york") in colls
        True
    """
    metric_map_bigram = {
        "pmi": BigramAssocMeasures.pmi,
        "chi_sq": BigramAssocMeasures.chi_sq,
        "likelihood": BigramAssocMeasures.likelihood_ratio,
    }
    metric_map_trigram = {
        "pmi": TrigramAssocMeasures.pmi,
        "chi_sq": TrigramAssocMeasures.chi_sq,
        "likelihood": TrigramAssocMeasures.likelihood_ratio,
    }

    if metric not in metric_map_bigram:
        msg = f"Unsupported metric '{metric}'. Supported: 'pmi', 'chi_sq', 'likelihood'"
        raise ValueError(msg)

    if n == 2:
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(metric_map_bigram[metric], top_n)

    if n == 3:
        finder = TrigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(metric_map_trigram[metric], top_n)

    msg = f"n must be 2 or 3, got {n}"
    raise ValueError(msg)
