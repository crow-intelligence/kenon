"""Semantic similarity graphs from embedding matrices.

Builds k-nearest-neighbour graphs and cosine similarity graphs where nodes
are vocabulary tokens and edges represent semantic similarity.
"""

from __future__ import annotations

import os
import pickle as pickle_mod
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.neighbors import NearestNeighbors

from kenon._types import Matrix, SemanticGraph, Token

if TYPE_CHECKING:
    from kenon.embeddings import EmbedderProtocol


def build_semantic_graph(
    embedder: EmbedderProtocol,
    corpus: list[str],
    similarity_threshold: float = 0.4,
    k_neighbors: int | None = None,
    stopwords: frozenset[str] | None = None,
) -> SemanticGraph:
    """Build a semantic similarity graph from corpus-internal embeddings.

    Nodes are vocabulary tokens. An edge (u, v) exists when the cosine
    similarity between u and v exceeds ``similarity_threshold``. Edge weight
    is the cosine similarity value.

    If ``k_neighbors`` is set, a k-NN graph is used to restrict connectivity
    before applying the threshold.

    Args:
        embedder: A fitted or unfitted embedder implementing ``EmbedderProtocol``.
            If unfitted, it will be fitted on ``corpus``.
        corpus: List of document strings. Used both to fit the embedder and
            to determine the vocabulary.
        similarity_threshold: Minimum cosine similarity for an edge. Must be
            in [0, 1].
        k_neighbors: If not ``None``, restrict each node to its k nearest
            neighbours before threshold filtering.
        stopwords: Tokens to exclude from the graph nodes.

    Returns:
        A ``networkx.Graph`` with ``weight`` edge attributes (cosine similarity).

    Raises:
        ValueError: If ``similarity_threshold`` is not in [0, 1].

    Contract:
        - The graph is undirected.
        - No self-loops (diagonal excluded).
        - All edge weights are in [0, 1].
        - Node labels are vocabulary token strings.

    Example:
        >>> from kenon.embeddings import TfidfEmbedder
        >>> emb = TfidfEmbedder()
        >>> corpus = ["cat mat sat", "dog ran fast", "cat ran fast"] * 5
        >>> g = build_semantic_graph(emb, corpus, similarity_threshold=0.1)
        >>> isinstance(g.number_of_nodes(), int)
        True
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        msg = f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
        raise ValueError(msg)

    sim_matrix, vocab = cosine_similarity_matrix(embedder, corpus)

    graph: SemanticGraph = nx.Graph()

    stopset = stopwords or frozenset()

    # If k_neighbors is set, build kNN mask
    knn_mask: np.ndarray | None = None
    if k_neighbors is not None and len(vocab) > 0:
        # Build word vectors from the embedder
        embedder.fit(corpus)
        mat = embedder.transform(corpus)
        # Transpose to get word-level vectors (n_features x n_docs)
        word_vectors = mat.T
        if word_vectors.shape[0] == len(vocab):
            k = min(k_neighbors + 1, len(vocab))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine")
            nn.fit(word_vectors)
            knn_mask = np.zeros((len(vocab), len(vocab)), dtype=bool)
            indices = nn.kneighbors(word_vectors, return_distance=False)
            for i, neighbors in enumerate(indices):
                for j in neighbors:
                    knn_mask[i, j] = True
                    knn_mask[j, i] = True

    for i in range(len(vocab)):
        if vocab[i] in stopset:
            continue
        for j in range(i + 1, len(vocab)):
            if vocab[j] in stopset:
                continue
            if knn_mask is not None and not knn_mask[i, j]:
                continue
            sim = min(float(sim_matrix[i, j]), 1.0)
            if sim > similarity_threshold:
                graph.add_edge(vocab[i], vocab[j], weight=sim)

    return graph


def cosine_similarity_matrix(
    embedder: EmbedderProtocol,
    corpus: list[str],
) -> tuple[Matrix, list[Token]]:
    """Return the full pairwise cosine similarity matrix for the vocabulary.

    Args:
        embedder: A fitted or unfitted embedder. Will be fitted on ``corpus``
            if not already fitted.
        corpus: Corpus used to fit the embedder.

    Returns:
        A tuple of ``(similarity_matrix, vocabulary_list)`` where
        ``similarity_matrix[i][j]`` is the cosine similarity between
        token ``i`` and token ``j``.

    Contract:
        - Diagonal values are 1.0 (self-similarity).
        - Matrix is symmetric.
        - All values are in [-1, 1].

    Example:
        >>> from kenon.embeddings import TfidfEmbedder
        >>> emb = TfidfEmbedder()
        >>> corpus = ["cat mat", "dog ran"] * 3
        >>> sim, vocab = cosine_similarity_matrix(emb, corpus)
        >>> sim.shape[0] == sim.shape[1] == len(vocab)
        True
    """
    mat = embedder.fit_transform(corpus)
    vocab_dict = embedder.vocabulary
    vocab = sorted(vocab_dict.keys(), key=lambda w: vocab_dict[w])

    # mat is (n_docs x n_features) for sklearn embedders
    # We want word-level similarity, so transpose to get (n_features x n_docs)
    word_vectors = mat.T  # (n_words x n_docs)

    sim_matrix = sklearn_cosine_similarity(word_vectors).astype(np.float64)
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
    return sim_matrix, vocab


def save_graph(
    graph: SemanticGraph,
    path: str | os.PathLike[str],
    fmt: str = "graphml",
) -> None:
    """Persist a graph to disk.

    Args:
        graph: The graph to save.
        path: Destination file path.
        fmt: Format string. Supported: ``"graphml"`` (default, human-readable XML),
            ``"gml"`` (compact text), ``"pickle"`` (fastest, not human-readable).

    Raises:
        ValueError: If ``fmt`` is not one of the supported formats.

    Contract:
        - The file is written atomically (or as atomically as the format allows).
        - ``"graphml"`` and ``"gml"`` produce human-readable output.

    Example:
        >>> import tempfile, os, networkx as nx
        >>> g = nx.Graph(); g.add_edge("a", "b", weight=0.5)
        >>> with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        ...     save_graph(g, f.name)
        ...     os.path.exists(f.name)
        True
    """
    supported = {"graphml", "gml", "pickle"}
    if fmt not in supported:
        msg = f"Unsupported format '{fmt}'. Supported: {sorted(supported)}"
        raise ValueError(msg)

    path_str = str(path)
    if fmt == "graphml":
        nx.write_graphml(graph, path_str)
    elif fmt == "gml":
        nx.write_gml(graph, path_str)
    elif fmt == "pickle":
        with open(path_str, "wb") as f:
            pickle_mod.dump(graph, f)


def load_graph(
    path: str | os.PathLike[str],
    fmt: str = "graphml",
) -> SemanticGraph:
    """Load a graph from disk.

    Args:
        path: Source file path.
        fmt: Format string. Must match the format used when saving.

    Returns:
        A ``networkx.Graph`` with edge and node attributes restored.

    Raises:
        ValueError: If ``fmt`` is not one of the supported formats.

    Contract:
        - Loaded graph preserves all node and edge attributes from the original.
        - ``"pickle"`` format is not safe for untrusted files.

    Example:
        >>> import tempfile, networkx as nx
        >>> g = nx.Graph(); g.add_edge("x", "y", weight=0.9)
        >>> with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        ...     save_graph(g, f.name)
        ...     g2 = load_graph(f.name)
        >>> g2["x"]["y"]["weight"]
        0.9
    """
    supported = {"graphml", "gml", "pickle"}
    if fmt not in supported:
        msg = f"Unsupported format '{fmt}'. Supported: {sorted(supported)}"
        raise ValueError(msg)

    path_str = str(path)
    if fmt == "graphml":
        return nx.read_graphml(path_str)
    if fmt == "gml":
        return nx.read_gml(path_str)
    # fmt == "pickle"
    with open(path_str, "rb") as f:
        return pickle_mod.load(f)  # noqa: S301
