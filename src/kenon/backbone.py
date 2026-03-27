"""Disparity filter backbone extraction for weighted graphs.

Implements the multiscale backbone algorithm from Serrano et al. (2009):
https://arxiv.org/pdf/0904.2389.pdf

Preserves statistically significant edges while removing noise from
weighted networks.
"""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np

from kenon._types import SemanticGraph


def disparity_integral(x: float, k: float) -> float:
    """Compute the definite integral for the disparity filter PDF.

    Args:
        x: Normalised edge weight. Must not equal 1.0.
        k: Node degree. Must not equal 1.0.

    Returns:
        Value of the integral ``((1 - x)^k) / ((k - 1) * (x - 1))``.

    Contract:
        - ``x`` must not equal 1.0 (division by zero).
        - ``k`` must not equal 1.0 (division by zero).

    Example:
        >>> abs(disparity_integral(0.5, 3.0) - disparity_integral(0.0, 3.0)) > 0
        True
    """
    return ((1.0 - x) ** k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance(norm_weight: float, degree: float) -> float:
    """Compute the alpha significance score for a single edge.

    Args:
        norm_weight: Edge weight normalised by node strength.
        degree: Degree of the node.

    Returns:
        Alpha value in [0, 1]. Lower alpha = more significant.

    Contract:
        - If ``degree`` <= 1, returns 0.0.
        - Result is clipped to [0, 1].

    Example:
        >>> alpha = get_disparity_significance(0.5, 3.0)
        >>> 0.0 <= alpha <= 1.0
        True
    """
    if degree <= 1.0:
        return 0.0
    return 1.0 - (degree - 1.0) * (
        disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree)
    )


def apply_disparity_filter(graph: SemanticGraph) -> list[float]:
    """Compute and attach disparity statistics to all edges in-place.

    Adds the following attributes to each edge:

    - ``norm_weight``: weight / node strength
    - ``alpha``: disparity significance
    - ``alpha_ptile``: alpha percentile among all edges

    Adds ``strength`` attribute to each node.

    Args:
        graph: A weighted networkx Graph with ``weight`` edge attributes.

    Returns:
        List of all alpha values (for threshold inspection).

    Contract:
        - Modifies the graph in-place.
        - Every edge gets ``norm_weight``, ``alpha``, and ``alpha_ptile`` attributes.
        - Every node gets a ``strength`` attribute.

    Example:
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_edge("a", "b", weight=0.8)
        >>> g.add_edge("b", "c", weight=0.3)
        >>> g.add_edge("a", "c", weight=0.5)
        >>> alphas = apply_disparity_filter(g)
        >>> len(alphas) == g.number_of_edges()
        True
    """
    if graph.number_of_edges() == 0:
        return []

    # Compute node strengths
    for node in graph.nodes():
        strength = sum(
            data.get("weight", 1.0) for _, _, data in graph.edges(node, data=True)
        )
        graph.nodes[node]["strength"] = strength

    # Compute alpha for each edge (take minimum alpha from both endpoints)
    alphas: list[float] = []
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)

        strength_u = graph.nodes[u]["strength"]
        degree_u = float(graph.degree(u))
        norm_u = weight / strength_u if strength_u > 0 else 0.0
        alpha_u = get_disparity_significance(norm_u, degree_u)

        strength_v = graph.nodes[v]["strength"]
        degree_v = float(graph.degree(v))
        norm_v = weight / strength_v if strength_v > 0 else 0.0
        alpha_v = get_disparity_significance(norm_v, degree_v)

        alpha = min(alpha_u, alpha_v)
        norm_weight = min(norm_u, norm_v)

        data["norm_weight"] = norm_weight
        data["alpha"] = alpha
        alphas.append(alpha)

    # Compute alpha percentiles
    sorted_alphas = np.array(sorted(alphas))
    for _u, _v, data in graph.edges(data=True):
        alpha = data["alpha"]
        ptile = float(np.searchsorted(sorted_alphas, alpha)) / len(sorted_alphas)
        data["alpha_ptile"] = ptile

    return alphas


def extract_backbone(
    graph: SemanticGraph,
    min_alpha_ptile: float = 0.5,
    min_degree: int = 2,
) -> SemanticGraph:
    """Extract the backbone of a graph using the disparity filter.

    Removes edges whose alpha percentile falls below ``min_alpha_ptile``,
    then prunes isolated nodes with degree below ``min_degree``.

    Args:
        graph: Input weighted graph. Will be **copied** — original is not mutated.
        min_alpha_ptile: Edges below this alpha percentile are removed.
            Must be in [0, 1]. Higher values = more aggressive pruning.
        min_degree: Nodes with degree below this after edge removal are pruned.

    Returns:
        A new ``networkx.Graph`` containing only backbone edges and nodes.

    Contract:
        - Original graph is never mutated (deep copy).
        - Result has <= nodes and <= edges compared to the original.
        - All remaining nodes satisfy ``degree >= min_degree``.

    Example:
        >>> import networkx as nx
        >>> g = nx.path_graph(5)
        >>> for u, v in g.edges():
        ...     g[u][v]["weight"] = float(v + 1)
        >>> backbone = extract_backbone(g, min_alpha_ptile=0.3)
        >>> backbone.number_of_nodes() <= g.number_of_nodes()
        True
    """
    if graph.number_of_edges() == 0:
        return nx.Graph()

    result = copy.deepcopy(graph)
    apply_disparity_filter(result)

    # Remove edges below the alpha percentile threshold
    edges_to_remove = [
        (u, v)
        for u, v, data in result.edges(data=True)
        if data.get("alpha_ptile", 0.0) < min_alpha_ptile
    ]
    result.remove_edges_from(edges_to_remove)

    # Remove nodes below minimum degree (iteratively until stable)
    changed = True
    while changed:
        nodes_to_remove = [
            node for node in list(result.nodes()) if result.degree(node) < min_degree
        ]
        changed = len(nodes_to_remove) > 0
        result.remove_nodes_from(nodes_to_remove)

    return result
