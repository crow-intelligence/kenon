"""Step 3: Evaluate every cached graph against the Nelson norms.

Loads every graph from graphs/, computes centrality measures, evaluates
against the norm graph, and writes results/results.csv.

Usage:
    uv run python experiments/google_and_the_mind/scripts/evaluate.py
    uv run python experiments/google_and_the_mind/scripts/evaluate.py --smoke-test
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
GRAPHS_DIR = EXPERIMENT_DIR / "graphs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_pickle, log, require_file  # noqa: E402

# ---------------------------------------------------------------------------
# Norm graph construction
# ---------------------------------------------------------------------------


def load_norm_graphs(norms_path: Path) -> tuple[nx.DiGraph, nx.Graph, dict]:
    """Return directed norm graph, undirected norm graph, and norm PageRank.

    The directed graph has FSG as edge weight.
    The undirected graph uses fsg_sym (symmetrised weight).
    norm_pagerank is pre-computed on the directed graph.
    """
    norms = pd.read_csv(norms_path, low_memory=False)
    valid = norms.dropna(subset=["fsg"])

    # Directed graph via fast pandas edgelist
    dg = nx.from_pandas_edgelist(
        valid, "cue", "target", ["fsg"], create_using=nx.DiGraph
    )
    # Rename edge attr to "weight" for consistency
    nx.set_edge_attributes(
        dg,
        {(u, v): d["fsg"] for u, v, d in dg.edges(data=True)},
        "weight",
    )

    # Undirected graph with symmetrised weight
    sym = valid[["cue", "target", "fsg_sym"]].dropna(subset=["fsg_sym"])
    ug = nx.from_pandas_edgelist(
        sym, "cue", "target", ["fsg_sym"], create_using=nx.Graph
    )
    nx.set_edge_attributes(
        ug,
        {(u, v): d["fsg_sym"] for u, v, d in ug.edges(data=True)},
        "weight",
    )

    # PageRank on directed graph
    n = dg.number_of_nodes()
    log("evaluate", f"Computing PageRank on norm graph ({n} nodes)...")
    norm_pagerank = nx.pagerank(dg, weight="weight", alpha=0.85)

    return dg, ug, norm_pagerank


# ---------------------------------------------------------------------------
# Centrality measures
# ---------------------------------------------------------------------------


def compute_centralities(
    g: nx.Graph,
    measures: list[str],
) -> dict[str, dict[str, float]]:
    """Compute requested centrality measures for a graph.

    Returns dict mapping measure name to {node: score} dict.
    """
    results: dict[str, dict[str, float]] = {}

    if g.number_of_nodes() == 0:
        return {m: {} for m in measures}

    for measure in measures:
        if measure == "degree":
            results[measure] = dict(nx.degree_centrality(g))

        elif measure == "weighted_degree":
            strength: dict[str, float] = {}
            for node in g.nodes():
                s = sum(d.get("weight", 1.0) for _, _, d in g.edges(node, data=True))
                strength[node] = s
            # Normalise by max
            max_s = max(strength.values()) if strength else 1.0
            if max_s > 0:
                results[measure] = {n: s / max_s for n, s in strength.items()}
            else:
                results[measure] = strength

        elif measure == "pagerank":
            try:
                results[measure] = nx.pagerank(g, weight="weight", alpha=0.85)
            except Exception:
                results[measure] = {n: float("nan") for n in g.nodes()}

        elif measure == "eigenvector":
            try:
                results[measure] = nx.eigenvector_centrality(
                    g, weight="weight", max_iter=500
                )
            except nx.PowerIterationFailedConvergence:
                results[measure] = {n: float("nan") for n in g.nodes()}

        elif measure == "betweenness":
            k_sample = min(100, g.number_of_nodes())
            results[measure] = nx.betweenness_centrality(
                g, weight="weight", k=k_sample, seed=42
            )

        elif measure == "closeness":
            results[measure] = dict(nx.closeness_centrality(g, distance="weight"))

        elif measure == "hits_hubs":
            try:
                hubs, _authorities = nx.hits(g, max_iter=200)
                results[measure] = hubs
            except nx.PowerIterationFailedConvergence:
                results[measure] = {n: float("nan") for n in g.nodes()}

    return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_centrality(
    centrality_scores: dict[str, float],
    norm_pagerank: dict[str, float],
) -> dict:
    """Spearman r between a centrality measure and norm PageRank."""
    common = set(centrality_scores) & set(norm_pagerank)
    # Filter out NaN values
    pairs = [
        (centrality_scores[w], norm_pagerank[w])
        for w in common
        if not (np.isnan(centrality_scores[w]) or np.isnan(norm_pagerank[w]))
    ]
    if len(pairs) < 3:
        return {"spearman_r": float("nan"), "p_value": float("nan"), "n_words": 0}

    x, y = zip(*pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(x, y)

    return {"spearman_r": float(r), "p_value": float(p), "n_words": len(pairs)}


def evaluate_edge_structure(
    g_kenon: nx.Graph,
    g_norm: nx.Graph,
) -> dict:
    """Precision, recall, F1 of edges against norm graph."""
    common_nodes = set(g_kenon.nodes()) & set(g_norm.nodes())
    if len(common_nodes) < 2:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_common_nodes": len(common_nodes),
            "n_kenon_edges": 0,
            "n_norm_edges": 0,
        }

    # Restrict to common nodes
    kenon_sub = g_kenon.subgraph(common_nodes)
    norm_sub = g_norm.subgraph(common_nodes)

    kenon_edges = set(kenon_sub.edges())
    norm_edges = set(norm_sub.edges())
    # Undirected: normalise edge tuples
    kenon_edges = {(min(u, v), max(u, v)) for u, v in kenon_edges}
    norm_edges = {(min(u, v), max(u, v)) for u, v in norm_edges}

    tp = len(kenon_edges & norm_edges)
    precision = tp / len(kenon_edges) if kenon_edges else 0.0
    recall = tp / len(norm_edges) if norm_edges else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_common_nodes": len(common_nodes),
        "n_kenon_edges": len(kenon_edges),
        "n_norm_edges": len(norm_edges),
    }


def evaluate_weight_correlation(
    g_kenon: nx.Graph,
    g_norm: nx.Graph,
) -> dict:
    """Spearman r of edge weights on shared edges."""
    common_nodes = set(g_kenon.nodes()) & set(g_norm.nodes())
    kenon_sub = g_kenon.subgraph(common_nodes)
    norm_sub = g_norm.subgraph(common_nodes)

    shared_edges = []
    for u, v in kenon_sub.edges():
        a, b = min(u, v), max(u, v)
        if norm_sub.has_edge(a, b):
            kw = kenon_sub[u][v].get("weight", 0.0)
            nw = norm_sub[a][b].get("weight", 0.0)
            shared_edges.append((kw, nw))

    if len(shared_edges) < 3:
        return {
            "weight_spearman_r": float("nan"),
            "weight_p": float("nan"),
            "n_shared_edges": len(shared_edges),
        }

    x, y = zip(*shared_edges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(x, y)

    return {
        "weight_spearman_r": float(r),
        "weight_p": float(p),
        "n_shared_edges": len(shared_edges),
    }


# ---------------------------------------------------------------------------
# Graph metadata parsing
# ---------------------------------------------------------------------------


def parse_graph_meta(name: str) -> dict:
    """Extract metadata from a graph filename."""
    is_backbone = name.endswith("_backbone")
    base_name = name.replace("_backbone", "")

    if base_name.startswith("cooc_w"):
        graph_type = "cooccurrence"
        match = re.search(r"w(\d+)", base_name)
        window = int(match.group(1)) if match else None
    else:
        graph_type = "semantic"
        window = None

    return {
        "is_backbone": is_backbone,
        "graph_type": graph_type,
        "window": window,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MEASURES = [
    "degree",
    "weighted_degree",
    "pagerank",
    "eigenvector",
    "betweenness",
    "closeness",
    "hits_hubs",
]

SMOKE_MEASURES = ["degree", "pagerank", "betweenness"]


def main() -> None:
    """Run evaluation for all graph × centrality combinations."""
    parser = argparse.ArgumentParser(description="Evaluate graphs")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate only degree, pagerank, betweenness",
    )
    args = parser.parse_args()

    measures = SMOKE_MEASURES if args.smoke_test else ALL_MEASURES

    # Check required files
    require_file(DATA_DIR / "norms.csv", "download_data.py")

    # Load norm graphs
    log("evaluate", "Loading norm graphs...")
    _dg, norm_ug, norm_pagerank = load_norm_graphs(DATA_DIR / "norms.csv")
    log(
        "evaluate",
        f"Norm graph: {norm_ug.number_of_nodes()} nodes, "
        f"{norm_ug.number_of_edges()} edges",
    )

    # Find all graph files
    graph_files = sorted(GRAPHS_DIR.glob("*.pkl"))
    if not graph_files:
        log("evaluate", "No graph files found in graphs/. Run build_graphs.py first.")
        sys.exit(1)

    log("evaluate", f"Found {len(graph_files)} graphs, {len(measures)} measures")

    rows: list[dict] = []

    for gf in graph_files:
        name = gf.stem
        log("evaluate", f"Evaluating {name}...")

        g = load_pickle(gf)
        meta = parse_graph_meta(name)

        # Compute centralities
        centralities = compute_centralities(g, measures)

        # Edge structure evaluation (once per graph)
        edge_eval = evaluate_edge_structure(g, norm_ug)
        weight_eval = evaluate_weight_correlation(g, norm_ug)

        for measure_name, scores in centralities.items():
            cent_eval = evaluate_centrality(scores, norm_pagerank)

            rows.append(
                {
                    "graph": name,
                    "centrality_measure": measure_name,
                    "spearman_r": cent_eval["spearman_r"],
                    "p_value": cent_eval["p_value"],
                    "n_words": cent_eval["n_words"],
                    "precision": edge_eval["precision"],
                    "recall": edge_eval["recall"],
                    "f1": edge_eval["f1"],
                    "n_common_nodes": edge_eval["n_common_nodes"],
                    "n_kenon_edges": edge_eval["n_kenon_edges"],
                    "weight_spearman_r": weight_eval["weight_spearman_r"],
                    "weight_p": weight_eval["weight_p"],
                    "n_shared_edges": weight_eval["n_shared_edges"],
                    "graph_nodes": g.number_of_nodes(),
                    "graph_edges": g.number_of_edges(),
                    "is_backbone": meta["is_backbone"],
                    "graph_type": meta["graph_type"],
                    "window": meta["window"],
                }
            )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)

    n_graphs = len(graph_files)
    n_evals = len(rows)
    log(
        "evaluate",
        f"{n_graphs} graphs × {len(measures)} centrality measures "
        f"= {n_evals} evaluations complete.",
    )
    log("evaluate", f"→ {results_path} ({n_evals} rows)")


if __name__ == "__main__":
    main()
