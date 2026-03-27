"""Step 2: Build every kenon graph variant from the cached corpus.

Reads corpus from data/, builds co-occurrence and semantic graphs with
varying parameters, saves each to graphs/<name>.pkl along with its
backbone variant. Safe to re-run — skips graphs that already exist.

Usage:
    uv run python experiments/google_and_the_mind/scripts/build_graphs.py
    uv run python experiments/google_and_the_mind/scripts/build_graphs.py --smoke-test
"""

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from kenon import (
    CountVectorizerEmbedder,
    PMIEmbedder,
    TfidfEmbedder,
    build_cooccurrence_graph,
    build_semantic_graph,
    extract_backbone,
    get_stopwords,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
GRAPHS_DIR = EXPERIMENT_DIR / "graphs"

# Parameter grid
COOC_WINDOWS = [2, 3, 4, 5]
SIMILARITY_THRESHOLD = 0.3
K_NEIGHBORS = 15
MIN_COOC_WEIGHT = 1e-7
PMI_N_COMPONENTS = 100
BACKBONE_ALPHA_PTILE = 0.5
BACKBONE_MIN_DEGREE = 2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_pickle, log, require_file, save_pickle  # noqa: E402


def get_eval_vocab(norms_path: Path, corpus_tokens: list[list[str]]) -> set[str]:
    """Return the evaluation vocabulary: norm words present in the corpus."""
    norms = pd.read_csv(norms_path, low_memory=False)
    norm_vocab = set(norms["cue"].unique()) | set(norms["target"].unique())
    corpus_vocab: set[str] = set()
    for tokens in corpus_tokens:
        corpus_vocab.update(tokens)
    return norm_vocab & corpus_vocab


def filter_graph_to_vocab(g: nx.Graph, vocab: set[str]) -> nx.Graph:
    """Return a subgraph containing only nodes in the eval vocabulary."""
    nodes_to_keep = set(g.nodes()) & vocab
    return g.subgraph(nodes_to_keep).copy()


def build_and_save(
    name: str,
    g: nx.Graph,
    eval_vocab: set[str],
) -> None:
    """Filter graph to eval vocab, save base + backbone variant."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    base_path = GRAPHS_DIR / f"{name}.pkl"
    backbone_path = GRAPHS_DIR / f"{name}_backbone.pkl"

    # Filter to evaluation vocabulary
    g = filter_graph_to_vocab(g, eval_vocab)

    if not base_path.exists():
        save_pickle(g, base_path)
        log(
            "build",
            f"{name:<20s} nodes={g.number_of_nodes():<6d} "
            f"edges={g.number_of_edges():<10d} → {base_path}",
        )
    else:
        log("build", f"{name:<20s} already exists → {base_path}")

    if not backbone_path.exists():
        backbone = extract_backbone(
            g,
            min_alpha_ptile=BACKBONE_ALPHA_PTILE,
            min_degree=BACKBONE_MIN_DEGREE,
        )
        save_pickle(backbone, backbone_path)
        log(
            "build",
            f"{name + '_backbone':<20s} nodes={backbone.number_of_nodes():<6d} "
            f"edges={backbone.number_of_edges():<10d} → {backbone_path}",
        )
    else:
        log("build", f"{name + '_backbone':<20s} already exists → {backbone_path}")


def build_cooccurrence_graphs(
    flat_tokens: list[str],
    stopwords: frozenset[str],
    eval_vocab: set[str],
    windows: list[int],
) -> None:
    """Build co-occurrence graphs for each window size."""
    for w in windows:
        name = f"cooc_w{w}"
        if (GRAPHS_DIR / f"{name}.pkl").exists() and (
            GRAPHS_DIR / f"{name}_backbone.pkl"
        ).exists():
            log("build", f"{name:<20s} already exists (skipping)")
            continue

        log("build", f"Building {name} (window={w})...")
        g = build_cooccurrence_graph(
            flat_tokens,
            window=w,
            stopwords=stopwords,
            min_weight=MIN_COOC_WEIGHT,
        )
        build_and_save(name, g, eval_vocab)


def build_semantic_graphs(
    corpus_strings: list[str],
    stopwords: frozenset[str],
    eval_vocab: set[str],
    include_pmi: bool = True,
) -> None:
    """Build semantic graphs with different embedders."""
    # Filter corpus strings to eval vocab to keep the similarity matrix
    # manageable (vocab × vocab). Without this, 32K+ vocab terms produce
    # a similarity matrix that doesn't fit in memory.
    filtered_corpus = [
        " ".join(w for w in doc.split() if w in eval_vocab) for doc in corpus_strings
    ]
    filtered_corpus = [doc for doc in filtered_corpus if doc.strip()]
    log("build", f"Filtered corpus for semantic graphs: {len(filtered_corpus)} docs")

    # CountVectorizer
    name = "count"
    if not (GRAPHS_DIR / f"{name}.pkl").exists():
        log("build", f"Building {name}...")
        emb = CountVectorizerEmbedder(stopwords=stopwords)
        g = build_semantic_graph(
            emb,
            filtered_corpus,
            similarity_threshold=SIMILARITY_THRESHOLD,
            k_neighbors=K_NEIGHBORS,
            stopwords=stopwords,
        )
        build_and_save(name, g, eval_vocab)
    else:
        log("build", f"{name:<20s} already exists (skipping)")

    # TF-IDF
    name = "tfidf"
    if not (GRAPHS_DIR / f"{name}.pkl").exists():
        log("build", f"Building {name}...")
        emb = TfidfEmbedder(stopwords=stopwords, sublinear_tf=True)
        g = build_semantic_graph(
            emb,
            filtered_corpus,
            similarity_threshold=SIMILARITY_THRESHOLD,
            k_neighbors=K_NEIGHBORS,
            stopwords=stopwords,
        )
        build_and_save(name, g, eval_vocab)
    else:
        log("build", f"{name:<20s} already exists (skipping)")

    # PMI
    if include_pmi:
        name = "pmi"
        if not (GRAPHS_DIR / f"{name}.pkl").exists():
            log("build", f"Building {name} (n_components={PMI_N_COMPONENTS})...")
            emb = PMIEmbedder(n_components=PMI_N_COMPONENTS)
            g = build_semantic_graph(
                emb,
                corpus_strings,
                similarity_threshold=SIMILARITY_THRESHOLD,
                stopwords=stopwords,
            )
            build_and_save(name, g, eval_vocab)
        else:
            log("build", f"{name:<20s} already exists (skipping)")


def main() -> None:
    """Build all graph variants."""
    parser = argparse.ArgumentParser(description="Build graphs for experiment")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build only cooc_w2 and tfidf for fast testing",
    )
    args = parser.parse_args()

    # Check required files
    require_file(DATA_DIR / "corpus_tokens.pkl", "download_data.py")
    require_file(DATA_DIR / "corpus_strings.pkl", "download_data.py")
    require_file(DATA_DIR / "norms.csv", "download_data.py")

    # Load data
    log("build", "Loading corpus...")
    corpus_tokens: list[list[str]] = load_pickle(DATA_DIR / "corpus_tokens.pkl")
    corpus_strings: list[str] = load_pickle(DATA_DIR / "corpus_strings.pkl")

    # Flatten all tokens for co-occurrence graphs, keeping only eval vocab
    # to avoid O(n^2)-scale pair counting on the full corpus
    stopwords_set = get_stopwords("english")
    eval_vocab = get_eval_vocab(DATA_DIR / "norms.csv", corpus_tokens)
    log("build", f"Evaluation vocabulary: {len(eval_vocab):,} words")

    flat_tokens: list[str] = []
    for tokens in corpus_tokens:
        flat_tokens.extend(t for t in tokens if t in eval_vocab)
    log("build", f"Flat tokens (eval-filtered): {len(flat_tokens):,}")

    stopwords = stopwords_set

    if args.smoke_test:
        # Reduced grid: cooc_w2, tfidf only
        build_cooccurrence_graphs(flat_tokens, stopwords, eval_vocab, windows=[2])
        build_semantic_graphs(corpus_strings, stopwords, eval_vocab, include_pmi=False)
    else:
        build_cooccurrence_graphs(flat_tokens, stopwords, eval_vocab, COOC_WINDOWS)
        build_semantic_graphs(corpus_strings, stopwords, eval_vocab, include_pmi=True)

    log("build", "Done.")


if __name__ == "__main__":
    main()
