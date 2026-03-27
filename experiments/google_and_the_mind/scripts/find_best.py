"""Step 4: Analyse results and write findings.md.

Reads results/results.csv, identifies winners across multiple dimensions,
and writes the formal findings report.

Usage:
    uv run python experiments/google_and_the_mind/scripts/find_best.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kenon

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATA_DIR = EXPERIMENT_DIR / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import log, require_file  # noqa: E402


def load_results() -> pd.DataFrame:
    """Load and validate results.csv."""
    path = RESULTS_DIR / "results.csv"
    require_file(path, "evaluate.py")
    df = pd.read_csv(path)
    return df


def detect_corpus_name() -> str:
    """Determine which corpus was used from file sizes."""
    tokens_path = DATA_DIR / "corpus_tokens.pkl"
    if not tokens_path.exists():
        return "Unknown"
    size_mb = tokens_path.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        return "Wikipedia (N=50,000 articles)"
    return "Brown Corpus (fallback)"


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------


def section_1_headline(df: pd.DataFrame) -> str:
    """Find the single best (graph, centrality) combination."""
    # Filter to centrality rows (not edge_structure)
    cent = df[df["spearman_r"].notna()].copy()
    best = cent.loc[cent["spearman_r"].idxmax()]

    graph = best["graph"]
    measure = best["centrality_measure"]
    r = best["spearman_r"]
    p = best["p_value"]
    n = int(best["n_words"])

    lines = [
        "## 1. Headline result\n",
        f"**Best configuration:** `{graph}` + `{measure}`  ",
        f"**Spearman r with norm PageRank:** {r:.4f} (p={p:.2e}, n={n} words)\n",
        f"The `{graph}` graph with `{measure}` centrality achieves the highest "
        f"rank correlation with human free-association norms, explaining how "
        f"word prominence in corpus-derived graphs reflects prominence in "
        f"human associative memory.\n",
    ]
    return "\n".join(lines)


def section_2_centrality_rankings(df: pd.DataFrame) -> str:
    """Rank centrality measures by mean Spearman r across all graphs."""
    cent = df[df["spearman_r"].notna()].copy()
    agg = (
        cent.groupby("centrality_measure")["spearman_r"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    # Find best graph per centrality
    best_graph_per = cent.loc[
        cent.groupby("centrality_measure")["spearman_r"].idxmax()
    ].set_index("centrality_measure")["graph"]

    lines = [
        "## 2. Centrality measure rankings\n",
        "Ranked by mean Spearman r across all graphs (mean ± std):\n",
        "| Rank | Centrality measure | Mean r | Std | Best graph |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(agg.itertuples(), 1):
        measure = row.centrality_measure
        best_g = best_graph_per.get(measure, "—")
        lines.append(f"| {i} | {measure} | {row.mean:.4f} | {row.std:.4f} | {best_g} |")

    winner = agg.iloc[0]["centrality_measure"]
    lines.append(
        f"\n**Winner:** {winner}. **Interpretation:** This centrality measure "
        f"most consistently recovers the relative prominence of words in "
        f"human associative memory across different graph construction methods.\n"
    )
    return "\n".join(lines)


def section_3_graph_rankings(df: pd.DataFrame) -> str:
    """Rank graphs by best centrality r achieved."""
    cent = df[df["spearman_r"].notna()].copy()
    # Best centrality per graph
    idx = cent.groupby("graph")["spearman_r"].idxmax()
    best = cent.loc[idx].sort_values("spearman_r", ascending=False)

    lines = [
        "## 3. Graph method rankings\n",
        "Ranked by best centrality r achieved:\n",
        "| Rank | Graph | Best centrality | Best r | F1 | Backbone |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(best.itertuples(), 1):
        lines.append(
            f"| {i} | {row.graph} | {row.centrality_measure} | "
            f"{row.spearman_r:.4f} | {row.f1:.4f} | "
            f"{'yes' if row.is_backbone else 'no'} |"
        )

    lines.append("")
    return "\n".join(lines)


def section_4_window_effect(df: pd.DataFrame) -> str:
    """Analyse window size sensitivity for co-occurrence graphs."""
    cent = df[df["spearman_r"].notna()].copy()
    cooc = cent[cent["graph_type"] == "cooccurrence"].copy()

    if cooc.empty:
        return (
            "## 4. Window size effect (co-occurrence graphs)\n\n"
            "No co-occurrence graphs evaluated.\n"
        )

    # Find the best centrality measure overall for cooc graphs
    best_measure = cooc.groupby("centrality_measure")["spearman_r"].mean().idxmax()

    # Filter to that measure
    subset = cooc[cooc["centrality_measure"] == best_measure].copy()

    # Separate base and backbone
    base = subset[~subset["is_backbone"]].sort_values("window")
    backbone = subset[subset["is_backbone"]].sort_values("window")

    lines = [
        "## 4. Window size effect (co-occurrence graphs)\n",
        f"For the best centrality measure (`{best_measure}`):\n",
        "| Window | r (base) | r (backbone) | Δ |",
        "|---|---|---|---|",
    ]

    base_dict = dict(zip(base["window"], base["spearman_r"]))
    backbone_dict = dict(zip(backbone["window"], backbone["spearman_r"]))

    rs_base = []
    for w in sorted(set(base_dict.keys()) | set(backbone_dict.keys())):
        rb = base_dict.get(w, float("nan"))
        rbb = backbone_dict.get(w, float("nan"))
        delta = rbb - rb if not (np.isnan(rb) or np.isnan(rbb)) else float("nan")
        rs_base.append(rb)
        rb_s = f"{rb:.4f}" if not np.isnan(rb) else "—"
        rbb_s = f"{rbb:.4f}" if not np.isnan(rbb) else "—"
        d_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        lines.append(f"| {w} | {rb_s} | {rbb_s} | {d_s} |")

    # Detect pattern
    valid = [r for r in rs_base if not np.isnan(r)]
    if len(valid) >= 2:
        diffs = [valid[i + 1] - valid[i] for i in range(len(valid) - 1)]
        if all(d > 0 for d in diffs):
            pattern = "monotonic increase (larger window = better)"
        elif all(d < 0 for d in diffs):
            pattern = "monotonic decrease (smaller window = better)"
        elif len(valid) >= 3:
            peak_idx = valid.index(max(valid))
            windows = sorted(set(base_dict.keys()) | set(backbone_dict.keys()))
            pattern = f"peaks at window {windows[peak_idx]}"
        else:
            pattern = "no clear trend"
    else:
        pattern = "insufficient data"

    lines.append(f"\n**Pattern:** {pattern}\n")
    return "\n".join(lines)


def section_5_backbone_effect(df: pd.DataFrame) -> str:
    """Analyse backbone effect on centrality correlations."""
    cent = df[df["spearman_r"].notna()].copy()

    # Strip _backbone suffix to get base name
    cent["base_graph"] = cent["graph"].str.replace("_backbone", "", regex=False)
    base = cent[~cent["is_backbone"]].set_index(["base_graph", "centrality_measure"])[
        "spearman_r"
    ]
    backbone = cent[cent["is_backbone"]].copy()
    backbone["base_graph"] = backbone["graph"].str.replace("_backbone", "", regex=False)
    backbone = backbone.set_index(["base_graph", "centrality_measure"])["spearman_r"]

    # Compute deltas
    common_idx = base.index.intersection(backbone.index)
    if common_idx.empty:
        return "## 5. Backbone effect\n\nNo matching base/backbone pairs found.\n"

    deltas = backbone.loc[common_idx] - base.loc[common_idx]
    delta_df = deltas.reset_index()
    delta_df.columns = ["base_graph", "centrality_measure", "delta_r"]

    # Summarise by centrality
    summary = (
        delta_df.groupby("centrality_measure")["delta_r"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )

    lines = [
        "## 5. Backbone effect\n",
        "| Centrality | Mean Δr (backbone − base) | Consistent direction? |",
        "|---|---|---|",
    ]
    for measure, row in summary.iterrows():
        consistent = "yes" if abs(row["mean"]) > row["std"] else "no"
        lines.append(f"| {measure} | {row['mean']:+.4f} | {consistent} |")

    overall_mean = delta_df["delta_r"].mean()
    direction = "improves" if overall_mean > 0 else "reduces"
    lines.append(
        f"\n**Summary:** Backbone extraction on average {direction} centrality "
        f"correlation with norms by {abs(overall_mean):.4f}. "
    )
    if overall_mean > 0:
        lines.append(
            "This suggests that removing noisy edges "
            "with the disparity filter reveals structure more aligned with "
            "human associative memory.\n"
        )
    else:
        lines.append(
            "This suggests that the disparity filter may remove some edges "
            "that carry associative signal, or that the noise-to-signal ratio "
            "in these graphs is already relatively low.\n"
        )

    return "\n".join(lines)


def section_6_semantic_vs_cooc(df: pd.DataFrame) -> str:
    """Compare best semantic vs best co-occurrence graph."""
    cent = df[df["spearman_r"].notna()].copy()

    cooc = cent[cent["graph_type"] == "cooccurrence"]
    semantic = cent[cent["graph_type"] == "semantic"]

    lines = ["## 6. Semantic vs. co-occurrence\n"]

    if cooc.empty or semantic.empty:
        lines.append("Insufficient data for comparison.\n")
        return "\n".join(lines)

    best_cooc = cooc.loc[cooc["spearman_r"].idxmax()]
    best_sem = semantic.loc[semantic["spearman_r"].idxmax()]

    lines.extend(
        [
            "| | Best co-occurrence | Best semantic |",
            "|---|---|---|",
            f"| Graph | {best_cooc['graph']} | {best_sem['graph']} |",
            f"| Best centrality | {best_cooc['centrality_measure']} | "
            f"{best_sem['centrality_measure']} |",
            f"| Best r | {best_cooc['spearman_r']:.4f} | "
            f"{best_sem['spearman_r']:.4f} |",
            f"| F1 | {best_cooc['f1']:.4f} | {best_sem['f1']:.4f} |",
        ]
    )

    if best_cooc["spearman_r"] > best_sem["spearman_r"]:
        winner = "co-occurrence"
        margin = best_cooc["spearman_r"] - best_sem["spearman_r"]
    else:
        winner = "semantic"
        margin = best_sem["spearman_r"] - best_cooc["spearman_r"]

    lines.append(f"\n**Winner:** {winner} by {margin:.4f}.\n")
    return "\n".join(lines)


def section_7_recommendation(df: pd.DataFrame) -> str:
    """Generate recommended configuration for the notebook."""
    cent = df[df["spearman_r"].notna()].copy()
    best = cent.loc[cent["spearman_r"].idxmax()]

    graph = best["graph"]
    measure = best["centrality_measure"]
    graph_type = best["graph_type"]
    is_backbone = bool(best["is_backbone"])

    # Determine method
    if "cooc" in graph:
        method = "cooc"
    elif "count" in graph:
        method = "count"
    elif "tfidf" in graph:
        method = "tfidf"
    elif "pmi" in graph:
        method = "pmi"
    else:
        method = graph

    window = int(best["window"]) if pd.notna(best["window"]) else None

    # Build search grid recommendations
    if graph_type == "cooccurrence" and window is not None:
        window_range = [max(2, window - 1), window + 1]
        threshold_range = None
    else:
        window_range = None
        threshold_range = [0.20, 0.45]

    # Top 3 centralities
    top_centralities = (
        cent.groupby("centrality_measure")["spearman_r"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    config = {
        "graph_type": graph_type,
        "method": method,
        "window": window,
        "centrality_measure": measure,
        "backbone": is_backbone,
        "spearman_r": round(float(best["spearman_r"]), 4),
        "notebook_search_grid": {
            "window_range": window_range,
            "threshold_range": threshold_range,
            "backbone_alphas": [0.3, 0.4, 0.5, 0.6, 0.7],
            "centralities_to_highlight": top_centralities,
        },
    }

    lines = [
        "## 7. Recommended notebook configuration\n",
        "```json",
        json.dumps(config, indent=2),
        "```\n",
    ]
    return "\n".join(lines)


def section_8_anomalies(df: pd.DataFrame) -> str:
    """Report anomalies and caveats."""
    cent = df[df["spearman_r"].notna()].copy()

    anomalies: list[str] = []

    # Check for NaN spearman values (convergence failures)
    nan_rows = cent[cent["spearman_r"].isna()]
    if not nan_rows.empty:
        pairs = [
            f"{r['graph']}+{r['centrality_measure']}" for _, r in nan_rows.iterrows()
        ]
        anomalies.append(f"Convergence failures (NaN results): {', '.join(pairs)}")

    # Check for very low n_words
    low_n = cent[cent["n_words"] < 100]
    if not low_n.empty:
        graphs = low_n["graph"].unique().tolist()
        anomalies.append(
            f"Very low overlap with norm vocabulary (<100 words): {', '.join(graphs)}"
        )

    # Check for graphs with very few edges after backbone
    backbone_rows = cent[cent["is_backbone"]]
    if not backbone_rows.empty:
        tiny = backbone_rows[backbone_rows["graph_edges"] < 10]
        if not tiny.empty:
            graphs = tiny["graph"].unique().tolist()
            anomalies.append(f"Backbone graphs with <10 edges: {', '.join(graphs)}")

    # Check for negative correlations
    negative = cent[cent["spearman_r"] < 0]
    if not negative.empty:
        pairs = [
            f"{r['graph']}+{r['centrality_measure']}"
            for _, r in negative.head(5).iterrows()
        ]
        anomalies.append(f"Negative correlations: {', '.join(pairs)}")

    lines = ["## 8. Anomalies and caveats\n"]
    if anomalies:
        for a in anomalies:
            lines.append(f"- {a}")
    else:
        lines.append("- No anomalies detected.")

    lines.append("")
    return "\n".join(lines)


def section_9_raw_table(df: pd.DataFrame) -> str:
    """Full pivot table of Spearman r values."""
    cent = df[df["spearman_r"].notna()].copy()

    pivot = cent.pivot_table(
        index="graph",
        columns="centrality_measure",
        values="spearman_r",
        aggfunc="first",
    )
    pivot = pivot.sort_index()

    lines = ["## 9. Raw results summary\n"]

    # Build markdown table
    cols = list(pivot.columns)
    header = "| Graph | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)

    for graph, row in pivot.iterrows():
        vals = [f"{row[c]:.4f}" if pd.notna(row[c]) else "—" for c in cols]
        lines.append(f"| {graph} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Analyse results and write findings.md."""
    require_file(RESULTS_DIR / "results.csv", "evaluate.py")

    log("find_best", "Loading results...")
    df = load_results()
    log("find_best", f"Loaded {len(df)} rows from results.csv")

    corpus_name = detect_corpus_name()
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")

    sections = [
        "# Findings: Google and the Mind Experiment",
        f"Generated: {now}  ",
        f"Corpus: {corpus_name}  ",
        "Kenon version: "
        f"{kenon.__version__ if hasattr(kenon, '__version__') else '0.1.0'}",
        "",
        "---\n",
        section_1_headline(df),
        "---\n",
        section_2_centrality_rankings(df),
        "---\n",
        section_3_graph_rankings(df),
        "---\n",
        section_4_window_effect(df),
        "---\n",
        section_5_backbone_effect(df),
        "---\n",
        section_6_semantic_vs_cooc(df),
        "---\n",
        section_7_recommendation(df),
        "---\n",
        section_8_anomalies(df),
        "---\n",
        section_9_raw_table(df),
    ]

    findings_path = EXPERIMENT_DIR / "findings.md"
    findings_path.write_text("\n".join(sections))
    log("find_best", f"→ {findings_path}")


if __name__ == "__main__":
    main()
