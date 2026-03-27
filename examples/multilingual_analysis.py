"""Demonstrate multilingual tokenisation and co-occurrence graph analysis.

Requires spaCy models for the languages used. Install with:
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm

This example tokenizes English and German texts, builds co-occurrence graphs,
and compares their density.
"""

import networkx as nx

from kenon import Tokenizer, build_cooccurrence_graph, get_stopwords

ENGLISH_TEXT = """
The researchers published their findings in a prestigious scientific journal.
Their study examined the effects of temperature on plant growth across
different climate zones. The results showed significant variation in
growth rates between tropical and temperate regions.
"""

GERMAN_TEXT = """
Die Forscher veröffentlichten ihre Ergebnisse in einer angesehenen
wissenschaftlichen Zeitschrift. Ihre Studie untersuchte die Auswirkungen
der Temperatur auf das Pflanzenwachstum in verschiedenen Klimazonen.
Die Ergebnisse zeigten erhebliche Unterschiede bei den Wachstumsraten
zwischen tropischen und gemäßigten Regionen.
"""


def analyse_text(
    text: str,
    lang_model: str,
    stopword_lang: str,
    label: str,
    stopword_sources: list[str] | None = None,
) -> None:
    """Tokenize, build co-occurrence graph, and print stats."""
    tokenizer = Tokenizer(lang_model, lemmatize=True)
    tokens = tokenizer.flat_tokens(text)

    sources = stopword_sources or ["nltk"]
    stopwords = get_stopwords(stopword_lang, sources=sources)
    filtered = [t for t in tokens if t not in stopwords]

    graph = build_cooccurrence_graph(filtered, window=2, stopwords=stopwords)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph)

    print(f"\n--- {label} ---")
    print(f"  Tokens: {len(tokens)}, After filtering: {len(filtered)}")
    print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
    print(f"  Density: {density:.4f}")

    if n_nodes > 0:
        top_nodes = sorted(
            graph.degree(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"  Top 5 nodes by degree: "
              f"{[(n, d) for n, d in top_nodes]}")


def main() -> None:
    """Run multilingual analysis."""
    print("Multilingual Co-occurrence Graph Analysis")
    print("=" * 50)

    analyse_text(
        ENGLISH_TEXT,
        "en_core_web_sm",
        "english",
        "English",
        stopword_sources=["nltk", "sklearn"],
    )

    try:
        analyse_text(
            GERMAN_TEXT,
            "de_core_news_sm",
            "german",
            "German",
            stopword_sources=["nltk"],
        )
    except RuntimeError as e:
        print(f"\nGerman analysis skipped: {e}")
        print("Install the German model with: python -m spacy download de_core_news_sm")


if __name__ == "__main__":
    main()
