"""Analyse a news article using co-occurrence and semantic graphs.

Steps:
1. Tokenize the article
2. Remove stopwords
3. Build co-occurrence graph (window=3)
4. Extract backbone
5. Print top 10 nodes by degree centrality
6. Build semantic graphs with CountVectorizer and TF-IDF embedders
"""

import networkx as nx

from kenon import (
    CountVectorizerEmbedder,
    TfidfEmbedder,
    Tokenizer,
    build_cooccurrence_graph,
    build_semantic_graph,
    extract_backbone,
    get_stopwords,
)

ARTICLE = """
The European Central Bank held interest rates steady on Thursday, pausing
after a historic series of hikes that took borrowing costs to record levels.
ECB President Christine Lagarde said inflation in the euro zone was declining
but remained too high. The central bank kept its main refinancing rate at
4.5 percent. Economists had widely expected the decision after inflation
data showed price growth slowing across the 20-nation currency bloc.
However, Lagarde warned that the fight against inflation was not over and
that the bank would keep rates at sufficiently restrictive levels for as
long as necessary. Financial markets reacted positively to the announcement,
with European stocks rising and bond yields falling. The euro weakened
slightly against the dollar as traders interpreted the decision as a sign
that the rate-hiking cycle had peaked. Analysts noted that the ECB faces
a delicate balancing act between controlling inflation and supporting
economic growth, which has been sluggish in recent quarters.
"""


def main() -> None:
    """Run the news article analysis."""
    # Tokenize
    tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
    tokens = tokenizer.flat_tokens(ARTICLE)
    print(f"Total tokens: {len(tokens)}")

    # Remove stopwords
    stopwords = get_stopwords("english")
    filtered_tokens = [t for t in tokens if t not in stopwords]
    print(f"Tokens after stopword removal: {len(filtered_tokens)}")

    # Build co-occurrence graph
    cooc_graph = build_cooccurrence_graph(
        filtered_tokens, window=3, stopwords=stopwords
    )
    print(f"\nCo-occurrence graph: {cooc_graph.number_of_nodes()} nodes, "
          f"{cooc_graph.number_of_edges()} edges")

    # Extract backbone
    backbone = extract_backbone(cooc_graph, min_alpha_ptile=0.3, min_degree=2)
    print(f"Backbone: {backbone.number_of_nodes()} nodes, "
          f"{backbone.number_of_edges()} edges")

    # Top 10 nodes by degree centrality
    centrality = nx.degree_centrality(backbone)
    top_10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 nodes by degree centrality (backbone):")
    for word, score in top_10:
        print(f"  {word}: {score:.3f}")

    # Semantic graph with CountVectorizer
    sentences = ARTICLE.strip().split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    print("\n--- CountVectorizer Semantic Graph ---")
    count_emb = CountVectorizerEmbedder(stopwords=stopwords)
    count_graph = build_semantic_graph(
        count_emb, sentences, similarity_threshold=0.2
    )
    print(f"Nodes: {count_graph.number_of_nodes()}, "
          f"Edges: {count_graph.number_of_edges()}")

    # Semantic graph with TF-IDF
    print("\n--- TF-IDF Semantic Graph ---")
    tfidf_emb = TfidfEmbedder(stopwords=stopwords)
    tfidf_graph = build_semantic_graph(
        tfidf_emb, sentences, similarity_threshold=0.2
    )
    print(f"Nodes: {tfidf_graph.number_of_nodes()}, "
          f"Edges: {tfidf_graph.number_of_edges()}")


if __name__ == "__main__":
    main()
