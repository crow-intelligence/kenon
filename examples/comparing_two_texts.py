"""Compare two texts by building co-occurrence graphs and analysing vocabulary gaps.

Steps:
1. Tokenize two short paragraphs on different topics
2. Build a co-occurrence graph for each
3. Compute Jaccard similarity of node sets
4. Find unique nodes in each graph
5. Print vocabulary gap summary
"""

from kenon import Tokenizer, build_cooccurrence_graph, get_stopwords

TEXT_A = """
Machine learning algorithms process large datasets to discover patterns.
Neural networks with deep architectures excel at image recognition and
natural language processing tasks. Training these models requires
significant computational resources including GPUs and large memory.
"""

TEXT_B = """
Climate change threatens biodiversity across the globe. Rising temperatures
cause ice caps to melt and sea levels to rise. Conservation efforts focus
on protecting endangered species and restoring natural habitats. Renewable
energy sources help reduce greenhouse gas emissions.
"""


def main() -> None:
    """Run the text comparison analysis."""
    tokenizer = Tokenizer("en_core_web_sm", lemmatize=True)
    stopwords = get_stopwords("english")

    tokens_a = tokenizer.flat_tokens(TEXT_A)
    tokens_b = tokenizer.flat_tokens(TEXT_B)

    filtered_a = [t for t in tokens_a if t not in stopwords]
    filtered_b = [t for t in tokens_b if t not in stopwords]

    graph_a = build_cooccurrence_graph(filtered_a, window=2, stopwords=stopwords)
    graph_b = build_cooccurrence_graph(filtered_b, window=2, stopwords=stopwords)

    print(f"Text A graph: {graph_a.number_of_nodes()} nodes, "
          f"{graph_a.number_of_edges()} edges")
    print(f"Text B graph: {graph_b.number_of_nodes()} nodes, "
          f"{graph_b.number_of_edges()} edges")

    nodes_a = set(graph_a.nodes())
    nodes_b = set(graph_b.nodes())

    # Jaccard similarity
    intersection = nodes_a & nodes_b
    union = nodes_a | nodes_b
    jaccard = len(intersection) / len(union) if union else 0.0
    print(f"\nJaccard similarity of node sets: {jaccard:.3f}")

    # Shared nodes
    print(f"\nShared nodes ({len(intersection)}): {sorted(intersection)}")

    # Unique to A
    only_a = nodes_a - nodes_b
    print(f"Only in Text A ({len(only_a)}): {sorted(only_a)}")

    # Unique to B
    only_b = nodes_b - nodes_a
    print(f"Only in Text B ({len(only_b)}): {sorted(only_b)}")

    # Vocabulary gap summary
    print(f"\nVocabulary gap: {len(only_a) + len(only_b)} unique terms "
          f"out of {len(union)} total")


if __name__ == "__main__":
    main()
