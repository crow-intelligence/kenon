"""Shared Hypothesis strategies for kenon tests."""

import string

import networkx as nx
from hypothesis import strategies as st

token_list = st.lists(
    st.text(
        alphabet=string.ascii_lowercase,
        min_size=2,
        max_size=20,
    ).filter(lambda s: len(s) >= 2),
    min_size=5,
    max_size=200,
)
"""Lists of lowercase alphabetic strings (2-20 chars), 5-200 items."""

small_corpus = st.lists(
    st.lists(
        st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=10).filter(
            lambda s: len(s) >= 2
        ),
        min_size=5,
        max_size=30,
    ).map(lambda words: " ".join(words)),
    min_size=3,
    max_size=20,
)
"""Lists of 3-20 'sentence' strings, each with 5-30 words."""

weight_value = st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
"""Floats in (0.0, 1.0) exclusive."""

similarity_threshold = st.floats(min_value=0.0, max_value=0.99, allow_nan=False)
"""Floats in [0.0, 0.99]."""


@st.composite
def weighted_graph(draw: st.DrawFn) -> nx.Graph:
    """Small undirected weighted nx.Graph with string node labels."""
    nodes = draw(
        st.lists(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8),
            min_size=2,
            max_size=8,
            unique=True,
        )
    )
    edge_spec = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=len(nodes) - 1),
                st.integers(min_value=0, max_value=len(nodes) - 1),
                st.floats(min_value=0.001, max_value=1000.0, allow_nan=False),
            ),
            min_size=0,
            max_size=15,
        )
    )
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(nodes)
    for u_idx, v_idx, w in edge_spec:
        if u_idx != v_idx:
            g.add_edge(nodes[u_idx], nodes[v_idx], weight=w)
    return g
