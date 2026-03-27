"""Shared Hypothesis strategies for kenon tests."""

import string

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
