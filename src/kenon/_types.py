"""Shared type aliases for the kenon package."""

from typing import TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import NDArray

Token: TypeAlias = str
"""A raw or lemmatised token."""

Sentence: TypeAlias = list[Token]
"""A sentence is a sequence of tokens (post-tokenisation)."""

Document: TypeAlias = list[Sentence]
"""A document is a sequence of sentences."""

Weight: TypeAlias = float
"""Edge weight (cosine similarity, PMI, relative frequency, etc.)."""

SemanticGraph: TypeAlias = nx.Graph
"""A weighted NetworkX undirected graph."""

Matrix: TypeAlias = NDArray[np.float64]
"""Float matrix — rows = documents/windows, cols = vocab."""
