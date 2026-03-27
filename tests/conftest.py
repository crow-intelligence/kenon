"""Shared pytest fixtures for kenon tests."""

import pytest


@pytest.fixture
def sample_text() -> str:
    """A short English text for testing."""
    return "The cat sat on the mat. The dog ran in the park."


@pytest.fixture
def sample_corpus() -> list[str]:
    """A small corpus of document strings."""
    return [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat and the dog played",
        "a bird flew over the park",
        "the mat was on the floor",
    ]


@pytest.fixture
def sample_tokens() -> list[str]:
    """A flat list of tokens for co-occurrence tests."""
    return [
        "cat",
        "sat",
        "mat",
        "dog",
        "ran",
        "park",
        "cat",
        "mat",
        "dog",
        "park",
        "cat",
        "dog",
        "sat",
        "mat",
        "ran",
        "park",
        "cat",
        "sat",
    ]
