"""Shared pytest fixtures for kenon tests."""

import os

import pytest
from hypothesis import HealthCheck, settings

# Mutation testing (mutmut) runs the same property test from multiple forked
# workers, which trips Hypothesis's ``differing_executors`` health check. This
# env-gated profile suppresses that check (and the shared example database) only
# during mutation runs — the normal test suite is unaffected.
settings.register_profile(
    "mutation",
    suppress_health_check=[HealthCheck.differing_executors],
    database=None,
    deadline=None,
)
if os.environ.get("KENON_MUTATION"):
    settings.load_profile("mutation")


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
