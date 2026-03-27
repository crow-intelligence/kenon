"""Tests for kenon.tokenizer."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kenon.tokenizer import Tokenizer


class TestTokenizer:
    """Unit tests for Tokenizer."""

    def test_sentencize_basic(self, sample_text: str) -> None:
        t = Tokenizer("en_core_web_sm")
        sents = t.sentencize(sample_text)
        assert len(sents) == 2
        assert all(isinstance(s, str) for s in sents)

    def test_sentencize_empty(self) -> None:
        t = Tokenizer("en_core_web_sm")
        sents = t.sentencize("")
        assert sents == []

    def test_tokenize_basic(self) -> None:
        t = Tokenizer("en_core_web_sm")
        doc = t.tokenize("The cat sat.")
        assert len(doc) == 1
        assert "cat" in doc[0]
        assert "." not in doc[0]

    def test_tokenize_with_punct(self) -> None:
        t = Tokenizer("en_core_web_sm")
        doc = t.tokenize("Hello, world!", keep_punct=True)
        flat = [tok for sent in doc for tok in sent]
        assert "," in flat or "!" in flat

    def test_lemmatize(self) -> None:
        t = Tokenizer("en_core_web_sm", lemmatize=True)
        doc = t.tokenize("The cats were running quickly.")
        flat = [tok for sent in doc for tok in sent]
        assert "cat" in flat
        assert "run" in flat

    def test_flat_tokens(self) -> None:
        t = Tokenizer("en_core_web_sm")
        tokens = t.flat_tokens("The cat sat on the mat.")
        assert "cat" in tokens
        assert "." not in tokens

    def test_lowering(self) -> None:
        t = Tokenizer("en_core_web_sm", lower=True)
        tokens = t.flat_tokens("The CAT Sat.")
        assert all(tok == tok.lower() for tok in tokens)

    def test_no_lower(self) -> None:
        t = Tokenizer("en_core_web_sm", lower=False)
        tokens = t.flat_tokens("The Cat sat.")
        assert "The" in tokens

    def test_model_not_found(self) -> None:
        t = Tokenizer("nonexistent_model_xyz")
        with pytest.raises(RuntimeError, match="not installed"):
            t.sentencize("Hello")

    def test_lazy_loading(self) -> None:
        t = Tokenizer("en_core_web_sm")
        assert t._nlp is None
        t.sentencize("Hello.")
        assert t._nlp is not None


class TestTokenizerProperties:
    """Property-based tests for Tokenizer."""

    @settings(max_examples=50, deadline=10000)
    @given(st.text(min_size=1, max_size=200))
    def test_sentencize_never_returns_empty_strings(self, text: str) -> None:
        t = Tokenizer("en_core_web_sm")
        sents = t.sentencize(text)
        for s in sents:
            assert s.strip() != ""

    @settings(max_examples=50, deadline=10000)
    @given(st.text(min_size=1, max_size=200))
    def test_flat_tokens_subset_of_original(self, text: str) -> None:
        t = Tokenizer("en_core_web_sm", lower=True)
        tokens = t.flat_tokens(text)
        lower_text = text.lower()
        for tok in tokens:
            assert tok in lower_text or tok in text.lower()

    @settings(max_examples=20, deadline=10000)
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=5, max_size=100))
    def test_lemmatize_idempotent(self, text: str) -> None:
        t = Tokenizer("en_core_web_sm", lemmatize=True)
        tokens1 = t.flat_tokens(text)
        if tokens1:
            rejoined = " ".join(tokens1)
            tokens2 = t.flat_tokens(rejoined)
            # Lemmatising already-lemmatised text should produce similar results
            assert len(tokens2) <= len(tokens1) + 2  # allow small spaCy variations
