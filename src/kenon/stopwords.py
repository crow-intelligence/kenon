"""Unified, extensible stopword list merging multiple sources."""

from __future__ import annotations

import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_NLTK_DATA_DOWNLOADED = False


def _ensure_nltk_stopwords() -> None:
    """Download NLTK stopwords corpus if not already available."""
    global _NLTK_DATA_DOWNLOADED  # noqa: PLW0603
    if _NLTK_DATA_DOWNLOADED:
        return
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _NLTK_DATA_DOWNLOADED = True


def get_stopwords(
    lang: str = "english",
    extra: list[str] | None = None,
    sources: list[str] | None = None,
) -> frozenset[str]:
    """Return a merged, frozen set of stopwords.

    Args:
        lang: Language name as used by NLTK (e.g. ``"english"``, ``"german"``).
        extra: Additional tokens to add.
        sources: Which source lists to include. Supported: ``"nltk"``, ``"sklearn"``.
            Defaults to both.

    Returns:
        A frozenset of lowercase stopword strings.

    Raises:
        ValueError: If ``"sklearn"`` is in sources for a non-English language.
        ValueError: If an unsupported source name is given.

    Contract:
        - The returned frozenset contains only lowercase strings.
        - ``extra`` words are always present in the result.
        - NLTK data is auto-downloaded if missing.

    Example:
        >>> sw = get_stopwords("english")
        >>> "the" in sw
        True
        >>> sw2 = get_stopwords("english", extra=["gonna", "wanna"])
        >>> "gonna" in sw2
        True
    """
    if sources is None:
        sources = ["nltk", "sklearn"]

    supported = {"nltk", "sklearn"}
    unknown = set(sources) - supported
    if unknown:
        msg = f"Unsupported stopword sources: {unknown}. Supported: {supported}"
        raise ValueError(msg)

    words: set[str] = set()

    if "sklearn" in sources:
        if lang != "english":
            msg = f"sklearn stopwords are only available for English, not '{lang}'"
            raise ValueError(msg)
        words.update(ENGLISH_STOP_WORDS)

    if "nltk" in sources:
        _ensure_nltk_stopwords()
        words.update(nltk.corpus.stopwords.words(lang))

    if extra:
        words.update(extra)

    return frozenset(w.lower() for w in words)
