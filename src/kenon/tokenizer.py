"""Sentence and word tokeniser backed by spaCy.

Supports optional lemmatisation for any language that has a spaCy model.
Long texts are automatically chunked to stay within spaCy's memory limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import spacy
import spacy.tokens

from kenon._types import Document, Token

if TYPE_CHECKING:
    from spacy.language import Language

# Maximum characters per spaCy nlp() call.  The parser and NER
# components use ~1 GB per 100 K chars; with sentencizer replacing
# the parser we can afford much larger chunks, but we still cap
# to keep peak memory reasonable on modest hardware.
_CHUNK_TARGET = 200_000


class Tokenizer:
    """Sentence and word tokeniser backed by spaCy.

    Supports optional lemmatisation for any language that has a spaCy model.
    Long texts are automatically split into chunks before processing so that
    book-length inputs work without running out of memory.

    Args:
        lang: spaCy model name, e.g. ``"en_core_web_sm"`` or ``"de_core_news_sm"``.
        lemmatize: If True, return lemmas instead of surface forms.
        lower: If True, lowercase all tokens.

    Contract:
        - ``lang`` must be a valid spaCy model name installed on the system.
        - Raises ``RuntimeError`` if the model is not installed.
        - All methods accept ``str`` inputs only, never file paths.
        - Pure whitespace and punctuation tokens are excluded by default.

    Example:
        >>> t = Tokenizer("en_core_web_sm")
        >>> sents = t.sentencize("The cat sat. The dog ran.")
        >>> len(sents)
        2
    """

    def __init__(
        self,
        lang: str = "en_core_web_sm",
        lemmatize: bool = False,
        lower: bool = True,
    ) -> None:
        self._lang = lang
        self._lemmatize = lemmatize
        self._lower = lower
        self._nlp: Language | None = None

    def _load(self) -> Language:
        """Lazily load the spaCy model on first use.

        The heavy parser and NER components are replaced by a lightweight
        rule-based sentencizer, which keeps memory usage low even for
        book-length texts while still providing accurate tokenization
        and lemmatization.
        """
        if self._nlp is None:
            try:
                self._nlp = spacy.load(
                    self._lang,
                    disable=["parser", "ner"],
                )
            except OSError as exc:
                msg = (
                    f"spaCy model '{self._lang}' is not installed.\n"
                    f"Run: python -m spacy download {self._lang}\n"
                    f"Available models: https://spacy.io/models"
                )
                raise RuntimeError(msg) from exc
            # Add rule-based sentencizer to replace the disabled parser
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")
        return self._nlp

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that are safe for spaCy to process.

        Splits on paragraph boundaries (double newlines) to avoid breaking
        mid-sentence. Falls back to single newlines, then whitespace.
        """
        if len(text) <= _CHUNK_TARGET:
            return [text]

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= _CHUNK_TARGET:
                chunks.append(remaining)
                break

            # Find a paragraph break near the target size
            split_at = -1
            for sep in ["\n\n", "\n", " "]:
                pos = remaining.rfind(sep, 0, _CHUNK_TARGET)
                if pos > _CHUNK_TARGET // 2:  # don't split too early
                    split_at = pos + len(sep)
                    break

            if split_at <= 0:
                # No good split point found — hard split at target
                split_at = _CHUNK_TARGET

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        return chunks

    def _process(self, text: str) -> list[spacy.tokens.Doc]:
        """Process text through spaCy, chunking if necessary.

        Chunks are processed one at a time (not batched) so that each
        spaCy Doc can be garbage-collected before the next is created,
        keeping peak memory low for book-length texts.
        """
        nlp = self._load()
        chunks = self._chunk_text(text)
        nlp.max_length = max(nlp.max_length, max(len(c) for c in chunks) + 100)
        return [nlp(chunk) for chunk in chunks]

    def _token_text(self, token: spacy.tokens.Token) -> str:  # type: ignore[name-defined]
        """Extract text from a spaCy token, applying lemmatisation and lowering."""
        text = token.lemma_ if self._lemmatize else token.text
        if self._lower:
            text = text.lower()
        return text

    def sentencize(self, text: str) -> list[str]:
        """Split text into sentence strings.

        Args:
            text: Input text to split into sentences.

        Returns:
            List of sentence strings.

        Contract:
            - Never returns empty strings in the output list.
            - Sentence boundaries are determined by spaCy's sentence segmenter.

        Example:
            >>> t = Tokenizer("en_core_web_sm")
            >>> sents = t.sentencize("Hello world. Goodbye world.")
            >>> len(sents) == 2
            True
        """
        docs = self._process(text)
        result: list[str] = []
        for doc in docs:
            for sent in doc.sents:
                stripped = sent.text.strip()
                if stripped:
                    result.append(stripped)
        return result

    def tokenize(self, text: str, *, keep_punct: bool = False) -> Document:
        """Split text into a nested list: sentences -> tokens.

        Args:
            text: Input text to tokenize.
            keep_punct: If True, keep punctuation tokens.

        Returns:
            A list of sentences, each a list of token strings.

        Contract:
            - Whitespace-only tokens are always excluded.
            - Punctuation tokens excluded unless ``keep_punct=True``.

        Example:
            >>> t = Tokenizer("en_core_web_sm", lemmatize=True)
            >>> doc = t.tokenize("The cats were running.")
            >>> "cat" in doc[0]
            True
            >>> "run" in doc[0]
            True
        """
        docs = self._process(text)
        result: Document = []
        for doc in docs:
            for sent in doc.sents:
                tokens: list[Token] = []
                for token in sent:
                    if token.is_space:
                        continue
                    if not keep_punct and token.is_punct:
                        continue
                    tokens.append(self._token_text(token))
                if tokens:
                    result.append(tokens)
        return result

    def flat_tokens(self, text: str, *, keep_punct: bool = False) -> list[Token]:
        """Return all tokens in a single flat list (no sentence structure).

        Args:
            text: Input text to tokenize.
            keep_punct: If True, keep punctuation tokens.

        Returns:
            Flat list of token strings.

        Contract:
            - Equivalent to flattening the result of ``tokenize()``.
            - All returned tokens are substrings of the original text
              (possibly lowercased or lemmatised).

        Example:
            >>> t = Tokenizer("en_core_web_sm")
            >>> tokens = t.flat_tokens("The cat sat on the mat.")
            >>> "cat" in tokens
            True
            >>> "." not in tokens
            True
        """
        doc = self.tokenize(text, keep_punct=keep_punct)
        return [token for sent in doc for token in sent]
