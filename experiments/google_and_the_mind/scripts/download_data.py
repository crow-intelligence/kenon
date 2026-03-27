"""Step 1: Download Nelson norms and build a text corpus.

Fetches free association norms from GitHub and streams Wikipedia articles
(or falls back to Brown Corpus) to build a tokenised corpus. All outputs
are cached to data/ — safe to re-run.

Usage:
    uv run python experiments/google_and_the_mind/scripts/download_data.py
    uv run python experiments/google_and_the_mind/scripts/download_data.py --smoke-test
"""

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

from kenon import Tokenizer, get_stopwords

# Resolve paths relative to the experiment root
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"

N_ARTICLES = 50_000
MIN_ARTICLE_TOKENS = 50
SPACY_MODEL = "en_core_web_sm"
NORMS_URL = (
    "https://raw.githubusercontent.com/teonbrooks/"
    "free_association/master/data/free_association_norms.csv"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import log, save_pickle  # noqa: E402

# ---------------------------------------------------------------------------
# Step 1.1 — Download and clean Nelson norms
# ---------------------------------------------------------------------------


def download_norms() -> None:
    """Parse Nelson norms from local file or URL, clean, symmetrise, and save.

    Checks for a local file first (data/free_association.txt or similar),
    then falls back to the URL download.
    """
    norms_path = DATA_DIR / "norms.csv"
    if norms_path.exists():
        log("norms", f"Already exists → {norms_path}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try local file first
    local_file = DATA_DIR / "free_association.txt"
    if not local_file.exists():
        local_file = DATA_DIR / "free_association.csv"

    if local_file.exists():
        log("norms", f"Parsing local file → {local_file}")
        df = pd.read_csv(
            local_file,
            skipinitialspace=True,
            na_values=["", "�"],
            low_memory=False,
            encoding="latin-1",
        )
    else:
        log("norms", f"Downloading from {NORMS_URL}")
        try:
            with urllib.request.urlopen(  # noqa: S310
                NORMS_URL, timeout=30
            ) as resp:
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
        except Exception as exc:
            log(
                "norms",
                f"Download failed: {exc}. Place free_association.txt "
                f"in {DATA_DIR} manually.",
            )
            sys.exit(1)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    df["cue"] = df["cue"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip().str.lower()

    # FSG may be string with leading dot — coerce to numeric
    df["fsg"] = pd.to_numeric(df["fsg"], errors="coerce")

    # BSG for symmetrisation
    if "bsg" in df.columns:
        df["bsg"] = pd.to_numeric(df["bsg"], errors="coerce").fillna(0.0)

    # Keep only rows with positive forward strength
    df = df[df["fsg"] > 0].copy()

    # Symmetrise: fsg_sym = max(fsg, bsg) if BSG column exists,
    # otherwise join on reversed cue/target pairs
    if "bsg" in df.columns:
        df["fsg_sym"] = df[["fsg", "bsg"]].max(axis=1)
    else:
        reverse = df.rename(columns={"cue": "target", "target": "cue"})[
            ["cue", "target", "fsg"]
        ].rename(columns={"fsg": "fsg_reverse"})
        df = df.merge(reverse, on=["cue", "target"], how="left")
        df["fsg_reverse"] = df["fsg_reverse"].fillna(0.0)
        df["fsg_sym"] = df[["fsg", "fsg_reverse"]].max(axis=1)
        df = df.drop(columns=["fsg_reverse"])

    df.to_csv(norms_path, index=False)
    n_pairs = len(df)
    n_cues = df["cue"].nunique()
    log("norms", f"{n_pairs:,} pairs, {n_cues:,} unique cues. → {norms_path}")


# ---------------------------------------------------------------------------
# Step 1.2 — Connectivity check
# ---------------------------------------------------------------------------


def _network_available(url: str = "https://huggingface.co", timeout: int = 5) -> bool:
    """Check if we can reach the given URL."""
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Step 1.3 — Stream and tokenise Wikipedia
# ---------------------------------------------------------------------------


def build_wikipedia_corpus(n_articles: int) -> tuple[list[list[str]], list[str]]:
    """Stream Wikipedia articles and return tokenised corpus."""
    from datasets import load_dataset

    log("corpus", f"Streaming {n_articles:,} Wikipedia articles...")

    tokenizer = Tokenizer(SPACY_MODEL, lemmatize=True, lower=True)
    stopwords = get_stopwords("english")

    corpus_tokens: list[list[str]] = []
    corpus_strings: list[str] = []
    total_tokens = 0

    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )

    for i, article in enumerate(tqdm(ds, total=n_articles, desc="Wikipedia")):
        if i >= n_articles:
            break

        text = article["text"]
        tokens = tokenizer.flat_tokens(text)
        tokens = [t for t in tokens if t not in stopwords and t.isalpha()]

        if len(tokens) < MIN_ARTICLE_TOKENS:
            continue

        corpus_tokens.append(tokens)
        corpus_strings.append(" ".join(tokens))
        total_tokens += len(tokens)

    log(
        "corpus",
        f"{len(corpus_tokens):,} articles, {total_tokens:,} tokens. "
        f"→ {DATA_DIR / 'corpus_tokens.pkl'}",
    )
    return corpus_tokens, corpus_strings


# ---------------------------------------------------------------------------
# Step 1.4 — Brown Corpus fallback
# ---------------------------------------------------------------------------


def build_brown_corpus() -> tuple[list[list[str]], list[str]]:
    """Fall back to NLTK Brown Corpus when Wikipedia is unavailable."""
    log("corpus", "WARNING: Wikipedia unavailable. Using Brown Corpus fallback.")

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)

    from nltk.corpus import brown

    tokenizer = Tokenizer(SPACY_MODEL, lemmatize=True, lower=True)
    stopwords = get_stopwords("english")

    # Process each file ID separately as an "article" to avoid
    # exceeding spaCy's max_length limit on large genre concatenations
    corpus_tokens: list[list[str]] = []
    corpus_strings: list[str] = []
    total_tokens = 0

    file_ids = brown.fileids()
    for fid in tqdm(file_ids, desc="Brown Corpus"):
        raw_text = " ".join(brown.words(fileids=fid))
        tokens = tokenizer.flat_tokens(raw_text)
        tokens = [t for t in tokens if t not in stopwords and t.isalpha()]

        if len(tokens) < MIN_ARTICLE_TOKENS:
            continue

        corpus_tokens.append(tokens)
        corpus_strings.append(" ".join(tokens))
        total_tokens += len(tokens)

    log(
        "corpus",
        f"Brown Corpus: {len(corpus_tokens)} articles (genres), "
        f"{total_tokens:,} tokens.",
    )
    return corpus_tokens, corpus_strings


# ---------------------------------------------------------------------------
# Step 1.5 — Vocabulary coverage report
# ---------------------------------------------------------------------------


def coverage_report(corpus_tokens: list[list[str]], norms_path: Path) -> None:
    """Print vocabulary coverage of norms against corpus."""
    norms = pd.read_csv(norms_path, low_memory=False)
    norm_vocab = {
        str(w)
        for w in (set(norms["cue"].unique()) | set(norms["target"].unique()))
        if pd.notna(w)
    }

    corpus_vocab: set[str] = set()
    for tokens in corpus_tokens:
        corpus_vocab.update(tokens)

    overlap = norm_vocab & corpus_vocab
    missing = norm_vocab - corpus_vocab

    log("coverage", f"Norm vocabulary:    {len(norm_vocab):,} words")
    log("coverage", f"Corpus vocabulary:  {len(corpus_vocab):,} words")
    pct = len(overlap) / len(norm_vocab) * 100 if norm_vocab else 0
    log("coverage", f"Overlap:            {len(overlap):,} words ({pct:.1f}%)")

    if missing:
        first_10 = sorted(missing)[:10]
        log(
            "coverage",
            f"Missing from corpus: {len(missing):,} words "
            f"(first 10: {', '.join(first_10)})",
        )

    if pct < 80:
        log(
            "coverage",
            "⚠ WARNING: Coverage below 80%. Consider increasing N_ARTICLES.",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full download pipeline."""
    parser = argparse.ArgumentParser(description="Download data for experiment")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use Brown Corpus instead of Wikipedia for fast testing",
    )
    args = parser.parse_args()

    # Step 1.1: Norms
    download_norms()

    # Check if corpus already exists
    tokens_path = DATA_DIR / "corpus_tokens.pkl"
    strings_path = DATA_DIR / "corpus_strings.pkl"

    if tokens_path.exists() and strings_path.exists():
        log("corpus", f"Already exists → {tokens_path}")
        corpus_tokens = load_pickle_data(tokens_path)
    else:
        if args.smoke_test:
            corpus_tokens, corpus_strings = build_brown_corpus()
        elif _network_available():
            corpus_tokens, corpus_strings = build_wikipedia_corpus(N_ARTICLES)
        else:
            corpus_tokens, corpus_strings = build_brown_corpus()

        save_pickle(corpus_tokens, tokens_path)
        save_pickle(corpus_strings, strings_path)

    # Step 1.5: Coverage
    coverage_report(corpus_tokens, DATA_DIR / "norms.csv")


def load_pickle_data(path: Path) -> list[list[str]]:
    """Load corpus tokens from pickle."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


if __name__ == "__main__":
    main()
