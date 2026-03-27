# Google and the Mind Experiment

Which kenon graph construction method, parameter settings, and centrality
measure best recovers the structure of human free association as documented
in the Nelson norms? This experiment runs a systematic parameter search and
produces `findings.md` with the answer.

## How to run

```bash
# Install dependencies
uv sync --dev
python -m spacy download en_core_web_sm

# Full run (takes ~1 hour with Wikipedia)
uv run python experiments/google_and_the_mind/scripts/download_data.py
uv run python experiments/google_and_the_mind/scripts/build_graphs.py
uv run python experiments/google_and_the_mind/scripts/evaluate.py
uv run python experiments/google_and_the_mind/scripts/find_best.py

# Quick smoke test (~2 minutes with Brown Corpus)
uv run python experiments/google_and_the_mind/scripts/download_data.py --smoke-test
uv run python experiments/google_and_the_mind/scripts/build_graphs.py  --smoke-test
uv run python experiments/google_and_the_mind/scripts/evaluate.py      --smoke-test
uv run python experiments/google_and_the_mind/scripts/find_best.py
```

Output: `findings.md` in this directory.
