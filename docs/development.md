# Development Guide

This document is for contributors working on the AI-Enhanced QKD pipeline.

---

## 1. Getting Started

```bash
./scripts/bootstrap.sh
```

This sets up a virtual environment and installs dependencies. All tooling is
local to the repository to avoid global Python package conflicts.

---

## 2. Running the Pipeline

```bash
./scripts/run.sh
```

The pipeline is deterministic. If you change any model logic, re-run the
pipeline and compare the resulting `results/metrics.json`.

---

## 3. Running Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

All tests are deterministic and should pass without internet access.

---

## 4. Code Layout

```
src/ai_qkd/
  anomaly.py            # Logistic regression anomaly detection
  cli.py                # Command-line interface
  config.py             # Config dataclasses and loaders
  data.py               # Synthetic data generation/preprocessing
  error_correction.py   # Window-based error correction
  evaluation.py         # Metrics evaluation utilities
  pipeline.py           # Orchestrates pipeline steps
  rl.py                 # Bandit policy training
```

Legacy wrappers in `src/` provide compatibility with older entrypoints but all
logic lives in `src/ai_qkd/`.

---

## 5. Adding New Models

If you add new model types:

1. Create a new module under `src/ai_qkd/`
2. Add a CLI subcommand in `ai_qkd/cli.py`
3. Add tests in `tests/`
4. Update `docs/data-contract.md`
5. Update `scripts/verify.sh` to include new smoke checks

---

## 6. Deterministic Randomness

Use `random.Random(seed)` for all random operations. Avoid the legacy global
RNG to preserve reproducibility.

---

## 7. Style Notes

- Prefer small, pure functions
- Use dataclasses for configuration and model metadata
- Avoid heavy dependencies unless strictly necessary
- Keep model artifacts JSON-serializable

---

## 8. Logging and Observability

The CLI prints key outputs such as model paths and metrics. For deeper
analysis, open `results/metrics.json` directly or parse it with Python.

---

## 9. Example Debug Session

```bash
PYTHONPATH=src python -m ai_qkd.cli generate-data
PYTHONPATH=src python -m ai_qkd.cli preprocess
PYTHONPATH=src python -m ai_qkd.cli train-anomaly
PYTHONPATH=src python -m ai_qkd.cli train-error
PYTHONPATH=src python -m ai_qkd.cli train-rl
PYTHONPATH=src python -m ai_qkd.cli evaluate
```

---

## 10. Contribution Checklist

Before submitting changes:

- [ ] `python -m unittest` passes
- [ ] `./scripts/verify.sh` passes
- [ ] Documentation updated
- [ ] No binary artifacts added to git
