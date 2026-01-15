# AI-Enhanced QKD (Deterministic Reference Pipeline)

This repository provides a **deterministic**, lightweight, and fully reproducible
pipeline that simulates AI-assisted Quantum Key Distribution (QKD) workflows.
It is designed to be runnable in CI without heavyweight ML frameworks while still
exercising the core concepts: data generation, anomaly detection, error
correction, and key-distribution policy optimization.

The pipeline is intentionally small, fully deterministic, and validated end-to-end
by `./scripts/verify.sh`.

---

## ✅ README Truth Contract

### Installation

```bash
./scripts/bootstrap.sh
```

This script:
- Creates a local virtual environment (`.venv`)
- Uses only the Python standard library (no external dependencies)

### Configuration

The pipeline uses a single JSON config:

- `config/pipeline.json`

You can regenerate it at any time:

```bash
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json init-config
```

### Run

```bash
./scripts/run.sh
```

This runs the full pipeline using `config/pipeline.json`:
1. Generates synthetic raw QKD data (`data/raw/quantum_data.csv`)
2. Preprocesses into train/val/test splits (`data/processed/*.csv`)
3. Trains anomaly detection model (`models/anomaly/logistic_model.json`)
4. Tunes error-correction window (`models/error_correction/window_model.json`)
5. Trains bandit policy for key distribution (`models/key_distribution/bandit_policy.json`)
6. Writes metrics to `results/metrics.json`

### Expected Output/Behavior

After `./scripts/run.sh`, you should see:

- `results/metrics.json` (contains accuracy, bit-error rate, and reward)
- Trained model JSON files in `models/`
- Processed datasets in `data/processed/`

### Verification / Tests

```bash
./scripts/verify.sh
```

This command:
- Runs the full pipeline
- Verifies metrics are above the required thresholds
- Executes the `unittest` suite

---

## Verified Quickstart (Commands Actually Executed)

```bash
./scripts/bootstrap.sh
./scripts/run.sh
```

## Verified Verification (Commands Actually Executed)

```bash
./scripts/verify.sh
```

---

## CLI Usage

All functionality is also exposed via the `qkd` CLI:

```bash
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json run
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json generate-data
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json preprocess
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json train-anomaly
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json train-error
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json train-rl
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json evaluate
```

---

## Troubleshooting

- **Missing dependencies**: re-run `./scripts/bootstrap.sh`.
- **Stale metrics/models**: delete `models/` or `results/` and re-run.
- **Configuration issues**: regenerate config with `init-config`.

For deeper details, see:
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/data-contract.md`](docs/data-contract.md)
- [`docs/verification.md`](docs/verification.md)
- [`docs/troubleshooting.md`](docs/troubleshooting.md)

---

## Repository Layout

```
config/          # Pipeline configuration
scripts/         # Bootstrap, run, and verify scripts
src/ai_qkd/      # Pipeline implementation
tests/           # unittest suite
results/         # Output metrics (generated)
models/          # Model artifacts (generated)
```

---

## License

MIT
