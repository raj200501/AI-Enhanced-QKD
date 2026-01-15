# Verification Guide

This document describes how the QKD pipeline is verified locally and in CI.

---

## 1. Canonical Verification Command

The single entrypoint for verification is:

```bash
./scripts/verify.sh
```

This script performs the following steps:

1. **Bootstrap the environment**
   - Creates `.venv`
   - Installs pinned dependencies
2. **Run the pipeline**
   - Generates raw data
   - Preprocesses into train/val/test splits
   - Trains anomaly detector
   - Tunes error correction
   - Trains bandit policy
   - Writes `results/metrics.json`
3. **Smoke checks the metrics**
   - Accuracy ≥ 0.70
   - BER ≤ 0.35
   - Mean reward ≥ 0.55
4. **Runs the unittest suite**

If any step fails, the script exits with a non-zero status.

---

## 2. CI Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the same
verification command on:

- Every `push`
- Every `pull_request`

This ensures that CI behavior matches the local workflow exactly.

---

## 3. Why the Smoke Checks Matter

The smoke checks ensure that the pipeline is not only syntactically valid, but
produces **meaningful** results. For example:

- If data normalization breaks, anomaly accuracy will drop below 0.70
- If error correction regresses, BER will exceed 0.35
- If RL policy training is broken, mean reward will fall below 0.55

---

## 4. Determinism Guarantees

The pipeline uses seeded RNGs in every stage:

- Data generation
- Error correction simulation
- Bandit policy training

This removes flakiness and makes it safe to run in CI environments.

---

## 5. Running Just the Tests

If you only want to run the Python unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

However, the full verification command should be used to validate
end-to-end behavior.

---

## 6. Reproducing Metrics

To regenerate the metrics file:

```bash
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json run
cat results/metrics.json
```

---

## 7. Troubleshooting Verification Failures

- **Dependency errors**: Re-run `./scripts/bootstrap.sh`
- **Metrics threshold failure**: Check `config/pipeline.json` and rerun
- **Filesystem permissions**: Ensure the repository is writable

---

## 8. What To Check In Logs

During a CI failure, inspect:

- `results/metrics.json`
- `unittest` output

These artifacts provide direct evidence of which phase failed.

---

## 9. Extending Verification

If you add new modules:

- Add unit tests under `tests/`
- Update `scripts/verify.sh` to include new smoke checks
- Document any new outputs in `docs/data-contract.md`

---

## 10. Summary

The verification flow is intentionally minimal but comprehensive:

- ✅ Build environment
- ✅ Run pipeline
- ✅ Validate metrics
- ✅ Run tests
