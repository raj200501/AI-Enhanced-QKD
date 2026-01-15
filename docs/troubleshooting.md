# Troubleshooting

This guide lists common issues and fixes when running the AI-Enhanced QKD
pipeline.

---

## 1. Installation Issues

### Symptoms
- `ModuleNotFoundError: ai_qkd`

### Resolution
Re-run the bootstrap script and ensure `PYTHONPATH` is set when invoking Python:

```bash
./scripts/bootstrap.sh
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json run
```

This will recreate the virtual environment and install all dependencies.

---

## 2. Pipeline Fails to Run

### Symptoms
- CLI exits early
- `results/metrics.json` is missing

### Root Causes
- `data/raw/quantum_data.csv` missing or invalid
- `config/pipeline.json` missing or malformed

### Resolution
Regenerate configuration and rerun:

```bash
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json init-config
PYTHONPATH=src python -m ai_qkd.cli --config config/pipeline.json run
```

---

## 3. Metrics Threshold Failures

### Symptoms
- `scripts/verify.sh` fails with threshold assertions

### Root Causes
- Configuration drift
- Code changes that reduce model accuracy

### Resolution
Inspect the metrics and configuration:

```bash
cat results/metrics.json
cat config/pipeline.json
```

If the thresholds are too strict after a legitimate model improvement,
update `scripts/verify.sh` and `docs/data-contract.md` accordingly.

---

## 4. Pytest Failures

### Symptoms
- Assertion errors in tests
- Import errors

### Resolution
Run tests with verbose output:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Review the failing test and corresponding module. Ensure the pipeline
functions remain deterministic.

---

## 5. CI Failures

### Symptoms
- GitHub Actions red
- `verify.sh` exit code non-zero

### Resolution
Make sure:

- The workflow uses `./scripts/verify.sh`
- Local verification passes before pushing

---

## 6. File Permissions

### Symptoms
- `Permission denied` when running scripts

### Resolution
Restore executable permissions:

```bash
chmod +x scripts/*.sh
```

---

## 7. Cleaning the Workspace

If generated artifacts interfere with local runs:

```bash
rm -rf data/processed models results
```

Then rerun:

```bash
./scripts/run.sh
```

---

## 8. FAQ

### Q: Why does the pipeline avoid TensorFlow/PyTorch?
A: CI environments often lack GPU acceleration and installing heavy ML
frameworks introduces long setup times. The lightweight pipeline still
exercises the core logic without those dependencies.

### Q: Can I plug in my own models?
A: Yes. Replace the modules in `src/ai_qkd` and update tests accordingly.

### Q: Is the pipeline deterministic?
A: Yes. All random generation is seeded from the configuration.

---

## 9. Support

If you are a maintainer, check the following files for additional context:

- `docs/architecture.md`
- `docs/verification.md`
