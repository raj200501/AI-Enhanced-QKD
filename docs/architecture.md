# Architecture Overview

This document describes the architecture of the deterministic QKD pipeline.
The goal is to make the system easy to run in CI while still exercising a
meaningful sequence of steps:

1. **Data generation**
2. **Preprocessing**
3. **Anomaly detection**
4. **Error correction**
5. **Key distribution policy optimization**
6. **Evaluation and verification**

The system is intentionally minimal but complete. It avoids heavy ML frameworks
so that the project can be built and tested quickly on standard CI runners.

---

## 1. Data Generation

### Purpose
The data generation stage synthesizes QKD measurement events. Each event
represents a single measurement within a QKD session.

### Columns
Each event contains the following columns:

| Column           | Type   | Description |
|------------------|--------|-------------|
| `session_id`     | float  | Session identifier (integer-like). |
| `time_index`     | float  | Index within the session sequence. |
| `basis`          | float  | 0 or 1 basis choice. |
| `bit`            | float  | Raw bit measured. |
| `noise_level`    | float  | Simulated channel noise. |
| `channel_loss`   | float  | Simulated channel loss. |
| `detector_click` | float  | Whether a detector click occurred. |

### Determinism
The generator is seeded using a single integer (`data.seed`), so the output is
fully reproducible. This seed also influences downstream error-correction
simulation to keep test results deterministic.

---

## 2. Preprocessing

The preprocessing step:

- Normalizes noise and loss values into z-scores
- Concatenates the normalized columns with categorical features (`basis`,
  `detector_click`)
- Generates binary anomaly labels based on a fixed threshold

### Feature Vector
Each row becomes a 4-dimensional vector:

```
[noise_norm, loss_norm, basis, detector_click]
```

### Labeling
An anomaly label is assigned when:

```
noise_level + channel_loss > 0.3
```

This labeling choice makes the classification problem tractable while
remaining anchored in signal quality.

---

## 3. Anomaly Detection

Anomaly detection uses a lightweight logistic regression classifier trained via
batch gradient descent. This decision keeps the pipeline quick, deterministic,
and easy to validate without GPU dependencies.

### Training
- The model is trained on the preprocessed dataset
- Loss is tracked each epoch
- The final accuracy is reported and stored as part of pipeline metrics

### Model Format
The model is serialized as JSON:

```
{
  "weights": [...],
  "bias": 0.12
}
```

This format is human-readable and avoids binary artifacts.

---

## 4. Error Correction

Error correction is implemented as a rolling-window majority filter. The
training phase searches for a window size that minimizes bit-error rate (BER)
against clean reference sequences.

### Why a Window Filter?
In QKD post-processing, simple parity checks and majority logic are common
baseline approaches. The window filter captures the effect of smoothing noisy
bit sequences without implementing full error-correcting codes.

### Search
The pipeline evaluates a candidate set of window sizes (configured in
`config/pipeline.json`) and selects the size with the lowest BER.

---

## 5. Key Distribution Policy (RL-lite)

Instead of a full RL framework, the pipeline implements an epsilon-greedy
bandit policy. This still captures the iterative reward-feedback loop without
introducing heavy dependencies.

### Environment Model
A simplified environment returns rewards based on action quality:

- Actions 0 and 1 have higher base reward
- Actions 2 and 3 are noisier and less reliable

This reward model encourages the policy to converge toward more reliable
bases while still exploring.

---

## 6. Evaluation

The evaluation stage aggregates metrics from all components:

| Component         | Metric |
|------------------|--------|
| Anomaly detector | Accuracy, precision, recall |
| Error correction | Bit-error rate |
| Key distribution | Mean reward |

These metrics are written to `results/metrics.json` and used by
`./scripts/verify.sh` to confirm that the pipeline meets minimum thresholds.

---

## Reliability and Determinism

Every stage uses a seeded RNG to guarantee reproducibility. This provides a
clean baseline for verifying correctness and prevents CI flakiness.

---

## Extensibility

This architecture is deliberately modular. Each component can be replaced with
more advanced ML models if desired:

- Swap the logistic regression with a neural network
- Replace the bandit policy with a full RL agent
- Add explicit error-correcting codes

The CLI already supports these phases as discrete subcommands, making future
substitution straightforward.

---

## References (Conceptual)

- Bennett & Brassard (1984), BB84 Protocol
- Standard QKD post-processing pipelines
- Classical error correction baselines

---

## Glossary

- **QKD**: Quantum Key Distribution
- **BER**: Bit Error Rate
- **Basis**: Measurement basis selection
- **Anomaly**: A measurement event with unusually high noise or loss

---

## Summary

The architecture balances realism and practicality:

- ✅ Reproducible, deterministic results
- ✅ Verified end-to-end in CI
- ✅ Simple enough to run anywhere
- ✅ Structured for future extension

