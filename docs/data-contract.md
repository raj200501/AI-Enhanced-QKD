# Data Contract

This document defines the data formats for the AI-Enhanced QKD pipeline. It is
intended to be a single source of truth for all input/output files that the
pipeline reads or writes.

---

## 1. Raw Data (`data/raw/quantum_data.csv`)

### Purpose
Represents simulated QKD measurement events prior to preprocessing.

### Format
CSV with a header row. Each row corresponds to one measurement event.

### Columns

| Column           | Type  | Example | Description |
|------------------|-------|---------|-------------|
| `session_id`     | float | 12      | Session identifier. |
| `time_index`     | float | 7       | Index within session. |
| `basis`          | float | 0       | Basis choice (0/1). |
| `bit`            | float | 1       | Raw measured bit. |
| `noise_level`    | float | 0.12    | Simulated noise. |
| `channel_loss`   | float | 0.09    | Simulated loss. |
| `detector_click` | float | 1       | Detector click indicator. |

### Example Row

```
12,7,0,1,0.12,0.09,1
```

### Notes
- Values are stored as floats for compatibility with simple list-based math.
- All rows are generated deterministically based on `config/pipeline.json`.

---

## 2. Processed Dataset (`data/processed/*.csv`)

### Purpose
Preprocessed data splits for training and evaluation.

### Files

| File | Shape | Description |
|------|-------|-------------|
| `train_features.csv` | (N, 4) | Normalized features for training. |
| `train_labels.csv`   | (N,)   | Anomaly labels for training. |
| `val_features.csv`   | (N, 4) | Validation features. |
| `val_labels.csv`     | (N,)   | Validation labels. |
| `test_features.csv`  | (N, 4) | Test features. |
| `test_labels.csv`    | (N,)   | Test labels. |

### Feature Schema
Each row has 4 columns in order:

1. `noise_norm`
2. `loss_norm`
3. `basis`
4. `detector_click`

### Label Schema
Binary label where `1` indicates anomaly:

```
label = (noise_level + channel_loss) > anomaly_threshold
```

---

## 3. Anomaly Model (`models/anomaly/logistic_model.json`)

### Purpose
Stores the trained logistic regression parameters.

### Format

```json
{
  "weights": [0.12, -0.4, 0.03, 0.07],
  "bias": 0.18
}
```

### Notes
- `weights` length must match feature dimension (4).
- `bias` is a scalar float.

---

## 4. Error-Correction Model (`models/error_correction/window_model.json`)

### Purpose
Defines the selected rolling window size used for smoothing noisy bit
sequences.

### Format

```json
{
  "window_size": 3,
  "threshold": 0.5
}
```

### Notes
- `window_size` must be a positive integer.
- `threshold` should typically be 0.5 for majority voting.

---

## 5. Key Distribution Policy (`models/key_distribution/bandit_policy.json`)

### Purpose
Stores estimated action values for the epsilon-greedy bandit policy.

### Format

```json
{
  "action_values": [0.69, 0.71, 0.45, 0.48]
}
```

### Notes
- `action_values` length must equal `rl.actions` in config.
- Higher values indicate more reliable actions.

---

## 6. Metrics (`results/metrics.json`)

### Purpose
Aggregates performance metrics for the pipeline. This file is the basis for
CI verification.

### Format

```json
{
  "anomaly_detection": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.87
  },
  "error_correction": {
    "bit_error_rate": 0.14,
    "sequence_length": 32
  },
  "key_distribution": {
    "mean_reward": 0.72,
    "min_reward": 0.45,
    "max_reward": 0.91
  }
}
```

---

## 7. Configuration (`config/pipeline.json`)

### Purpose
Defines the deterministic parameters for the entire pipeline.

### Format

```json
{
  "data": {
    "seed": 1337,
    "sessions": 200,
    "sequence_length": 32,
    "noise_mean": 0.08,
    "noise_std": 0.03,
    "loss_mean": 0.12,
    "loss_std": 0.04
  },
  "anomaly": {
    "learning_rate": 0.4,
    "epochs": 120,
    "anomaly_threshold": 0.15
  },
  "error_correction": {
    "window_sizes": [1, 3, 5, 7]
  },
  "rl": {
    "actions": 4,
    "episodes": 200,
    "epsilon": 0.12,
    "alpha": 0.4
  }
}
```

---

## 8. Contract Validation Rules

The verification script (`scripts/verify.sh`) enforces the following:

- Anomaly detection accuracy ≥ 0.70
- Error correction BER ≤ 0.35
- Mean reward ≥ 0.55

These thresholds are calibrated to match deterministic pipeline behavior and
prevent regressions. The anomaly threshold itself is configured in
`config/pipeline.json`.

---

## 9. Versioning

Data formats are tied to the package version (`ai_qkd.__version__`). If you
change the data schema, update:

- `docs/data-contract.md`
- `src/ai_qkd/data.py`
- `src/ai_qkd/pipeline.py`
- `tests/` to validate new formats

---

## 10. Summary Checklist

- [x] Raw data format defined
- [x] Processed datasets defined
- [x] Model artifacts defined
- [x] Metrics format defined
- [x] Verification thresholds documented
