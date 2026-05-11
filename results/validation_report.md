# Validation Report — Phase 1–3

**Date:** 2026-05-10  
**Project:** Adversarial Attacks and Structural Defense in GNNs for Node Classification

---

## 1. Data Validation

### Cora (Static Dataset)

| Check | Result |
|---|---|
| Nodes | 2,708 |
| Edges | 5,278 |
| Features | 1,433 (binary BoW, range [0, 1]) |
| Classes | 7 |
| Feature dtype | float32 |
| NaN in features | None |
| NaN in adj_norm | None |
| Adjacency symmetric | ✅ True |
| Self-loops in adj | 0 (added implicitly via A+I in normalisation) |
| Train ∩ Val overlap | 0 |
| Train ∩ Test overlap | 0 |
| Train / Val / Test split | 140 / 500 / 1000 |

**Label distribution:**

| Class | Nodes | Share |
|---|---|---|
| 0 | 351 | 13.0% |
| 1 | 217 | 8.0% |
| 2 | 418 | 15.4% |
| 3 | 818 | 30.2% |
| 4 | 426 | 15.7% |
| 5 | 298 | 11.0% |
| 6 | 180 | 6.6% |

---

### Elliptic Bitcoin (Dynamic Dataset — 49 Timesteps)

| Check | Result |
|---|---|
| Total timesteps | 49 |
| Nodes/snapshot (avg) | 4,158 |
| Edges/snapshot (avg) | 4,782 |
| Features | 165 per node |
| NaN in features | None |
| NaN in adj_norm | None |
| Adjacency symmetric | ✅ True (symmetrised from directed graph) |
| Train ∩ Test overlap | 0 |

**Label convention:** `0` = licit, `1` = illicit, `-1` = unknown (excluded from metrics)

**Final snapshot (t=49) label distribution:**

| Label | Nodes | Share |
|---|---|---|
| licit (0) | 420 | 17.1% |
| illicit (1) | 56 | 2.3% |
| unknown (−1) | 1,978 | 80.6% |

**Class imbalance across timesteps (illicit %):**

| Timestep | Labeled Nodes | Illicit % |
|---|---|---|
| t=01 | 2,147 | 0.8% |
| t=10 | 972 | 1.9% |
| t=25 | 594 | 19.9% |
| t=35 | 1,341 | 13.6% |
| t=49 | 476 | 11.8% |
| **avg** | — | **11.3%** |

---

## 2. Training Validation

### GCN on Cora

| Epoch | Loss | Val Acc |
|---|---|---|
| 1 | 1.9478 | 0.5460 |
| 6 | 0.8257 | 0.7860 |
| 11 | 0.2096 | 0.7920 |
| 21 | 0.0133 | 0.7680 |
| 28 (stopped) | 0.0064 | 0.7860 |

- Loss decreases from 1.95 → 0.006 — healthy exponential decay ✅
- 3/27 non-monotone steps — normal stochasticity from dropout ✅
- Early stopping triggered at epoch 28 (patience=20) ✅
- Best val acc: **0.7980** at epoch 8

### GCN on Elliptic (t=49)

- Early stopping triggered at epoch 44 ✅
- Best val acc: **0.9684** at epoch 24

---

## 3. Accuracy Validation

### Cora — 7-Class Node Classification

| Metric | Value |
|---|---|
| Accuracy | **0.8010** |
| Precision (macro) | 0.7781 |
| Recall (macro) | 0.8195 |
| F1 (macro) | 0.7930 |

**Per-class accuracy on test set:**

| Class | Test Nodes | Accuracy |
|---|---|---|
| 0 | 130 | 0.800 |
| 1 | 91 | 0.912 |
| 2 | 144 | 0.931 |
| 3 | 319 | 0.734 |
| 4 | 149 | 0.765 |
| 5 | 103 | 0.767 |
| 6 | 64 | 0.828 |

> Class 3 (largest class, 30% of nodes) has the lowest accuracy at 73.4% — consistent with GCN literature on Cora.

**Literature comparison:**  
Known GCN baseline on Cora: ~81.5% — our result of **80.1% is within expected range** ✅

---

### Elliptic — Binary Fraud Classification

| Metric | Value |
|---|---|
| Accuracy | **0.9688** |
| F1 (macro) | 0.9354 |
| Precision (macro) | 0.9000 |
| Recall (macro) | 0.9821 |

**Majority-class baseline check** (always predict licit):

| | Accuracy |
|---|---|
| Majority-class baseline | 0.8750 |
| GCN | **0.9688** |
| Improvement over baseline | +9.4pp |

> GCN significantly outperforms the majority-class baseline. The high accuracy is **not** due to class collapse. ✅

**Per-class accuracy on test set:**

| Class | Test Nodes | Accuracy |
|---|---|---|
| licit (0) | 84 | 0.964 |
| illicit (1) | 12 | **1.000** |

**Prediction distribution:**

| | Predicted licit | Predicted illicit |
|---|---|---|
| True licit | 81 | 3 |
| True illicit | 0 | 12 |

> Illicit recall = 100% — all 12 fraudulent transactions in the test set are correctly identified. ✅

**Probability sanity check:**  
Max |sum(probs) − 1| = 1.19e-07 (float32 precision) ✅  
NaN in probabilities: None ✅

---

### Optional GAT Comparison (Cora)

| Model | Accuracy | F1 |
|---|---|---|
| GCN | 0.8010 | 0.7930 |
| GAT | **0.8130** | **0.8062** |

GAT outperforms GCN by +1.2pp on Cora — consistent with published results (GAT typically ~83%).

---

## 4. Summary

| Check | Status |
|---|---|
| No NaN in features or adjacency | ✅ |
| Masks non-overlapping | ✅ |
| Adjacency symmetric | ✅ |
| Loss converges | ✅ |
| Early stopping functional | ✅ |
| Cora accuracy within literature range | ✅ |
| Elliptic beats majority-class baseline | ✅ |
| No class collapse on Elliptic | ✅ |
| Checkpoints saved | ✅ |

**All Phase 1–3 checks passed. Ready for Phase 4 (Attack Module).**
