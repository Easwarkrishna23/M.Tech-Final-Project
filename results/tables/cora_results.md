# Cora Dataset — Final Attack & Defense Results

**Baseline:** acc=0.8010

## Attack Impact

| Attack | Type | Accuracy | F1 | Drop |
| --- | --- | --- | --- | --- |
| nettack | Poisoning | 0.6810 | 0.6370 | +0.1200 |
| dice | Poisoning | 0.6980 | 0.6866 | +0.1030 |
| meta_attack | Poisoning | 0.7870 | 0.7810 | +0.0140 |
| random_structure | Poisoning | 0.7280 | 0.7176 | +0.0730 |
| feature_perturbation | Evasion | 0.4190 | 0.1784 | +0.3820 |
| edge_flip | Evasion | 0.6360 | 0.6320 | +0.1650 |
| gradient_attack | Evasion | 0.0000 | 0.0000 | +0.8010 |

## Defense Performance

| Attack | After Attack | After Defense | Recovery Rate |
| --- | --- | --- | --- |
| nettack | 0.6810 | 0.6770 | -3.3% |
| dice | 0.6980 | 0.7070 | 8.7% |
| meta_attack | 0.7870 | 0.7840 | N/A |
| random_structure | 0.7280 | 0.7310 | 4.1% |
| feature_perturbation | 0.4190 | 0.7730 | 92.7% |
| edge_flip | 0.6360 | 0.6380 | 1.2% |
| gradient_attack | 0.0000 | 0.9070 | 113.2% |