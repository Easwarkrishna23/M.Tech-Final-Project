# Cora Dataset — Final Attack & Defense Results

**Baseline:** acc=0.8010

## Attack Impact

| Attack | Type | Accuracy | F1 | Drop |
| --- | --- | --- | --- | --- |
| nettack | Poisoning | 0.7780 | 0.7685 | +0.0230 |
| dice | Poisoning | 0.7480 | 0.7442 | +0.0530 |
| meta_attack | Poisoning | 0.7990 | 0.7924 | +0.0020 |
| random_structure | Poisoning | 0.7540 | 0.7473 | +0.0470 |
| feature_perturbation | Evasion | 0.4190 | 0.1784 | +0.3820 |
| edge_flip | Evasion | 0.7360 | 0.7248 | +0.0650 |
| gradient_attack | Evasion | 0.0000 | 0.0000 | +0.8010 |

## Defense Performance

| Attack | After Attack | After Defense | Recovery Rate |
| --- | --- | --- | --- |
| nettack | 0.7780 | 0.7750 | N/A |
| dice | 0.7480 | 0.7340 | -26.4% |
| meta_attack | 0.7990 | 0.7910 | N/A |
| random_structure | 0.7540 | 0.7300 | N/A |
| feature_perturbation | 0.4190 | 0.7390 | 83.8% |
| edge_flip | 0.7360 | 0.7290 | -10.8% |
| gradient_attack | 0.0000 | 0.9210 | 115.0% |