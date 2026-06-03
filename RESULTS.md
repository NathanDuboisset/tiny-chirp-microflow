# Results

INT8-quantized TFLite models, evaluated on the test set. Threshold chosen for best F2 on the train split.

| Model | Size (KB) | Avg time (ms) | Accuracy | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| cnn_mel | 25.1 | TBD | 0.9777 | 0.9402 | 0.9956 | **0.9840** |
| sincnet_mimic_mel | 49.2 | TBD | 0.9777 | 0.9533 | 0.9803 | 0.9748 |
| sincnet_multi | 9.8 | TBD | 0.9598 | 0.9020 | 0.9847 | 0.9670 |
| cnn_time | 1.2 | TBD | 0.9282 | 0.8303 | 0.9825 | 0.9478 |
| sincnet | 15.5 | TBD | 0.8528 | 0.6920 | 0.9956 | 0.9153 |
| leaf | 23.2 | TBD | 0.5219 | 0.4063 | 0.9847 | 0.7665 |

cnn_mel scores (non_target / target) on the 4 testing clips. Delta columns are
signed relative error vs the host TF reference, `(value - ref) / |ref|`.

| clip | host TF      | mel_cpu          | delta mel_cpu    | mel_axon         | delta mel_axon   |
|:----:|-------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| 0    |  1.8 /  2.9  |   1.601 /  2.979 |  -11.1% /  +2.7% |   1.601 /  2.979 |  -11.1% /  +2.7% |
| 1    | -3.9 / 12.4  |  -4.101 /  9.335 |   -5.2% / -24.7% |  -3.978 /  8.870 |   -2.0% / -28.5% |
| 2    | 57.8 /-11.2  |  43.801 / -7.824 |  -24.2% / +30.1% |  39.421 / -9.571 |  -31.8% / +14.5% |
| 3    | -6.5 / 10.6  |  -5.857 /  7.424 |   +9.9% / -30.0% |  -5.210 /  6.275 |  +19.8% / -40.8% |

Per-clip mel extraction time: `mel_cpu` ≈2.62 s, `mel_axon` ≈0.94 s (3 s audio).