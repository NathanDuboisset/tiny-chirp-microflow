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

Sources: `results/<model>_tf.jsonl` (latest record per file).
