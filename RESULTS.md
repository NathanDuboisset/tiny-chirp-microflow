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

For reference scores, mel cnn tf model:
for samples 0 to 3, for testing, 
clip 0 : 1.8 / 2.9
clip 1 : -3.9 / 12.4 
clip 2 : 57.8 /-11.2
clip 3 : -6.5 / 10.6