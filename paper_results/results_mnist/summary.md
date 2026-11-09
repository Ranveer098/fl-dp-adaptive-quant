| group | clients | quant | epsilon | xi | lambda_h | alpha | best_acc_% | best_round | total_GiB | comm_reduction_vs_fp32_% | minutes | finished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| check_leaf | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 14.73 | 14 | 0.95 |  | 3.2 | True |
| fast_smoke_datmnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 10.98 | 0 | 0.402 |  | 1.0 | True |
| fast_smoke_datmnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 10.23 | 0 | 0.335 |  | 0.8 | True |
| paper_clip_mnist | 100 | fp32 | inf | 10.0 | 0.75 | 0.5 | 70.83 | 970 | 123.931 | 0.0 | 38.3 | True |
| paper_clip_mnist | 100 | fp32 | inf | 20.0 | 0.75 | 0.5 | 78.09 | 970 | 123.931 | 0.0 | 38.4 | True |
| paper_clip_mnist | 100 | fp32 | inf | 50.0 | 0.75 | 0.5 | 84.67 | 970 | 123.931 | 0.0 | 38.3 | True |
| paper_clip_mnist | 100 | fp32 | inf | 100.0 | 0.75 | 0.5 | 83.44 | 900 | 123.931 | 0.0 | 38.4 | True |
| paper_clip_mnist | 100 | fp32 | inf | inf | 0.75 | 0.5 | 98.85 | 970 | 123.931 | 0.0 | 38.2 | True |
| paper_dp_mnist | 100 | fp32 | 5000.0 | 100.0 | 0.75 | 0.5 | 75.38 | 920 | 123.931 | 0.0 | 39.1 | True |
| paper_dp_mnist | 100 | fp32 | 7500.0 | 100.0 | 0.75 | 0.5 | 76.77 | 970 | 123.931 | 0.0 | 39.1 | True |
| paper_dp_mnist | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 73.64 | 970 | 123.931 | 0.0 | 39.1 | True |
| paper_dp_mnist | 100 | fp32 | inf | 100.0 | 0.75 | 0.5 | 83.42 | 900 | 123.931 | 0.0 | 38.2 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.25 | 0.5 | 75.76 | 890 | 63.764 |  | 39.4 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.5 | 0.5 | 73.84 | 900 | 62.527 |  | 39.5 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 74.37 | 900 | 61.3 |  | 39.5 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 1.0 | 0.5 | 78.17 | 900 | 60.081 |  | 18.1 | True |
| paper_quant_mnist | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 71.2 | 600 | 123.931 | 0.0 | 39.1 | True |
| paper_quant_mnist | 100 | int4 | 10000.0 | 100.0 | 0.75 | 0.5 | 55.13 | 340 | 15.491 | 87.5 | 39.2 | True |
| paper_quant_mnist | 100 | int8 | 10000.0 | 100.0 | 0.75 | 0.5 | 72.95 | 870 | 30.983 | 75.0 | 39.3 | True |
| paper_quant_mnist | 100 | int16 | 10000.0 | 100.0 | 0.75 | 0.5 | 71.43 | 600 | 61.965 | 50.0 | 39.4 | True |
| paper_quant_mnist | 100 | cosine | 10000.0 | 100.0 | 0.75 | 0.5 | 73.33 | 890 | 77.503 | 37.46 | 39.3 | True |
| paper_quant_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 71.31 | 999 | 61.3 | 50.54 | 39.4 | True |
