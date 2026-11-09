| group | clients | quant | epsilon | xi | lambda_h | alpha | best_acc_% | best_round | total_GiB | comm_reduction_vs_fp32_% | minutes | finished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_clip_mnist | 100 | fp32 | inf | 10.0 | 0.75 | 0.5 | 73.19 | 980 | 123.931 | 0.0 | 37.8 | True |
| paper_clip_mnist | 100 | fp32 | inf | 20.0 | 0.75 | 0.5 | 82.01 | 970 | 123.931 | 0.0 | 38.0 | True |
| paper_clip_mnist | 100 | fp32 | inf | 50.0 | 0.75 | 0.5 | 89.07 | 999 | 123.931 | 0.0 | 37.8 | True |
| paper_clip_mnist | 100 | fp32 | inf | 100.0 | 0.75 | 0.5 | 90.39 | 810 | 123.931 | 0.0 | 38.0 | True |
| paper_clip_mnist | 100 | fp32 | inf | inf | 0.75 | 0.5 | 98.78 | 760 | 123.931 | 0.0 | 37.5 | True |
| paper_dp_mnist | 100 | fp32 | 5000.0 | 100.0 | 0.75 | 0.5 | 83.07 | 999 | 123.931 | 0.0 | 38.8 | True |
| paper_dp_mnist | 100 | fp32 | 7500.0 | 100.0 | 0.75 | 0.5 | 79.56 | 940 | 123.931 | 0.0 | 38.8 | True |
| paper_dp_mnist | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 79.23 | 810 | 123.931 | 0.0 | 38.7 | True |
| paper_dp_mnist | 100 | fp32 | inf | 100.0 | 0.75 | 0.5 | 90.39 | 810 | 123.931 | 0.0 | 37.5 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.25 | 0.5 | 78.17 | 160 | 63.764 |  | 39.2 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.5 | 0.5 | 79.05 | 999 | 62.527 |  | 39.0 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 78.17 | 160 | 61.3 |  | 38.9 | True |
| paper_lambda_mnist | 100 | dynamic | 10000.0 | 100.0 | 1.0 | 0.5 | 80.74 | 770 | 60.081 |  | 18.2 | True |
| paper_quant_mnist | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 81.62 | 970 | 123.931 | 0.0 | 38.7 | True |
| paper_quant_mnist | 100 | int4 | 10000.0 | 100.0 | 0.75 | 0.5 | 73.06 | 970 | 15.491 | 87.5 | 39.0 | True |
| paper_quant_mnist | 100 | int8 | 10000.0 | 100.0 | 0.75 | 0.5 | 78.2 | 970 | 30.983 | 75.0 | 39.1 | True |
| paper_quant_mnist | 100 | int16 | 10000.0 | 100.0 | 0.75 | 0.5 | 79.05 | 970 | 61.965 | 50.0 | 38.9 | True |
| paper_quant_mnist | 100 | cosine | 10000.0 | 100.0 | 0.75 | 0.5 | 78.47 | 999 | 77.503 | 37.46 | 38.9 | True |
| paper_quant_mnist | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 81.41 | 970 | 61.3 | 50.54 | 39.2 | True |
