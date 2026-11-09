| group | clients | quant | epsilon | xi | lambda_h | alpha | best_acc_% | best_round | total_GiB | comm_reduction_vs_fp32_% | minutes | finished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_dp_cifar10 | 100 | fp32 | 5000.0 | 100.0 | 0.75 | 0.5 | 73.68 | 710 | 24.267 | 0.0 | 120.7 | True |
| paper_dp_cifar10 | 100 | fp32 | 7500.0 | 100.0 | 0.75 | 0.5 | 75.35 | 950 | 24.267 | 0.0 | 118.6 | True |
| paper_dp_cifar10 | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 76.4 | 750 | 24.267 | 0.0 | 120.8 | True |
| paper_dp_cifar10 | 100 | fp32 | inf | 100.0 | 0.75 | 0.5 | 78.58 | 950 | 24.267 | 0.0 | 115.8 | True |
| paper_quant_cifar10 | 100 | fp32 | 10000.0 | 100.0 | 0.75 | 0.5 | 76.58 | 960 | 24.267 | 0.0 | 120.5 | True |
| paper_quant_cifar10 | 100 | int4 | 10000.0 | 100.0 | 0.75 | 0.5 | 51.03 | 60 | 3.033 | 87.5 | 121.2 | True |
| paper_quant_cifar10 | 100 | int8 | 10000.0 | 100.0 | 0.75 | 0.5 | 76.69 | 960 | 6.067 | 75.0 | 121.9 | True |
| paper_quant_cifar10 | 100 | int16 | 10000.0 | 100.0 | 0.75 | 0.5 | 76.81 | 960 | 12.133 | 50.0 | 121.4 | True |
| paper_quant_cifar10 | 100 | cosine | 10000.0 | 100.0 | 0.75 | 0.5 | 75.73 | 960 | 15.176 | 37.46 | 60.5 | True |
| paper_quant_cifar10 | 100 | dynamic | 10000.0 | 100.0 | 0.75 | 0.5 | 77.2 | 960 | 13.643 | 43.78 | 60.1 | True |
