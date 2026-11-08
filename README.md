# Reproducing: Federated Learning with Laplacian DP + Adaptive Quantization under Non-IID Data

A self-contained PyTorch reproduction of

> E. Ardıç and Y. Genç, *"Enhanced Privacy and Communication Efficiency in Non-IID Federated Learning with
> Adaptive Quantization and Differential Privacy,"* IEEE Access, vol. 13, 2025. DOI 10.1109/ACCESS.2025.3554138 · arXiv:2604.23426

The authors' official code is [eardic/FL_DPQS](https://github.com/eardic/FL_DPQS), a modified copy of the whole FedML
framework (Python 3.6, PyTorch 1.8.2 + CUDA 11.1, Windows `.bat` scripts). This repo re-implements the same algorithm
in about 800 lines of plain PyTorch. **Wherever the paper and the official code disagree, this repo follows the
official code** (section 5 lists every case). It keeps all data in memory and adds **checkpoint/resume** so
Colab/Kaggle timeouts don't lose work.

---

## 1. What is implemented (paper → code)

| Paper | What | Code |
|---|---|---|
| Alg. 1 | FedAvg server/client loop, weighted aggregation (Eq. 2, line 14) | `fl/federated.py: run()` |
| Sec. III-B | MNIST: 1000 users × 2 digits, merged into N clients (user u → client u mod N) · CIFAR10: Dirichlet(α=0.5) | `fl/data.py` |
| (code only) | CIFAR10 training augmentation of the official code | `fl/augment.py` |
| Sec. III-C, Fig. 2–3 | 2-layer CNN (1.66M params) · VGG7 (325K params) | `fl/models.py` |
| Eq. 3–7 | Symmetric per-tensor stochastic uniform quantization | `fl/quant.py: stochastic_quantize` |
| Eq. 8 | Cosine-annealed bit-length (32 → b_min) | `fl/quant.py: cosine_bits` |
| Eq. 9 | Client importance ν = λ_h·(normalized Shannon entropy) + (1−λ_h)·n_i/n_max | `fl/quant.py: client_importance` |
| Alg. 1 l.21 | L1 gradient clipping (ξ) | `fl/dp.py: clip_grad_l1_` |
| Eq. 12 | Per-client L1 sensitivity (3 cases, E0) | `fl/dp.py: l1_sensitivity` |
| Eq. 13 | Local Lipschitz estimate λ_i (per batch, across epochs, max) | `fl/dp.py: LipschitzTracker` |
| Alg. 1 l.25 | Laplace noise, scale = (PT/(NE))·Ξ/ε | `fl/dp.py: laplace_scale, add_laplace_noise_` |
| Table 6 | Total communicated GiB (up + down, weights + BN stats) | `fl/quant.py: payload_bits` |

Quantization modes (`--quant`): `fp32`, `int4`, `int8`, `int16` (static), `cosine` (global scheduler, ν=1),
`dynamic` (cosine at the server, cosine × ν at clients).

The communication-cost formula reproduces the paper's numbers exactly (MNIST FP32, 50 clients → **61.97 GiB**;
CIFAR10 → **12.13 GiB**). This is checked in `tests/test_core.py`.

---

## 2. Setup (Ubuntu / WSL, CPU laptop)

```bash
git clone https://github.com/YOUR_USERNAME/fl-dp-adaptive-quant.git
cd fl-dp-adaptive-quant
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only build (small download)
pip install numpy matplotlib pytest
python -m pytest -q                      # expect: 10 passed
```

MNIST and CIFAR10 download automatically into `./data` on the first run.

**Optional, for the exact MNIST data of the paper.** The authors use the LEAF/FedProx MNIST
(1000 real users, 61,664 train / 7,371 test). Download and unzip it once:

```bash
mkdir -p data && cd data
wget https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip && unzip MNIST.zip   # -> data/MNIST/train, data/MNIST/test
cd ..
```

Then add `--mnist_source leaf` to `run.py`, or `--extra "--mnist_source leaf"` to `run_suite.py`. The first run
converts the JSON to a fast `.npz` cache (needs about 3 GB of RAM once). Without it, the code builds a synthetic
version of the same structure from torchvision MNIST.

---

## 3. Why your earlier runs took hours (and how this repo fixes it)

Reading the official code shows several costs on top of the long schedule:

* every 10 rounds it evaluates the model on the **training data of all clients** (a full pass over the dataset), plus the test set;
* with DP on, it stores a copy of all weights **and** gradients for **every batch of every epoch** to estimate λ;
* CIFAR10 images go through a per-image PIL/tensor augmentation in a DataLoader;
* quantization loops over conv filters in Python and also computes the quantization error every time;
* the label entropy is recomputed every round by iterating the client's whole DataLoader;
* the config enables **Weights & Biases logging with the authors' API key** (`enable_wandb: true`). If you ever run
  their code, set `enable_wandb: false` first, otherwise your runs are uploaded to their account or stall on network calls.

This repo avoids all of these. On top of that, the paper's protocol is **T = 1000 rounds × E = 5 local epochs** with 10% of clients per round. Every round then
processes about 10% of the training set 5 times, so one run is about **500 passes over the dataset**.
The paper reports roughly 60 such runs on a GPU. On a laptop CPU, a single run taking 2–3 hours is expected,
and Colab's free tier disconnects before a run finishes.

This repo addresses that in three ways:

1. **Presets.** The algorithm is the same; only the schedule is shorter.

   | preset | rounds T | local epochs E | ≈ dataset passes | use it for |
   |---|---|---|---|---|
   | `tiny`  | 60   | 1 | 6   | first look, debugging |
   | `fast`  | 150  | 2 | 30  | **laptop CPU, today** (≈10–25 min/run, check the ETA printed in the log) |
   | `paper` | 1000 | 5 | 500 | GPU (Kaggle), final numbers for the paper |

   With E = 1 (`tiny`) the Lipschitz estimate of Eq. 13 needs two epochs, so λ = 0 and the first case of Eq. 12 is used.
   Note: the noise scale depends on T and E through T_i = PT/(NE). With a shorter schedule, the same ε adds
   **less** noise. Compare *trends* between `fast` runs, and compare to the paper's absolute numbers only with `paper` runs.
2. **Speed.** Data stays as in-memory tensors (no DataLoader), one model object is reused for all clients, and
   several runs execute in parallel (`--jobs`).
3. **Resume.** A checkpoint is written at every evaluation. If the process dies (Colab/Kaggle timeout, laptop
   sleep), run the same command again and it continues where it stopped. `--max_minutes 300` stops cleanly before
   a session limit.

---

## 4. Running

```bash
# smoke test (1–2 min): prints accuracy, GiB, bit-lengths, lambda, noise scale, s/round and ETA
python scripts/run_suite.py --suite smoke --preset fast
cat results/logs/*smoke*.log

# list all experiment groups
python scripts/run_suite.py --list

# reproduce a table on the laptop (2 runs at a time; finished runs are skipped on re-launch)
python scripts/run_suite.py --suite quant_mnist --preset fast --jobs 2   # Table 6 / Fig. 8
python scripts/run_suite.py --suite dp_mnist    --preset fast --jobs 2   # Table 5 / Fig. 5
python scripts/run_suite.py --suite clip_mnist  --preset fast --jobs 2   # Table 3

# tables + figures
python scripts/plot.py      # -> results/summary.md, results/summary.csv, results/figures/*.png
```

Single run with any setting:

```bash
python run.py --dataset cifar10 --n_clients 100 --quant dynamic --lambda_h 0.75 --bmin 8 \
              --epsilon 1e4 --xi 100 --rounds 1000 --local_epochs 5
python run.py --help   # every option
```

Use `--epsilon inf` for no DP noise and `--xi inf` for no clipping.

| suite | runs | paper reference |
|---|---|---|
| `clip_<ds>` | ξ ∈ {10, 20, 50, 100, ∞}, no noise | Table 3 |
| `dp_<ds>` | ε ∈ {∞, 5e3, 7.5e3, 1e4}, ξ = 100 | Table 5, Fig. 5 |
| `quant_<ds>` | fp32 / int4 / int8 / int16 / cosine / dynamic, ε = 1e4 | Table 6, Fig. 8 |
| `lambda_<ds>` | λ_h ∈ {0.25, 0.5, 0.75, 1.0} | Fig. 7 |
| `clients_<ds>` | N ∈ {50, 100, 200} | Table 4 |
| `noniid_<ds>` | **extension**: Dirichlet α ∈ {0.1, 0.5, 1, 100} × {no DP, DP, DP + dynamic} | — |

`<ds>` is `mnist` or `cifar10`. `all_mnist` / `all_cifar10` = clip + dp + quant + lambda.

### Kaggle (free GPU, runs in the background)

Open `notebooks/kaggle_paper_protocol.ipynb` on kaggle.com → *File → Import Notebook*. Set *GPU* and *Internet On*,
replace `YOUR_USERNAME`, then **Save Version → Save & Run All (Commit)**. Unlike Colab, a committed Kaggle notebook
keeps running after you close the browser. Download `results.zip` from the *Output* tab when it finishes.

---

## 5. Paper vs. official code: every difference we found

This repo follows the **official code** in each case (the paper's version is noted). These are also strong material
for your reproduction section and critical evaluation.

| # | Topic | Paper says | Official code does (and this repo) |
|---|---|---|---|
| 1 | Gradient clipping (Alg. 1 l.21) | `∇f / max(‖∇f‖₁, ξ)`, which would stop learning at ξ = ∞ | standard L1 clipping `clip_grad_norm_(params, ξ, norm_type=1)` |
| 2 | Weight decay | 0.001 | **not used**: the SGD optimizer is created without `weight_decay` (`--weight_decay` default 0) |
| 3 | Batch size, MNIST with N < 1000 | 64 | merged clients are trained as **one full batch**, so each local epoch is one SGD step (`--batch_mode fixed` for the paper's version) |
| 4 | Data augmentation (CIFAR10) | not mentioned | one random op per image: crop / flip / cutout / colour jitter / rotation 45° / affine (`fl/augment.py`, `--augment none` to disable) |
| 5 | Input normalization | not mentioned | images in [0, 1], no mean/std normalization |
| 6 | Quantization granularity | "each 3-D tensor in each layer" | one scale per conv output filter (`per_batch`), one per tensor otherwise |
| 7 | Cosine schedule | Eq. 8: T; Alg. 1: T − 1 | T |
| 8 | Gradient used for λ_i (Eq. 13) | not specified | the **clipped** gradient, pre-step weights, consecutive epochs, max over batches |
| 9 | E0 in Eq. 12 | smallest integer | real value log(1+n)/log(1+λη) |
| 10 | What gets DP noise | "model weights" | every state_dict entry, incl. BatchNorm running statistics |
| 11 | Client sampling | random | `np.random.seed(round)`, so every experiment sees the same clients in round t |
| 12 | Evaluation | every 10 rounds | rounds 0, 10, 20, … and the last; "Best Round" is 0-based (e.g. 999) |
| 13 | VGG7 | Fig. 3 | an extra max-pool after the 4th conv (8×8 → 4×4) before the 2×2 average pool |
| 14 | Dynamic weights in the shipped script | λ_h = 0.75 (Table 6) | `.bat` file uses 0.5 / 0.5 (we default to 0.75 as in the paper; set `--lambda_h 0.5` to match the script) |

Two further observations for the critical evaluation:

* **Sensitivity blow-up.** If λ_i is large enough that E ≥ E0, Eq. 12 gives Ξ ≈ 2ξ = 200 for ξ = 100, i.e. noise
  about 1000× larger than the λ = 0 case. Watch the `lambda~` and `noise~` columns in the logs; compare with
  `--lipschitz zero`. This may explain the "convergence failures" the paper reports at small ε.
* **Memory.** The official λ estimate keeps E × (number of batches) model copies per client; this repo keeps only the
  previous epoch, which gives the same value. `--lip_batches` can limit the tracked batches further (the `fast` preset tracks 4).

---

## 6. Extension directions (for the conference version)

The suites above are the reproduction. These directions turn it into a contribution that fits the title
*"Privacy-Preserving Federated Learning Using Differential Privacy under Non-IID Data"*:

- **Meaningful privacy budgets.** ε = 10⁴ bounds the privacy loss only by e^10000, which is no formal protection.
  Report what accuracy is reachable at ε ∈ [1, 10] with proper composition accounting, e.g. Gaussian
  noise + Rényi DP via Opacus, versus the paper's Laplace mechanism. This is the most important critique.
- **Non-IID severity.** The `noniid_<ds>` suite (α from 0.1 to IID) shows how DP noise and quantization interact
  with label skew. The paper fixes α = 0.5.
- **The bit-length is a side channel.** In `dynamic` mode each client's upload precision depends on its label
  entropy and dataset size (Eq. 9). The server can therefore infer how skewed and how large a client's data is
  just from the bit-length. Measure this leakage, then privatize ν (e.g. add Laplace noise to the entropy score
  before choosing b_i).
- **Empirical attacks.** Membership inference or gradient-inversion attacks with and without DP/quantization show
  how much protection is real at each ε.
- **Non-IID-robust optimizers.** Use FedProx or SCAFFOLD under the same DP + quantization pipeline.

---

## 7. Repository layout

```
fl/data.py         datasets in memory (torchvision or LEAF MNIST), non-IID partitions, label entropy
fl/augment.py      batched version of the official CIFAR10 augmentation
fl/models.py       MNIST CNN, VGG7
fl/quant.py        stochastic quantization, cosine / dynamic schedulers, payload size
fl/dp.py           L1 clipping, Lipschitz tracking, sensitivity, Laplace noise
fl/federated.py    Algorithm 1 + logging + checkpoint/resume
run.py             one experiment from the command line
scripts/run_suite.py   experiment grids, parallel runner, skip-if-done
scripts/plot.py        summary.md / summary.csv / figures
notebooks/kaggle_paper_protocol.ipynb
tests/test_core.py     equation-level unit tests
```
