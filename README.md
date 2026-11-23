# ChaosFormer: Hardware-Bound Edge Transformer Model Obfuscation for IP Protection (ACM TODAES)

Code for ChaosFormer, a hardware-bound obfuscation framework that protects the intellectual property of transformer models (vision, language, and speech) deployed on edge devices. A Physical Unclonable Function (PUF) roots a device-specific key hierarchy, and the derived keys encrypt the model weights: the Arnold Cat Map (ACM) permutes the attention matrices, a Knuth shuffle permutes the Feed-Forward Network (FFN) matrices, and `ChaosFormer-S` additionally diffuses every protected tensor with a ChaCha20 keystream. A copied checkpoint runs correctly only on the authorized device.

## Overview

The repository implements two schemes:

- **`ChaosFormer` (base):** Arnold Cat Map (ACM) permutation of the attention weights and Knuth-shuffle permutation of the FFN weights
- **`ChaosFormer-S` (secure):** the base permutations followed by ChaCha20 bit diffusion of every protected tensor

Both work on ViT, DeiT, BEiT, RoBERTa, OPT, and Wav2Vec 2.0 checkpoints from the HuggingFace Hub.

## Installation

### Prerequisites

- CUDA-capable GPU (RTX 4090 and A100 tested)
- Around 200GB Storage (~160GB for ImageNet dataset; misc downstream attack datasets and protected models)

### Method 1: Setup with uv (Recommended)

[uv](https://docs.astral.sh/uv/) provides fast, reliable dependency management and automatic virtual environment handling.

1. **Install uv** (if not already installed):

```bash
# On macOS and Linux, recommended
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

2. **Clone and setup the project**:

```bash
git clone https://github.com/sarendis56/ChaosFormer-IP-Protection.git
cd ChaosFormer-IP-Protection

# Create virtual environment and install all dependencies
uv sync
```

3. **Run experiments**:

```bash
# Run basic mode encryption with Top-K strategy
uv run src/experiments/vit_encryption_experiment.py --mode basic --strategy top-k --k 4

# Run advanced mode encryption
uv run src/experiments/vit_encryption_experiment.py --mode advanced
```

### Method 2: Setup with venv + pip

If you prefer traditional Python virtual environments:

1. **Clone the repository**:

```bash
git clone https://github.com/sarendis56/ChaosFormer-IP-Protection.git
cd ChaosFormer-IP-Protection
```

2. **Create and activate virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run experiments**:

```bash
# Run basic mode encryption with Top-K strategy
python src/experiments/vit_encryption_experiment.py --mode basic --strategy top-k --k 4

# Run advanced mode encryption
python src/experiments/vit_encryption_experiment.py --mode advanced
```

### Dataset Setup

Download ImageNet dataset and place it under `dataset/imagenet`. The validation set should be organized as a flat directory structure under `dataset/imagenet/val_nolabel/` with images named sequentially (e.g., `ILSVRC2012_val_00000001.JPEG`, `ILSVRC2012_val_00000002.JPEG`, etc.). This differs from the official ImageNet validation structure which organizes images in synset subdirectories. The ground truth labels are provided through separate mapping files (`src/attacks/synset_words.txt` and `src/attacks/imagenet_2012_validation_synset_labels.txt`). Update the `imagenet_path` in configuration files as needed. Models will be automatically downloaded from HuggingFace Hub on first use.

The partial-key, layer-subset, and CKA experiments instead read the HuggingFace parquet release of ImageNet-1k (`data/imagenet-1k/data/{train,validation}-*.parquet`); pass its root with `--data-dir`.

## Quick Start

### Main Encryption Scheme

Run a ViT-base encryption experiment with different modes and strategies:

**With uv:**

```bash
# Basic mode with Top-K strategy (K=4)
uv run src/experiments/vit_encryption_experiment.py --mode basic --strategy top-k --k 6 --device cuda:0

# Basic mode with Random-K strategy (K=4)
uv run src/experiments/vit_encryption_experiment.py --mode basic --strategy random-k --k 6 --device cuda:1

# Basic mode with Last-K strategy (K=4)
uv run src/experiments/vit_encryption_experiment.py --mode basic --strategy last-k --k 6 --device cuda:2

# Secure mode (all layers protected with ChaCha20 bit diffusion)
uv run src/experiments/vit_encryption_experiment.py --mode advanced --device cuda:3

# Avalanche-effect experiment (random-k with key perturbations)
# Demonstrates high accuracy with the correct key and near-random accuracy with slightly incorrect keys
uv run src/experiments/vit_avalanche_experiment.py --k 6 --device cuda:0
uv run src/experiments/vit_avalanche_experiment.py \
  --model google/vit-base-patch16-224 \
  --device cuda \
  --k 6 \
  --bit-flips 1,2,4,6,8,10,12,16 \
  --samples-per-distance 16 \
  --hamming1-extra-samples 36 \
  --noise-levels 0.0001,0.0003,0.0007,0.0015,0.003,0.006,0.01,0.02 \
  --noise-samples 16

# Replot only from an existing results file (regenerates both PNG and PDF)
uv run src/experiments/vit_avalanche_experiment.py \
  --replot \
  --replot-json results/vit_avalanche_random-k_k6_YYYYMMDD_HHMMSS/avalanche_results.json
```

### Attack Training

Evaluate the robustness of encrypted models by training on a subset of ImageNet:

**With uv:**

```bash
# Attack the latest Top-K encrypted model with 20% of ImageNet
uv run src/attacks/train.py --setup basic_top-k_k6 --rate 0.2 --epoch 20 --device cuda:0
uv run src/attacks/train.py --setup basic_last-k_k6 --rate 0.2 --epoch 20 --device cuda:1
uv run src/attacks/train.py --setup basic_random-k_k6 --rate 0.2 --epoch 20 --device cuda:2

# Attack the latest Advanced mode model with 30% of ImageNet
uv run src/attacks/train.py --setup advanced_all --rate 0.3 --epoch 20 --device cuda:3

# Attack with random initialization (baseline comparison)
uv run python src/attacks/train.py \
  --setup advanced_all \
  --model "google/vit-base-patch16-224" \
  --rand_init \
  --device cuda:1
  
uv run src/attacks/train.py --setup advanced_all --model "facebook/deit-small-patch16-224" --rand_init --device cuda:2

uv run src/attacks/train.py --setup advanced_all --model "google/vit-base-patch16-224" --device cuda:2
```

### Downstream Fine-tuning (Attack Models)

Evaluate the transferability of retrained models to downstream tasks. After running attack training (see [Attack Training](#attack-training)), you can fine-tune the resulting models on various downstream datasets.

**Finding checkpoint paths**: Attack models are saved in `results/attacks/` with names following the pattern:

- `attack_{setup}_{rate}_{timestamp}` for attack-trained models
- `randinit_{setup}_{rate}_{timestamp}` for randomly initialized baselines

You can list available checkpoints with:

```bash
ls results/attacks/
```

**Examples**:

```bash
# Fine-tune an attack-trained model on ImageNet-100
# Replace <checkpoint_path> with your actual checkpoint directory from results/attacks/
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset imagenet100 --epoch 20 --device cuda:0

# Fine-tune on CIFAR-100
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset cifar100 --epoch 20 --device cuda:0

# Fine-tune on Caltech-256
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset caltech256 --epoch 20 --device cuda:0

# Fine-tune on CUB-200-2011
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset cub200 --epoch 20 --device cuda:0

# Fine-tune on OfficeHome (domain adaptation: source -> target)
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset officehome --officehome_source product --officehome_target realworld --epoch 20 --device cuda:0

# Fine-tune on DomainNet-Clipart
uv run src/attacks/fine_tune.py --model_path results/attacks/<checkpoint_path> --dataset domainnet_clipart --domainnet_domain clipart --epoch 20 --device cuda:0

# Baseline: Fine-tune the original (unencrypted) model for comparison
uv run src/attacks/fine_tune.py --model_path original --dataset imagenet100 --epoch 20 --device cuda:0
```

**Batch evaluation**: To evaluate multiple models across multiple datasets, you can chain commands or use shell loops:

```bash
# Example: Evaluate multiple attack models on ImageNet-100
for checkpoint in results/attacks/attack_*; do
    uv run src/attacks/fine_tune.py --model_path "$checkpoint" --dataset imagenet100 --epoch 20 --device cuda:0
done
```

### Partial-Key Recovery

Clean ImageNet accuracy (all 50,000 validation images) and the K=6 accuracy
when only the attention (ACM) keys or only the FFN keys are recovered:

```bash
uv run python -m src.experiments.vision_partial_key_eval --data-dir data/imagenet-1k --device cuda:0
```

### Retraining Attacks on RoBERTa

The attacker fine-tunes a protected RoBERTa classifier with 20% of the training
data, starting from the encrypted checkpoint (`base_k6`, `secure_all`) or after
re-initializing the encrypted matrices (`base_k6_reinit`, `secure_all_reinit`).
Baselines: `origin`, `public`, `randinit`.

```bash
# 20 Newsgroups victim (fine-tuned roberta-base, full training set)
uv run python -m src.experiments.roberta_retraining_attack --task 20ng --setup victim --lr 3e-5 --epochs 10 --batch-size 16
# One attack run; repeat over --lr {1e-5,3e-5,1e-4} and --seed {0,1,2}
uv run python -m src.experiments.roberta_retraining_attack --task 20ng --setup secure_all_reinit --lr 1e-4 --seed 0 --epochs 10 --batch-size 16
uv run python -m src.experiments.roberta_retraining_attack --task sst2 --setup base_k6_reinit --lr 3e-5 --seed 0
```

### SRAM-PUF Key Reconstruction

Fuzzy-extractor reliability, entropy accounting, and reconstruction timing on
the public Arduino UNO SRAM-PUF corpus:

```bash
uv run python -m src.analysis.puf_fuzzy_extractor --dataset-root <Raw_data_UNO>
taskset -c 2 uv run python -m src.analysis.puf_entropy --dataset-root <Raw_data_UNO>
```

### Visualization Demo (ACM + Diffusion on Images)

To visually demonstrate how Arnold Cat Map (ACM) and diffusion-style (XOR-based) encryption obfuscate images, use the small demo script:

```bash
# From the repository root
uv run demo.py path/to/img1.jpg path/to/img2.jpg
```

This will open a 2×3 grid figure showing, for each image, **Original**, **ACM-encrypted**, and **Diffusion-encrypted** versions, and will also save `demo_output.png` and `demo_output.pdf` to the current directory.

### Baseline: Training CNN from Scratch

#### Pretraining on ImageNet Subset

You can pretrain a model on a subset of ImageNet before fine-tuning on downstream tasks:

```bash
# Pretrain ConvNeXt-Tiny on 20% ImageNet
uv run src/experiments/pretrain.py --model_name convnext_tiny --rate 0.2 --epochs 100 --device cuda:0

uv run src/experiments/pretrain.py --model_name efficientnet_b0 --rate 0.2 --epochs 100 --device cuda:2
```

**Note**: Pretrained models are saved in `results/pretrain/` with names following the pattern `pretrain_{model_name}_rate{rate}_{timestamp}`. List available checkpoints with `ls results/pretrain/`.

#### Fine-tuning on Downstream Datasets (Post-training)

After pretraining (or starting from scratch), you can fine-tune models on downstream tasks using `src/experiments/posttrain.py`. This script supports both randomly initialized models and pretrained checkpoints.

**Supported architectures**: `resnet18`, `resnet50`, `resnet101`, `vgg16`, `mobilenet_v2`, `convnext_tiny`, `efficientnet_b0`
**Supported datasets**: `cifar100`, `caltech256`, `cub200`, `imagenet100`, `officehome`, `domainnet_clipart`

**Fine-tuning from pretrained checkpoint**:

```bash
# Fine-tune a pretrained model on downstream tasks
# Replace <pretrain_dir> with the actual directory name from results/pretrain/
uv run src/experiments/posttrain.py --dataset officehome --pretrained_path results/pretrain/<pretrain_dir>/pretrained_model.pth --model_name convnext_tiny --device cuda:0
uv run src/experiments/posttrain.py --dataset cifar100 --pretrained_path results/pretrain/<pretrain_dir>/pretrained_model.pth --model_name efficientnet_b0 --device cuda:0
```

## Encryption Methods

### Arnold Cat Map (ACM)

ACM permutes the positions of an $S \times S$ attention matrix ($S$ is the hidden size):

```
[x']   [1   p   ] [x]
[y'] = [q  pq+1 ] [y]  (mod S),  applied n times
```

The matrix has determinant 1, so the map is invertible for any `p, q`. `A^n mod S` is computed once with a fixed six-round exponentiation schedule, so the number of matrix products does not depend on the secret `n`.

**Applied to**: Attention weights (Query, Key, Value, Output)

### FFN Permutation

A Knuth (Fisher-Yates) shuffle permutes the model dimension of each FFN matrix: the input columns of the intermediate layer and the output rows of the output layer.

**Applied to**: FFN weights (Intermediate and Output layers)

### Key Schedule

`src/encryption/keys.py` follows Secs. 4.2 and 4.2.3 of the paper:

- The layer key `K_i = HKDF(K_mvs, i)` (HKDF-SHA-256, 512 bits) splits into a 256-bit attention key and a 256-bit FFN key.
- Each attention matrix takes one 64-bit block of the attention key: 29 bits each for `p, q = 1 + (b mod (S-1))` and 6 bits for `n = 3 + (b_n mod 32)`, so `n` is in [3, 34].
- Each FFN matrix `f` is shuffled with indices drawn from a ChaCha20 stream keyed by `HKDF(K_FFN, f)`.
- Each protected tensor's ChaCha20 diffusion key and nonce come from `HKDF(K_mvs, i || tensor id)`.

```bash
uv run python -m src.encryption.keys   # self-check (RFC 5869 test vector, key ranges)
```

### Secure Diffusion

Secure mode applies a domain-separated RFC 8439 ChaCha20 keystream after the
ACM or FFN permutation. The implementation generates the keystream on the fly
and does not allocate a tensor-sized key.

### Language, LLM, and Speech Models

Clean, authorized, and wrong-key quality plus latency (median of five runs) for RoBERTa on SST-2, OPT on WikiText-103, and Wav2Vec 2.0 on LibriSpeech. `--layers` is the number of protected layers (evenly spaced) or `all`.

```bash
uv run python -m src.experiments.nlp_scalability_experiment --model textattack/roberta-base-SST-2 --task sst2 --mode base --layers 6 --output results/nlp_scalability/roberta.json
uv run python -m src.experiments.nlp_scalability_experiment --model facebook/opt-6.7b --task causal --mode secure --layers 6 --batch-size 16 --output results/nlp_scalability/opt_6_7b.json
uv run python -m src.experiments.speech_asr_experiment --mode base --layers 6 --output results/speech_asr/wav2vec2.json
```

### Layer-Selection Study

Preregistered DeiT-S/ViT-B study of which encrypted layers resist retraining (representation damage, five-epoch screen, 20-epoch confirmation, oracle attacker):

```bash
uv run python -m src.experiments.run_layer_subset_study --phase short --devices 0 1 2 3 --data-dir data/imagenet-1k
uv run python -m src.experiments.extra_layers_cka --device cuda:0
```

### ACM Fixed-Schedule Timing

Host and GPU timing of variable- and fixed-schedule ACM exponentiation over `n` in [3, 34]:

```bash
uv run python -m src.experiments.acm_fixed_schedule_timing --device cuda:0 --seed 20260713 --output results/acm_fixed_schedule/timing_seed_20260713.json
uv run python -m src.experiments.summarize_acm_fixed_schedule_timing results/acm_fixed_schedule/timing_seed_*.json --output results/acm_fixed_schedule/cross_run_summary.json
```

### Inference Overhead Analysis

End-to-end ViT-B latency (batch size 64) for the baseline, `ChaosFormer`, and
`ChaosFormer-S`, sweeping the number of encrypted layers and ACM iterations.
Two protected pipelines are available. `--pipeline layerwise` decrypts each
protected layer into shared buffers immediately before it runs and discards the
plaintext after it (`TransformerCipher.authorized_inference`). `--pipeline reencrypt`
decrypts the protected layers in place, runs the forward pass and re-encrypts them.
Every configuration is timed with one boundary: synchronize, R x protected pass,
synchronize. Within each trial the baseline and every distinct config run in rotating order, and
each config is reported as its median over trials relative to the shared baseline.

```bash
uv run python -m src.experiments.vit_latency_sweep --device cuda:0 --trials 15 --reps 20 \
    --pipeline reencrypt --output results/latency/gpu.json
uv run python -m src.experiments.vit_latency_sweep --device cpu --threads 32 --trials 15 --reps 50 \
    --pipeline layerwise --output results/latency/cpu.json
uv run python -m src.analysis.plot_overhead   # writes results/latency/{base,secure}_{cpu,gpu}.pdf
```

### Cipher comparison

One encryption + decryption round of the whole ViT-B and ViT-L checkpoints with AES
(CBC/CTR/GCM, OpenSSL with AES-NI), TEA and XOR, against `ChaosFormer` and
`ChaosFormer-S` on their protected layers (K = 6 and 7, n = 3). Methods run in rotating
order; each value is the median over trials. On the CPU every method runs on one thread.

```bash
OMP_NUM_THREADS=1 taskset -c 0 uv run python -m src.experiments.cipher_comparison --device cpu --threads 1 --trials 15 \
    --output results/cipher_comparison/cpu.json
uv run python -m src.experiments.cipher_comparison --device cuda:0 --trials 15 \
    --output results/cipher_comparison/gpu.json
```

## Contact

For issues or questions about this repository, please refer to the paper or contact the authors by email (peichunhua@link.cuhk.edu.cn).

## Citation

The first citation will be updated after publication. The second citation is our conference version.

```bibtex
@article{hua2026chaosformer,
  author  = {Peichun Hua and Yue Zheng},
  title   = {ChaosFormer: Hardware-Bound Edge Transformer Model Obfuscation for Intellectual Property Protection},
  journal = {ACM Transactions on Design Automation of Electronic Systems},
  year    = {2026},
  note    = {To appear}
}

@inproceedings{hua2025securing,
  author    = {Peichun Hua and Hanxiu Zhang and Tuo Li and Yue Zheng},
  title     = {Securing On-device Transformer with Hardware Binding and Reversible Obfuscation},
  booktitle = {IEEE Annual Computer Security Applications Conference, {ACSAC} 2025},
  pages     = {956--972},
  publisher = {IEEE},
  year      = {2025},
  doi       = {10.1109/ACSAC67867.2025.00079}
}
```
