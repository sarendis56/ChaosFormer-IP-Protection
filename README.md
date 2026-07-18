# ChaosFormer

This paper introduces a novel hardware-bound encryption framework for protecting intellectual property in Vision Transformer models. The approach leverages Physical Unclonable Functions (PUF) to generate device-specific cryptographic keys, which are used to encrypt model weights through a encryption pipeline: Arnold Cat Map (ACM) encryption for attention weights, permutation-based encryption for Feed-Forward Network (FFN) weights, and diffusion of ultimate protection. This combination of hardware binding and heterogeneous cryptographic methods ensures that models can only execute correctly on authorized devices while providing enhanced security against various attack vectors.

## Overview

This repository implements a novel triple encryption approach for Vision Transformer (ViT) models that combines:

- **Arnold Cat Map (ACM)** encryption for attention weights
- **Permutation-based** encryption for Feed-Forward Network (FFN) weights
- **Diffusion** (optional) encryption for the final obfuscation of the weights distribution

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
git clone <repository-url>
cd transformer-ip-protection

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
git clone <repository-url>
cd transformer-ip-protection
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

The Arnold Cat Map is a chaotic transformation that scrambles 2D matrices:

```
[x']   [a b] [x]
[y'] = [c d] [y]  (mod N)
```

Where `(ad - bc) ≡ ±1 (mod N)` ensures invertibility.

**Applied to**: Attention weights (Query, Key, Value, Output)

### Permutation Encryption

Row/column permutation using cryptographically secure permutation matrices:

```
Encrypted = P × Original × P^T
```

**Applied to**: FFN weights (Intermediate and Output layers)

### Secure Diffusion

Secure mode applies a domain-separated RFC 8439 ChaCha20 keystream after the
ACM or FFN permutation. The implementation generates the keystream on the fly
and does not allocate a tensor-sized key.

### Inference Overhead Analysis

Detailed analysis of dual encryption inference overhead during model execution:

```bash
# With uv
uv run src/analysis/overhead_secure.py --device cuda:0 --triton # Use triton for faster execution
uv run src/analysis/overhead_secure.py --device cpu
uv run src/analysis/overhead_secure_8bit.py --device cuda:0 --triton

# If when running the cpu one with numba triggers "NumbaWarning" with TBB versioning issue, try:
LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH uv run src/analysis/overhead_secure.py --device cpu

# With activated venv
python src/analysis/overhead_secure.py
```

**Analyzes overhead from:**
- ACM (Arnold Cat Map) encryption/decryption
- FFN (Feed-Forward Network) permutation encryption/decryption
- Forward pass execution time

## Contact

For issues or questions about this repository, please refer to the paper or contact the authors by email (peichunhua@link.cuhk.edu.cn).

## Citation
