# FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p align="center">
  <em>Accepted at AAAI 2026</em>
</p>

This repository contains the official implementation of **FGM-HD**, a novel framework that enhances the output diversity of Fractal Generative Models (FGMs) by incorporating the Hausdorff Dimension (HD) as a geometric indicator of structural complexity.

> **FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction**
>
> Haowei Zhang, Yuanpei Zhao, Jizhe Zhou\*, Mao Li\*
>
> College of Computer Science, Sichuan University, China
>
> \* Corresponding authors

## Overview

Fractal Generative Models (FGMs) are efficient in generating high-quality images through recursive self-similarity. However, this same property limits output diversity. FGM-HD addresses this by introducing the **Hausdorff Dimension (HD)** — a fractal geometry concept that quantifies structural complexity — to guide generation toward more diverse outputs.

### Key Contributions

1. **Learnable HD Estimation**: A ResNet152-based multi-scale convolutional network that efficiently predicts HD directly from image embeddings, avoiding expensive numerical box-counting methods during training.

2. **Monotonic Momentum-Driven Scheduling (MMDS)** : A dynamic weight adjustment strategy that progressively balances visual quality and structural diversity during training, ensuring stable optimization without sacrificing generation quality.

3. **HD-Guided Rejection Sampling**: An inference-time strategy that leverages FGM's recursive structure to generate multiple candidate patches and retain only those with HD values above a threshold, resulting in geometrically richer outputs.

**Experimental Results**: On ImageNet, FGM-HD achieves a **39% improvement in Recall** compared to vanilla FGMs while maintaining comparable FID and Inception Score.

### Architecture

The FGM-HD framework consists of three main components:

- **Training Phase**: Input noise is recursively processed by the FGM. Generated images are evaluated by the HD estimation module. The HD loss is dynamically weighted by the MMDS strategy through λ(t).
- **Inference Phase**: Multiple candidate patches are generated via FGM's recursive structure. HD-guided rejection sampling filters out structurally simple outputs, retaining only those exceeding a threshold τ.
- **HD Estimation Module**: A multi-scale convolutional network built on ResNet152 enables fast HD prediction from image embeddings.

## Project Structure

```
fractal_HD/
├── main_fractalgen.py         # Main training/evaluation script
├── engine_fractalgen.py       # Training engine with HD loss integration
├── environment.yaml           # Conda environment specification
├── LICENSE
├── README.md
├── weight.txt                 # MMDS weight schedule example
│
├── models/
│   ├── fractalgen.py          # FractalGen model (supports 64/256/512 resolutions)
│   ├── mar.py                 # Masked Autoregressive (MAR) generator
│   ├── ar.py                  # Autoregressive (AR) generator
│   └── pixelloss.py           # Pixel-level loss module
│
├── resnet/
│   ├── train.py               # MultiScaleResNet152 for HD estimation
│   ├── createDB.py            # Dataset creation for HD training
│   ├── testNet.py             # Single-image HD inference test
│   └── hd_values.txt          # Pre-computed HD values for training
│
├── util/
│   ├── misc.py                # Distributed training utilities & metric logging
│   ├── lr_sched.py            # Learning rate scheduling
│   ├── crop.py                # Image cropping transforms
│   ├── download.py            # Download pre-trained models
│   ├── visualize.py           # Generation visualization
│   ├── filtering.py           # HD-guided rejection sampling
│   ├── mmds.py                # Monotonic Momentum-Driven Scheduling
│   └── datasetHD.py           # HD dataset utilities
│
├── src/
│   └── torch-fidelity/        # FID/IS evaluation metrics library
│
├── imagenet_train.sh          # Training script examples
├── imagenet_train_256.sh
├── imagenet_train_512.sh
├── imagenet_train_256_continue.sh
├── inf.sh                     # Inference script
├── inf_256.sh
└── mar.sh                     # MAR-specific training script
```

## Preparation

### Requirements

- Python 3.8+
- PyTorch 2.2.2+ with CUDA 11.8+
- 4+ GPUs with sufficient VRAM (tested on H100 GPUs)

### Dataset

Download [ImageNet](http://image-net.org/download) dataset and place it in your `IMAGENET_PATH`. The directory should contain `train/` and `val/` subdirectories.

### Installation

```bash
# Clone the repository
git clone https://github.com/chaoerlie/fractal_HD.git
cd fractal_HD

# Create and activate conda environment
conda env create -f environment.yaml
conda activate fractalgen
```

Alternatively, install dependencies manually:

```bash
pip install torch==2.2.2 torchvision==0.17.2
pip install opencv-python==4.1.2.30 timm==0.9.12 tensorboard==2.10.0 scipy==1.9.1 gdown==5.2.0
pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity
```

### Pre-trained Models

The pre-trained HD estimation model (ResNet152-based) should be placed at `resnet/model_epoch_400.pth`. You can train your own HD estimator using `resnet/train.py`:

```bash
python resnet/train.py
```

## Usage

### Training with HD Loss

FGM-HD extends the original FractalGen training with HD-specific parameters:

- `--hd_model`: Path to the pre-trained HD prediction model
- `--standard_hd_value`: Target Hausdorff Dimension value
- `--hd_weight_schedule`: HD loss weight schedule. Format: `"linear:start_val:end_val"` or path to a `.txt` file
- `--mmds`: Enable Monotonic Momentum-Driven Scheduling (recommended)

**Example: Training FractalAR on ImageNet 64×64 with HD loss:**

```bash
torchrun --nproc_per_node=8 --nnodes=4 \
  --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  main_fractalgen.py \
  --model fractalar_in64 --img_size 64 --num_conds 1 \
  --batch_size 64 --eval_freq 40 --save_last_freq 10 \
  --epochs 800 --warmup_epochs 40 \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 \
  --lr_schedule cosine \
  --gen_bsz 256 --num_images 8000 --num_iter_list 64,16 \
  --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  --hd_model resnet/model_epoch_400.pth \
  --standard_hd_value 1.85 \
  --hd_weight_schedule weight.txt \
  --mmds
```

**Example: Training FractalMAR on ImageNet 256×256:**

```bash
torchrun --nproc_per_node=8 --nnodes=4 \
  --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  main_fractalgen.py \
  --model fractalmar_large_in256 --img_size 256 --num_conds 5 --guiding_pixel \
  --batch_size 32 --eval_freq 40 --save_last_freq 10 \
  --epochs 800 --warmup_epochs 40 \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 \
  --lr_schedule cosine \
  --gen_bsz 256 --num_images 8000 --num_iter_list 64,16,16 \
  --cfg 21.0 --cfg_schedule linear --temperature 1.1 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  --hd_model resnet/model_epoch_400.pth \
  --standard_hd_value 1.85 \
  --hd_weight_schedule weight.txt \
  --mmds
```

### Evaluation (Generation)

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_fractalgen.py \
  --model fractalmar_large_in256 --img_size 256 --num_conds 5 --guiding_pixel \
  --gen_bsz 1024 --num_images 50000 \
  --num_iter_list 64,16,16 --cfg 21.0 --cfg_schedule linear --temperature 1.1 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --seed 0 --evaluate_gen
```

### HD-Guided Inference with Rejection Sampling

Use the filtering module to perform HD-guided rejection sampling on generated images:

```bash
# Threshold-based filtering (per-class median HD as threshold)
python util/filtering.py <generated_images_dir> util/val_stats.csv threshold median \
  --n 50 --apply --workers 8

# Top-K fraction filtering (keep top 50% per class)
python util/filtering.py <generated_images_dir> util/val_stats.csv top 0.5 \
  --n 50 --apply --workers 8
```

### NLL Evaluation

```bash
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  main_fractalgen.py \
  --model fractalmar_in64 --img_size 64 --num_conds 5 \
  --nll_bsz 128 --nll_forward_number 10 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --seed 0 --evaluate_nll
```

## Key Components Explained

### HD Estimation Module (`resnet/train.py`)

`MultiScaleResNet152` is a ResNet152-based regression network with multi-scale convolutional heads (3×3, 5×5, 7×7 kernels) that predicts the Hausdorff Dimension of an input image. It takes a 224×224 RGB image and outputs a single scalar HD value.

### MMDS (`util/mmds.py`)

The Monotonic Momentum-Driven Scheduling dynamically adjusts the HD loss weight λ during training. It maintains an exponential moving average of the loss improvement (delta), ensuring λ increases monotonically as training progresses — this prevents λ from oscillating or decreasing when the loss temporarily fluctuates.

### HD-Guided Rejection Sampling (`util/filtering.py`)

A post-hoc selection mechanism that filters generated images based on their HD values. Supports two modes:
- **Threshold mode**: Keep images with HD ≥ per-class threshold
- **Top-K mode**: Keep the top K% images with highest HD per class

Uses efficient box-counting algorithm for HD computation with parallel processing support.

## Available Models

| Model | Image Size | FID-50K | IS | #Params |
|-------|-----------|---------|-----|---------|
| FractalAR | 64×64 | 5.30 | 56.8 | 432M |
| FractalMAR | 64×64 | 2.72 | 87.9 | 432M |
| FractalMAR-Base | 256×256 | 11.80 | 274.3 | 186M |
| FractalMAR-Large | 256×256 | 7.30 | 334.9 | 438M |
| FractalMAR-Huge | 256×256 | 6.15 | 348.9 | 848M |
| FractalMAR-Huge | 512×512 | — | — | — |

## Results

Our FGM-HD framework achieves a **39% improvement in Recall** compared to vanilla FGMs on ImageNet, while preserving comparable FID and Inception Score. The MMDS strategy ensures smooth training dynamics without sacrificing image quality.

For detailed experimental results, please refer to our paper.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhang2026fgmhd,
  title={FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction},
  author={Zhang, Haowei and Zhao, Yuanpei and Zhou, Jizhe and Li, Mao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

The original Fractal Generative Models paper:

```bibtex
@article{li2025fractal,
  title={Fractal Generative Models},
  author={Li, Tianhong and Sun, Qinyi and Fan, Lijie and He, Kaiming},
  journal={arXiv preprint arXiv:2502.17437},
  year={2025}
}
```

## Acknowledgements

This project builds upon [Fractal Generative Models](https://github.com/LTH14/fractalgen) by Li et al. We thank the original authors for their excellent work and open-source release. We also thank the authors of [torch-fidelity](https://github.com/toshas/torch-fidelity) for the evaluation metrics library.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions, please contact: zhanghaowei1@stu.scu.edu.cn
