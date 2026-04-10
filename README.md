# vision-transformers-cifar10 + SHAP

A PyTorch playground for training Vision Transformers (ViT) and related models on CIFAR-10/CIFAR-100, extended with **SHAP patch-importance analysis** to interpret which image patches drive each model prediction.

The repo has been used in [30+ papers](https://scholar.google.co.jp/scholar?hl=en&as_sdt=0%2C5&q=vision-transformers-cifar10&btnG=) including CVPR, ICLR, and NeurIPS.

Please use this citation format if you use this in your research.
```
@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}
```

---

## Setup (Anaconda — including Apple Silicon M-series)

```bash
conda env create -f environment.yml
conda activate vit-cifar10
```

The environment includes PyTorch with MPS (Metal) support for M-series Macs, plus all dependencies for training and SHAP analysis.

---

## Training

```bash
# ViT patch=4, 200 epochs (default)
python train_cifar10.py --nowandb

# CIFAR-100
python train_cifar10.py --dataset cifar100 --nowandb

# Longer training for higher accuracy
python train_cifar10.py --n_epochs 500 --nowandb

# Smaller patch size
python train_cifar10.py --patch 2 --nowandb

# Other architectures
python train_cifar10.py --net vit_small --n_epochs 400 --nowandb
python train_cifar10.py --net convmixer --n_epochs 400 --nowandb
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --nowandb
python train_cifar10.py --net cait --n_epochs 200 --nowandb
python train_cifar10.py --net swin --n_epochs 400 --nowandb
python train_cifar10.py --net res18 --nowandb
python train_cifar10.py --net dyt --nowandb
python train_cifar10.py --net vit_timm  # pretrained ViT-base from timm
```

Checkpoints are saved to `./checkpoint/`, logs to `./log/`.
Use `--resume` to continue from a saved checkpoint.

---

## SHAP Patch-Importance Analysis

For each test image, SHAP computes which 4×4 patches contributed positively (red) or negatively (blue) to the model's predicted class. The `shap_analysis.py` script uses a partition explainer with a blur masker over the ViT's natural 8×8 patch grid.

```bash
# Analyse 100 test images (10 per class), ~1 minute on M-series Mac
python shap_analysis.py --nowandb

# More images for robust per-class statistics
python shap_analysis.py --n_images 500

# Point at a different checkpoint
python shap_analysis.py --checkpoint ./checkpoint/vit-cifar10-4-ckpt.t7
```

### Output files (`./shap_results/`)

| File | Description |
|------|-------------|
| `individual_explanations.png` | Per-image panels: original / SHAP heatmap / overlay (first 30 images) |
| `mean_shap_per_class.png` | Average SHAP heatmap per class across correct predictions |
| `patch_importance_summary.png` | Global mean \|SHAP\| map + top-5 most-attended patch positions |
| `patch_shap.npy` | Raw SHAP arrays `(N, 8, 8)` for custom analysis |
| `labels.npy` / `predicted.npy` / `images.npy` | Ground-truth labels, model predictions, and input images |

---

## Results

| CIFAR-10 | Accuracy | Notes |
|:---------:|:--------:|:-----:|
| ViT patch=4 @ 200 epochs | 78–80% | Our run: **78.21%** on M3 Max |
| ViT patch=4 @ 500 epochs | 85% | |
| ViT patch=4 @ 1000 epochs | 89% | |
| ViT patch=2 | 80% | |
| ViT small | 80% | |
| DyT | 74% | Layernorm-less ViT |
| MLP Mixer | 88% | |
| CaiT | 80% | |
| Swin-t | 90% | |
| ViT small (timm transfer) | 97.5% | |
| ViT base (timm transfer) | 98.5% | |
| ConvMixer (no pretrain) | 96.3% | |
| ResNet18 | 93% | |
| ResNet18 + RandAug | 95% | |

| CIFAR-100 | Accuracy |
|:---------:|:--------:|
| ViT patch=4 @ 200 epochs | 52% |
| ResNet18 + RandAug | 71% |

---

## Model Export

Export a trained model to ONNX or TorchScript:

```bash
python export_models.py --checkpoint ./checkpoint/vit-cifar10-4-ckpt.t7 \
    --model_type vit --output_dir exported_models --verify
```

---

## Updates

* Added SHAP patch-importance analysis (`shap_analysis.py`) with MPS support (2025/4)
* Fixed MPS (Apple Silicon) compatibility for all training and inference (2025/4)
* Added conda environment (`environment.yml`) for reproducible setup (2025/4)
* Added CIFAR-100 support (2025/4)
* Added Dynamic Tanh ViT / DyT (2025/3)
* Added MobileViT (2025/1)
* Added ONNX and TorchScript model export (2024/12)
* Fixed bugs and training settings (2024/2)
* Added MLP Mixer (2022/6)
* Added Swin Transformers, CaiT, ViT-small (2022/3)
* Added wandb logging (2022/3)
* Added ConvMixer (2021/10)

---

## Used in

* Vision Transformer Pruning — [arxiv](https://arxiv.org/abs/2104.08500) / [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
* Understanding why ViT trains badly on small datasets — [arxiv](https://arxiv.org/abs/2302.03751)
* Training deep neural networks with adaptive momentum — [arxiv](https://arxiv.org/abs/2110.09057)
* Moderate coreset: A universal method of data selection — [openreview](https://openreview.net/forum?id=7D5EECbOaf9)
