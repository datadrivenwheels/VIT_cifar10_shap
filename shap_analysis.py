# -*- coding: utf-8 -*-
"""
SHAP patch-importance analysis for ViT trained on CIFAR-10.

For each test image, computes SHAP values that show which patches (4x4 pixel
regions) contributed positively or negatively to the predicted class.

Usage:
    python shap_analysis.py
    python shap_analysis.py --n_images 200 --output_dir shap_results
    python shap_analysis.py --checkpoint ./checkpoint/vit-cifar10-4-ckpt.t7
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from tqdm import tqdm

from models.vit import ViT

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='SHAP analysis for ViT on CIFAR-10')
parser.add_argument('--checkpoint', default='./checkpoint/vit-cifar10-4-ckpt.t7')
parser.add_argument('--n_images', type=int, default=100,
                    help='Number of test images to explain (default 100)')
parser.add_argument('--output_dir', default='./shap_results')
parser.add_argument('--patch', type=int, default=4)
parser.add_argument('--size', type=int, default=32)
parser.add_argument('--max_evals', type=int, default=1000,
                    help='SHAP evaluations per image (higher = more accurate, slower)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for model forward passes inside SHAP')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR-10 normalization constants
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
print(f'Using device: {device}')

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print('Loading model...')
net = ViT(
    image_size=args.size,
    patch_size=args.patch,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
)
checkpoint = torch.load(args.checkpoint, map_location=device)
net.load_state_dict(checkpoint['net'])
net.to(device)
net.eval()
print(f"Checkpoint accuracy: {checkpoint['acc']:.2f}%  (epoch {checkpoint['epoch']})")

# ---------------------------------------------------------------------------
# Test data  — load as raw uint8 (no normalization) so SHAP masker can work
# on natural-looking images; we normalise inside predict()
# ---------------------------------------------------------------------------
transform_test = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),   # → [0,1], shape (C,H,W)
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# Sample n_images uniformly — one from each class first, then fill randomly
np.random.seed(args.seed)
indices_by_class = [[] for _ in range(10)]
for idx, (_, label) in enumerate(testset):
    indices_by_class[label].append(idx)

selected = []
per_class = max(1, args.n_images // 10)
for c in range(10):
    chosen = np.random.choice(indices_by_class[c],
                               min(per_class, len(indices_by_class[c])),
                               replace=False).tolist()
    selected.extend(chosen)
# Top up to n_images if needed
remaining = list(set(range(len(testset))) - set(selected))
np.random.shuffle(remaining)
selected = (selected + remaining)[:args.n_images]
np.random.shuffle(selected)

images_tensor = []  # (N, C, H, W) float [0,1]
labels = []
for idx in selected:
    img, lbl = testset[idx]
    images_tensor.append(img)
    labels.append(lbl)

images_tensor = torch.stack(images_tensor)          # (N, C, H, W)
images_np = images_tensor.permute(0, 2, 3, 1).numpy()  # (N, H, W, C) for SHAP
labels = np.array(labels)

print(f'Selected {len(selected)} images  ({per_class} per class)')

# ---------------------------------------------------------------------------
# Prediction function for SHAP
# SHAP masker passes images as float32 numpy (N, H, W, C) in [0,1]
# ---------------------------------------------------------------------------
def predict(imgs_np: np.ndarray) -> np.ndarray:
    """imgs_np: (N, H, W, C) float32 [0,1]  →  (N, 10) softmax probabilities"""
    x = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float()  # (N,C,H,W)
    # Normalize
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1)
    std_t  = torch.tensor(STD).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    probs_all = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.batch_size):
            batch = x[start:start + args.batch_size].to(device)
            logits = net(batch)
            probs_all.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs_all, dim=0).numpy()

# ---------------------------------------------------------------------------
# SHAP explainer setup
# masker: blur replaces masked pixels with a blurred version (looks natural,
# avoids out-of-distribution artefacts from black or mean masking)
# ---------------------------------------------------------------------------
print('Setting up SHAP explainer...')
masker = shap.maskers.Image(f'blur({args.size},{args.size})',
                             images_np[0].shape)   # (H, W, C)
explainer = shap.Explainer(predict, masker, output_names=list(CLASSES))

# ---------------------------------------------------------------------------
# Run SHAP  (partition explainer — efficient for image patches)
# ---------------------------------------------------------------------------
print(f'Running SHAP on {len(images_np)} images '
      f'(max_evals={args.max_evals}, batch_size={args.batch_size})...')
shap_values = explainer(
    images_np,
    max_evals=args.max_evals,
    batch_size=args.batch_size,
)

# shap_values.values: (N, H, W, C, 10)  — one output per class
raw_all = shap_values.values   # (N, H, W, C, 10)

# Get predicted class from model directly
probs = predict(images_np)                   # (N, 10)
predicted_classes = probs.argmax(axis=1)     # (N,)

# Select SHAP values for each image's predicted class
raw = raw_all[np.arange(len(images_np)), :, :, :, predicted_classes]  # (N, H, W, C)

# ---------------------------------------------------------------------------
# Aggregate pixel-level SHAP → patch-level heatmap
# Average over colour channels, then pool into patch grid
# ---------------------------------------------------------------------------
P = args.patch
n_patches = args.size // P          # 8 patches per side

def pixel_to_patch(shap_map: np.ndarray) -> np.ndarray:
    """
    shap_map: (H, W, C)  pixel-level SHAP values for one image/class
    returns:  (n_patches, n_patches)  mean absolute SHAP per patch
    """
    per_channel_mean = shap_map.mean(axis=-1)           # (H, W)
    grid = per_channel_mean.reshape(n_patches, P, n_patches, P)
    return grid.mean(axis=(1, 3))                        # (n_patches, n_patches)

patch_shap = np.stack([
    pixel_to_patch(raw[i])
    for i in range(len(images_np))
])  # (N, n_patches, n_patches)

# Save raw SHAP arrays for later analysis
np.save(os.path.join(args.output_dir, 'patch_shap.npy'), patch_shap)
np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
np.save(os.path.join(args.output_dir, 'predicted.npy'), predicted_classes)
np.save(os.path.join(args.output_dir, 'images.npy'), images_np)
print(f'Saved raw SHAP arrays to {args.output_dir}/')

# ---------------------------------------------------------------------------
# Helper: draw one explained image
# ---------------------------------------------------------------------------
def plot_explanation(ax_orig, ax_heat, ax_over, img_np, shap_grid, title):
    """Draw original | heatmap | overlay on three axes."""
    vmax = np.abs(shap_grid).max() or 1e-6

    # Original
    ax_orig.imshow(img_np)
    ax_orig.set_title('Image', fontsize=7)
    ax_orig.axis('off')

    # SHAP patch heatmap
    im = ax_heat.imshow(shap_grid, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax,
                         interpolation='nearest',
                         extent=[0, args.size, args.size, 0])
    ax_heat.set_title('SHAP patches', fontsize=7)
    ax_heat.axis('off')
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Overlay: original image + semi-transparent SHAP heatmap
    ax_over.imshow(img_np)
    ax_over.imshow(shap_grid, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   alpha=0.55, interpolation='nearest',
                   extent=[0, args.size, args.size, 0])
    ax_over.set_title(title, fontsize=7)
    ax_over.axis('off')

# ---------------------------------------------------------------------------
# Plot 1: Individual explanations (up to 30 shown)
# ---------------------------------------------------------------------------
n_show = min(30, len(images_np))
n_cols = 5
n_rows = n_show  # each image gets one row of 3 panels

fig_h = n_show * 1.8
fig, axes = plt.subplots(n_show, 3, figsize=(9, fig_h))
if n_show == 1:
    axes = axes[np.newaxis, :]

for i in range(n_show):
    pred_c   = predicted_classes[i]
    true_c   = labels[i]
    correct  = '✓' if pred_c == true_c else '✗'
    title    = f'{correct} pred:{CLASSES[pred_c]} true:{CLASSES[true_c]}'
    plot_explanation(axes[i, 0], axes[i, 1], axes[i, 2],
                     images_np[i], patch_shap[i], title)

plt.suptitle('ViT SHAP Patch Importance — CIFAR-10\n'
             'Red = pushes toward predicted class, Blue = pushes away',
             fontsize=9, y=1.001)
plt.tight_layout()
out_path = os.path.join(args.output_dir, 'individual_explanations.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved individual explanations → {out_path}')

# ---------------------------------------------------------------------------
# Plot 2: Mean SHAP heatmap per class  (average over all correctly predicted
# images of that class — shows what the model "looks at" for each category)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for c in range(10):
    mask = (predicted_classes == c) & (labels == c)
    ax = axes[c]
    if mask.sum() == 0:
        ax.set_title(CLASSES[c], fontsize=9)
        ax.axis('off')
        continue
    mean_shap = patch_shap[mask].mean(axis=0)   # (n_patches, n_patches)
    vmax = np.abs(mean_shap).max() or 1e-6
    im = ax.imshow(mean_shap, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(f'{CLASSES[c]}\n(n={mask.sum()})', fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_patches, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_patches, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.4, alpha=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Mean SHAP Patch Importance per Class\n'
             '(correctly classified images only)',
             fontsize=11)
plt.tight_layout()
out_path = os.path.join(args.output_dir, 'mean_shap_per_class.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved per-class mean SHAP → {out_path}')

# ---------------------------------------------------------------------------
# Plot 3: Top-5 most important patch positions overall (bar chart)
# ---------------------------------------------------------------------------
# For each image, take abs(SHAP) for predicted class
abs_shap = np.abs(patch_shap)   # (N, n_patches, n_patches)
mean_abs  = abs_shap.mean(axis=0)  # (n_patches, n_patches) — average importance

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im = axes[0].imshow(mean_abs, cmap='hot', interpolation='nearest')
axes[0].set_title('Mean |SHAP| per patch position\n(all images)', fontsize=10)
axes[0].set_xticks(np.arange(-0.5, n_patches, 1), minor=True)
axes[0].set_yticks(np.arange(-0.5, n_patches, 1), minor=True)
axes[0].grid(which='minor', color='white', linewidth=0.5, alpha=0.6)
axes[0].tick_params(which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
# Label patch row/col
for r in range(n_patches):
    for c_ in range(n_patches):
        axes[0].text(c_, r, f'{r*n_patches+c_}', ha='center', va='center',
                     fontsize=5, color='cyan')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

flat = mean_abs.flatten()
top5_idx = np.argsort(flat)[::-1][:5]
top5_rows = top5_idx // n_patches
top5_cols = top5_idx % n_patches
top5_labels = [f'({r},{c_})' for r, c_ in zip(top5_rows, top5_cols)]
axes[1].barh(top5_labels[::-1], flat[top5_idx][::-1], color='tomato')
axes[1].set_xlabel('Mean |SHAP|')
axes[1].set_title('Top 5 most important patch positions', fontsize=10)

plt.tight_layout()
out_path = os.path.join(args.output_dir, 'patch_importance_summary.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved patch importance summary → {out_path}')

# ---------------------------------------------------------------------------
# Print summary stats
# ---------------------------------------------------------------------------
accuracy = (predicted_classes == labels).mean() * 100
print(f'\n=== Summary ===')
print(f'Images analysed : {len(images_np)}')
print(f'Accuracy on subset: {accuracy:.1f}%')
print(f'Output files in  : {args.output_dir}/')
print(f'  individual_explanations.png  — per-image heatmaps (first 30)')
print(f'  mean_shap_per_class.png      — average heatmap per class')
print(f'  patch_importance_summary.png — overall most-attended patches')
print(f'  patch_shap.npy               — raw SHAP arrays (N, {n_patches}, {n_patches})')
print(f'  labels.npy / predicted.npy / images.npy')
