# InLine-DS: Concentration Beyond Injectivity with Magnitude-Modulated Spectral Kernels for Linear Attention

Official implementation of **InLine-DS**, an extension of [InLine attention](https://arxiv.org/abs/2412.06590) that adds a third property, **content-dependent concentration**, alongside injectivity and local modeling, to close the remaining gap between linear and Softmax attention.

This repository builds directly on the [InLine](https://github.com/LeapLabTHU/InLine) codebase and retains its subtractive-normalization attention formulation and learned local residual unchanged. InLine-DS introduces:

- **MMSK (Magnitude-Modulated Spectral Kernel)** — a feature map that factors each token into a learned orthogonal direction and a per-head, content-adaptive magnitude (temperature τ and floor η), replacing the fixed, unconditional kernels (ReLU, Identity) used in standard linear attention.
- **Dual-frequency streams** — a fixed-basis (DCT-II) decomposition of tokens into low- and high-frequency components, recombined through a content-adaptive gate.
- **Stabilization machinery** — RMSNorm, LayerScale, running key centering, and a bounded skew-symmetric parameterization for the learned orthogonal rotation, needed to make the more expressive kernel trainable at ViT scale.

All InLine-DS operators retain **O(N) complexity** in token count, matching the linear-attention cost profile of the original InLine model.

---

## Contents

```
.
├── inlineds_models/
│   ├── inline_deit_ds.py       # InLine-DS DeiT implementation
│   ├── inline_swin_ds.py       # InLine-DS Swin implementation
│   ├── inline_cswin_ds.py      # InLine-DS CSwin implementation
│   └── inline_pvt_ds.py        # InLine-DS PVT implementation
├── data/                    # Holds the unzipped datasets
|__ data_archives/            # Holds the zipped datasets
├── InLine-DS_train.ipynb     # Training notebook
└── README.md
```

## Supported Backbones

| Backbone | Variants | Design |
|---|---|---|
| DeiT | Tiny, Small | Isotropic |
| Swin Transformer | Tiny, Small | Hierarchical, windowed |
| CSwin Transformer | Tiny, Small | Hierarchical, cross-shaped windows |
| PVT | Tiny | Hierarchical, spatial-reduction |

> **Note:** Base-scale variants are not currently trained/released due to compute constraints (single-GPU setup); see [Limitations](#limitations).

## Datasets

- **CIFAR-10 / CIFAR-100** ([Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf))
- **SVHN** ([Netzer et al., 2011](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf))
- **TinyImageNet-200** ([Le & Yang, 2015](http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf))


## Installation

```bash
git clone https://github.com/<your-org>/inline-ds.git
cd inline-ds
```

Requires PyTorch ≥ 1.13 and [`timm`](https://github.com/rwightman/pytorch-image-models).

## Training
Train with InLine-DS_train.ipynb.


## Evaluation
Use InLine-DS_train.ipynb.


## Appendix Diagnostics

Scripts used to generate the diagnostic figures reported in the paper's appendix are provided in the InLine-DS_train.ipynb notebook, including:

- Learned kernel parameter (τ, η) distributions by depth
- Patch attention / gradient-weighted attention rollout visualizations
- Attention distance / locality heatmaps
- Attention head redundancy / similarity matrices
- Head pruning sensitivity curves
- Layer-wise performance attribution by block bypass
- Attention output dropout sensitivity
- Reliability diagrams, risk-coverage curves, and per-class F1 analysis
- Penultimate-feature PCA by class

## Results (selected)

| Backbone | Dataset | Baseline (InLine) | InLine-DS |
|---|---|---|---|
| DeiT-Tiny | CIFAR-10 | 73.35% | 93.71% |
| DeiT-Tiny | TinyImageNet-200 | 44.70% | 60.42% |
| DeiT-Small | SVHN | 93.80% | 98.01% |
| CSwin-Tiny | CIFAR-100 | 57.14% | 61.08% |
| PVT-Tiny | TinyImageNet-200 | 53.22% | 58.74% |

Full results tables are provided in the paper.

## Limitations

- Base-scale backbone variants were not evaluated due to limited compute (single NVIDIA L4 GPU via Google Colab); results are reported for Tiny/Small(/Medium) configurations only.
- Evaluation is restricted to CIFAR-10/100, SVHN, and TinyImageNet-200; full-resolution ImageNet-1K results are left for future work.

## Citation

If you use this code, please cite both this work and the original InLine paper it builds on:

```bibtex
@article{inlinemethodname2026,
  title   = {Concentration Beyond Injectivity with Magnitude-Modulated Spectral Kernels for Linear Attention},
  author  = {N/A},
  journal = {N/A},
  year    = {N/A}
}

@inproceedings{han2024bridging,
  title     = {Bridging the Divide: Reconsidering Softmax and Linear Attention},
  author    = {Han, Dongchen and Pu, Yifan and Xia, Zhuofan and Han, Yizeng and Pan, Xuran and Li, Xiu and Lu, Jiwen and Song, Shiji and Huang, Gao},
  booktitle = {NeurIPS},
  year      = {2024}
}
```

## Acknowledgements

This repository is built directly on top of [LeapLabTHU/InLine](https://github.com/LeapLabTHU/InLine). We thank the original authors for open-sourcing their implementation.

