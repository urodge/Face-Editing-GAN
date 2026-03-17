# Identity-Preserving Face Editing GAN

> A custom UNet-based Generative Adversarial Network for high-resolution (512×512+) facial attribute editing that preserves subject identity through a multi-component loss stack.

![PyTorch](https://img.shields.io/badge/PyTorch-GAN-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Resolution](https://img.shields.io/badge/Resolution-512×512+-blueviolet?style=flat-square)
![CLIP](https://img.shields.io/badge/CLIP-Identity_Loss-412991?style=flat-square)
![VGG](https://img.shields.io/badge/VGG-Perceptual_Loss-orange?style=flat-square)

---

## Overview

Standard GANs for face editing trade identity fidelity for attribute control — edited outputs often drift from the source subject. This project tackles that problem with a multi-objective loss function that simultaneously optimises for photorealism (adversarial), perceptual similarity (VGG), and semantic identity preservation (CLIP-based embedding similarity).

The result is a generator capable of applying controllable facial attribute edits at 512×512+ resolution while keeping the subject recognisable across edits.

---

## Architecture

### Generator — UNet with skip connections

```
Input Image (512×512×3)
        │
   ┌────▼──────────────────────────────────────────────────┐
   │  Encoder (downsampling path)                           │
   │  Conv → IN → LeakyReLU  ×N  (stride 2 each block)    │
   └────┬──────────────────────────────────────────────────┘
        │ Bottleneck (latent + attribute conditioning)
   ┌────▼──────────────────────────────────────────────────┐
   │  Decoder (upsampling path)                             │
   │  ConvTranspose → IN → ReLU  ×N                        │
   │  + Skip connections from encoder (spatial detail)      │
   └────┬──────────────────────────────────────────────────┘
        │
   Output Image (512×512×3)
```

### Discriminator — PatchGAN

Classifies overlapping image patches as real/fake rather than the full image, pushing the generator to maintain local texture fidelity at high resolution.

---

## Loss Function

The generator is trained against a composite loss:

```
L_total = λ_adv · L_adversarial
        + λ_perceptual · L_perceptual
        + λ_identity · L_identity
        + λ_pixel · L_pixel
```

| Loss component | What it enforces | Implementation |
|---|---|---|
| **Adversarial** `L_adv` | Photorealism — output must fool the discriminator | Hinge loss on PatchGAN D output |
| **Perceptual** `L_perceptual` | Mid-level feature similarity to source image | L2 distance on VGG-19 relu3_4 activations |
| **Identity** `L_identity` | Semantic identity — same person before/after edit | Cosine distance on CLIP ViT-B/32 embeddings |
| **Pixel** `L_pixel` | Low-level structure preservation in non-edited regions | L1 loss on masked pixel regions |

The λ weights are tuned empirically — identity loss weight is highest for edits that affect face structure (hairstyle, facial hair), lower for colour/texture edits (skin tone, lighting).

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | PyTorch |
| Generator | Custom UNet (encoder–decoder + skip connections) |
| Discriminator | PatchGAN |
| Perceptual loss | VGG-19 (pretrained, frozen) |
| Identity loss | CLIP ViT-B/32 (OpenAI, frozen) |
| Normalisation | Instance Normalisation throughout |
| Training | Adam optimiser, β₁=0.5, β₂=0.999 |

---

## Attribute Control

Attributes are injected as conditioning vectors at the bottleneck layer. Supported edit categories:

- Age progression / regression
- Facial hair addition / removal
- Expression (smile, neutral)
- Hair colour and style
- Lighting and skin tone

Edits can be composed — multiple attribute vectors can be applied simultaneously.

---

## Setup

```bash
git clone https://github.com/urodge/<repo-name>
cd <repo-name>
pip install -r requirements.txt

# Download pretrained weights
python download_weights.py

# Run inference on a single image
python edit.py --input path/to/face.jpg --attributes smile +0.8 age +0.4 --output result.jpg

# Train from scratch
python train.py --config configs/train_512.yaml
```

---

## Project Structure

```
├── train.py                # Training loop (G + D alternating updates)
├── edit.py                 # Inference — load weights, apply attribute edits
├── models/
│   ├── generator.py        # UNet generator
│   ├── discriminator.py    # PatchGAN discriminator
│   └── losses.py           # Adversarial, perceptual, identity, pixel losses
├── utils/
│   ├── data.py             # Dataset loader + augmentation
│   └── clip_loss.py        # CLIP embedding extraction + cosine distance
├── configs/
│   └── train_512.yaml      # Hyperparameters and loss weights
├── requirements.txt
└── README.md
```

---

## Key Challenges

**Identity drift at high resolution**
Early experiments showed the generator learning to edit identity-irrelevant regions (background, lighting) correctly but drifting on facial structure at 512px. Adding the CLIP identity loss as a regulariser significantly reduced drift without sacrificing edit sharpness.

**Balancing adversarial vs. perceptual objectives**
High λ_perceptual produced blurry outputs (the perceptual loss penalises sharp texture differences). The final λ schedule starts perceptual-dominant for early training stability, then increases λ_adv as the discriminator matures.

---

## License

MIT
