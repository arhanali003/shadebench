# Method and provenance

## Research context

Nightshade explores whether carefully modified image/text training examples can
alter diffusion-model behavior for targeted concepts. Its public paper motivates
feature-space changes constrained by visible-image similarity. This repository
examines a much smaller mechanism and does not reproduce its training pipeline,
semantic encoders, sample selection, perceptual optimization or reported efficacy.

Primary reference: Shan et al., [Nightshade: Prompt-Specific Poisoning Attacks on
Text-to-Image Generative Models](https://arxiv.org/abs/2310.13828), IEEE S&P 2024.
See also the [project explanation](https://nightshade.cs.uchicago.edu/whatis.html)
and [FAQ](https://nightshade.cs.uchicago.edu/faq.html). Source code here was written
independently; the paper's figures and application assets are not redistributed.

## Objective

For source x, target t and a fixed 32×1024 random orthonormal projection F, minimize

    L(z) = ||Fz - Ft||²
    subject to max(0, x - epsilon) <= z <= min(1, x + epsilon)

Images are 32×32 grayscale arrays in [0,1]. The gradient is
`2 Fᵀ(Fz - Ft)`. Projected gradient descent uses step size
`1 / (2 ||F||₂²)`, with a denominator floor for the degenerate zero matrix.
Each iteration clips back into the feasible pixel interval. This is a convex
quadratic surrogate, not a perceptual or semantic objective.

The control adds independent uniform noise in [-epsilon, epsilon], then clips to
[0,1]. It has the same maximum budget, not necessarily the same RMSE or perceptual
severity. Both controls and optimized pixels are quantized to 8-bit before reporting.

## Training experiment

Generate two classes of original synthetic stripe images, with varying centers,
widths, contrast and noise. Use 16 training images per class and a separate held-out
set of 40 per class. Modify eight source-class training images toward eight
opposite-class references; keep all labels unchanged. Train the same regularized
least-squares classifier on clean, random and optimized arms, using the fixed
features and an unpenalized intercept. Evaluate all arms on the same clean held-out
images. Neither optimization nor fitting uses held-out images.

This deliberately small classifier is not a text-to-image generator. Reporting no
accuracy change is a valid outcome, not a reason to silently increase the budget
until a preferred result appears.

## Robustness measurements

Apply JPEG quality 85, bilinear half-size down/up sampling, and a 3×3 median filter
to source, random and optimized images separately. Report feature distances for
all arms so effects of the transformation itself remain visible. These tests do
not retrain on transformed data and do not establish downstream robustness.

## Reproducibility

Reports record seeds and dependency versions; image SHA-256 values bind metrics to
exports. The tested baseline is NumPy 1.26.4 and Pillow 11.3.0 on Python 3.9.
Floating point libraries and QR decomposition can vary across platforms. Compare
metrics with appropriate tolerances; do not demand universal byte-identical replay.
