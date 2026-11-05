# Reliability boundaries

Validated software properties and scientific validity are separate.

What the tests can support: finite-input checks, bounded optimization, known
analytic behavior, no silent overwrites, reproducible same-environment runs,
structured metrics, and a working local pipeline.

What they cannot support: semantic protection of art, Nightshade equivalence,
transfer across image generators, resistance to adaptive removal, imperceptibility,
or reliable corruption of trained models. PSNR and pixel bounds are not a user study.

The encoder is random and linear. Images are tiny and grayscale. The classifier,
data distribution, sample count and regularization are intentionally simple. Five
seeds are illustrative, not a population-level estimate. No confidence intervals
or real-world efficacy claims are made.

Custom inputs do not trigger a training experiment. There is no GPU backend, batch
artwork application, diffusion trainer, pretrained-model adapter, scraping feature
or remote deployment mechanism. Interrupted runs may leave partial output folders;
keep them separate from completed reports. Local inputs should remain stable during
a run. This tool is not a sandbox for hostile image decoders or filesystem races.

Before considering an artist-facing protection tool: reproduce a licensed diffusion
setup, evaluate multiple models and independent concepts, measure perceptual quality,
include adaptive transformations and publish negative as well as positive results.
