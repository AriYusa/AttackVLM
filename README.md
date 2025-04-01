# AttackVLM Reproduction Study

This repository contains the code for reproducing and extending the AttackVLM methodology, which evaluates the robustness of large vision-language models (VLMs) against adversarial image-grounded text generation.

### Reproduction Setup

The reproduction follows the original experimental design:
- **Dataset**: 500 random images from the ILSVRC2012 validation dataset paired with random captions from the COCO dataset.
- **Generative Model**: Stable Diffusion.
- **Surrogate Model**: CLIP ViT-B/32.
- **Victim Models**: UniDiffuser (original paper) and Moondream (newer model).

#### Implementation Details
- Pixel values and perturbation parameters were normalized between [0,1].
- Hyperparameters were set according to the original paper.
- For UniDiffuser, query attack steps were reduced from 8 to 4 due to computational constraints (4 RGF steps * 100 perturbations = 400 queries per image, requiring ~30 minutes on an NVIDIA 3090).
- For Moondream, the original 8 steps were maintained.