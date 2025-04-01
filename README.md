# AttackVLM Reproduction Study

This repository contains the code for reproducing and extending the [AttackVLM](https://github.com/yunqing-me/AttackVLM) methodology, which evaluates the robustness of large vision-language models (VLMs) against adversarial image-grounded text generation.

### Reproduction Setup

1. Sample random images from the ILSVRC2012 validation dataset paired with random captions from the COCO dataset using `sample_clean_images.py` and `sample_coco_captions.py`

2. Generate images conditioned on target text using Stable Diffusion.
Use `stable_diffusion_setup.sh` and `txt2img_coco.py`

3. Perform tranfer attacks using `common\tranfer_attacks\` and calculate score on white-box attack (`common\baseline surrogate scores.py`)

4. Generate captions using victim model (`<model>\generate_captions.py`). In this reproduction 2 models were tested: UniDiffuser and Moondream. Calculate baseline victim scores (`common\baseline_victim_scores.py`)

5. Perform query attack (`<model>\query_attack.sh`)

### Implementation Details
- Pixel values and perturbation parameters were normalized between [0,1].
- Hyperparameters were set according to the original paper.
- For UniDiffuser, query attack steps were reduced from 8 to 4 due to computational constraints (4 RGF steps * 100 perturbations = 400 queries per image, requiring ~30 minutes on an NVIDIA 3090).
- For Moondream, the original 8 steps were maintained.
