import os
import clip
import pandas as pd
import torch
from omegaconf import OmegaConf
import pickle

from utils import seedEverything, prepare_texts, compute_clip_scores

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    config = OmegaConf.from_cli()

    seedEverything(config.seed)

    # Load additional CLIP models
    clip_img_model_rn50, _ = clip.load("RN50", device=device, jit=False)
    clip_img_model_rn101, _ = clip.load("RN101", device=device, jit=False)
    clip_img_model_vitb16, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_img_model_vitl14, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_models = {
        "RN50": clip_img_model_rn50,
        "RN101": clip_img_model_rn101,
        "ViT-B/16": clip_img_model_vitb16,
        "ViT-L/14": clip_img_model_vitl14,
        "ViT-B/32": clip_img_model_vitb32,
    }

    clean_img_text = prepare_texts(config.clean_img_captions, config.num_samples)
    gen_gen_text = prepare_texts(config.gen_img_captions, config.num_samples)
    img_img_text = prepare_texts(config.transfer_ii_captions, config.num_samples)
    img_text_text = prepare_texts(config.transfer_it_captions, config.num_samples)


    tgt_text = prepare_texts(config.tgt_texts, config.num_samples)

    texts_a = {
        "clean": clean_img_text,
        "img_img": img_img_text,
        "img_text": img_text_text,
        "gen": gen_gen_text
    }
    texts_b = {"tgt":tgt_text, "gen":gen_gen_text}

    os.makedirs(config.output_path, exist_ok=True)
    for a in texts_a:
        for b in texts_b:
            baseline_scores = compute_clip_scores(texts_a[a], texts_b[b], clip_models, device)
            with open(os.path.join(config.output_path, f"{a}_with_{b}.pkl"), "wb") as f:
                pickle.dump(baseline_scores, f)

            print(a, b)
            t = pd.DataFrame.from_dict(baseline_scores)
            print(t.mean(), t.std())

if __name__ == "__main__":
    main()
