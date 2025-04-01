import torch
from PIL import Image
import clip
import os
import pickle
import pandas as pd
from omegaconf import OmegaConf

from utils import prepare_texts, get_texts_clip_features, seedEverything

device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_images(image_folder, preprocess, num_samples):
    folder_path = os.path.join(image_folder, "samples")

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)][:num_samples]
    images = [preprocess(Image.open(path)) for path in image_paths]
    return torch.stack(images).to(device)

def compute_clip_score(image_folder, text_file):
    # Load CLIP model and processor
    clip_model, preprocessor = clip.load("ViT-B/32", device=device, jit=False)
    tgt_text = prepare_texts(text_file, 500)
    text_features = get_texts_clip_features(tgt_text, clip_model, device)

    image_tensor = prepare_images(image_folder, preprocessor, num_samples=500)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity (CLIP score)
    clip_scores = torch.sum(image_features * text_features, dim=1).squeeze().detach().cpu().numpy()

    return clip_scores


config = OmegaConf.from_cli()
seedEverything(config.seed)

text_file = config.tgt_texts

image_sources = {
    "clean": config.clean_imgs_path,
    "img_img": config.transfer_ii_imgs_path,
    "img_text": config.transfer_it_imgs_path,
    "gen": config.gen_imgs_path
}

results = {}
os.makedirs(config.output_path, exist_ok=True)

for source_type, image_folder in image_sources.items():
    results[source_type] = compute_clip_score(image_folder, text_file)
with open(os.path.join(config.output_path,"surrogate_results.pkl"), "wb") as f:
    pickle.dump(results, f)

    t = pd.DataFrame.from_dict(results)
    print(t.mean())
    print()
