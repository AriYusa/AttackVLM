import os
import clip
import numpy as np
import pandas as pd
import torch
import torchvision
from img2txt import generate_captions, load_model
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from common.utils import seedEverything, ImageFolderWithPaths, \
    get_main_preprocess, get_texts_clip_features, prepare_texts, compute_clip_scores

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    config = OmegaConf.from_cli()

    seedEverything(config.seed)

    model = load_model()

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
    clip_model = clip_img_model_vitb32

    num_sub_query, num_query = config.num_sub_query, config.num_query
    batch_size = config.batch_size
    sigma = config.sigma / 255
    alpha = config.alpha / 255
    epsilon = config.epsilon / 255

    transform = get_main_preprocess(config.resolution)
    vit_adv_data = ImageFolderWithPaths(config.adv_data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # clean img, same size as clip img encoder
    clean_data = ImageFolderWithPaths(config.clean_data_path, transform=transform)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=4)

    transfer_attack_texts = prepare_texts(config.adv_text_path, config.num_samples)
    tgt_texts = prepare_texts(config.tgt_text_path, config.num_samples)

    # Compute baseline similarity
    baseline_scores = compute_clip_scores(transfer_attack_texts, tgt_texts, clip_models, device)

    if config.wandb_project_name:
        run = wandb.init(project=config.wandb_project_name, reinit=True,
                         config=OmegaConf.to_container(config, resolve=True))

    def get_victim_captions_for_preturbed(perturbed_images):
        texts = []
        for query_idx in range(num_query * batch_size // num_sub_query):
            batch_preturbed_images = perturbed_images[num_sub_query * query_idx: num_sub_query * (query_idx + 1)]
            with torch.no_grad():
                batch_captions = generate_captions(model, batch_preturbed_images)
            texts.extend(batch_captions)
        return texts

    # Initialize empty DataFrame for logging
    columns = ["img_id", "best_caption",
               "best_rgf_step"]
    columns.extend([f"{model_name}_baseline_score" for model_name in clip_models.keys()])
    columns.extend([f"{model_name}_query_attack_score" for model_name in clip_models.keys()])
    log_df = pd.DataFrame(columns=columns)
    os.makedirs(config.output_path, exist_ok=True)

    for batch_i, ((adv_images, _, path), (clean_images, _, _)) in enumerate(zip(data_loader, clean_data_loader)):
        if batch_size * (batch_i + 1) > config.num_samples:
            break
        adv_images = adv_images.to(device)
        clean_images = clean_images.to(device)

        # obtain all text features (via CLIP text encoder)
        batch_tgt_texts = tgt_texts[batch_size * batch_i: batch_size * (batch_i + 1)]
        tgt_text_features = get_texts_clip_features(batch_tgt_texts, clip_model, device)

        # ------------------- random gradient-free method
        delta = adv_images - clean_images
        wandb.log(
            {
                "real mean delta:": torch.mean(torch.abs(delta)).item(),
                "batch_idx": batch_i,
            })


        best_captions = [""] * batch_size
        best_step_info = np.zeros(batch_size)
        best_clip_scores = {}
        query_attack_results = np.zeros(batch_size)
        best_adv_images = adv_images

        with torch.no_grad():
            victim_captions = generate_captions(model, adv_images)
            adv_text_features = get_texts_clip_features(victim_captions, clip_model, device)
            torch.cuda.empty_cache()

        for step_idx in tqdm(range(config.rgf_steps)):
            # step 1. obtain captions of perturbed images
            images_repeat = adv_images.repeat(num_query, 1, 1, 1)  # size = (num_query x batch_size, 3, 224, 224)
            query_noise = torch.randn_like(images_repeat).sign()
            perturbed_versions = torch.clamp(images_repeat + (sigma * query_noise), 0,
                                             1)  # size = (num_query x batch_size, 3, 224, 224)
            text_of_perturbed_imgs = get_victim_captions_for_preturbed(perturbed_versions)
            perturb_text_features = get_texts_clip_features(text_of_perturbed_imgs,
                                                            clip_model, device)  # (num_query x batch_size, 768)

            # step 2. estimate grad
            coefficient = torch.sum(
                (perturb_text_features - adv_text_features.repeat(num_query, 1)) * tgt_text_features.repeat(num_query,
                                                                                                            1), dim=-1)
            mean_coef = coefficient.mean(dim=0)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, 224, 224)
            pseudo_gradient = coefficient * query_noise / sigma  # [-1, 1] * [-1, 1] * 255 / 8
            pseudo_gradient = pseudo_gradient.mean(0)

            # step 3. log metrics
            with torch.no_grad():
                delta = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
                adv_images = torch.clamp(clean_images + delta, 0, 1)

            victim_captions = generate_captions(model, adv_images)
            adv_text_features = get_texts_clip_features(victim_captions, clip_model, device)

            clip_scores = compute_clip_scores(victim_captions, batch_tgt_texts, clip_models, device)
            if best_clip_scores == {}:
                best_clip_scores = clip_scores.copy()

            wandb.log(
                {
                    "max delta:": torch.max(torch.abs(delta)).item(),
                    "mean delta:": torch.mean(torch.abs(delta)).item(),
                    "coefficient": mean_coef,
                    "batch_idx": batch_i,
                    "clip_scores": clip_scores["ViT-B/32"].mean().item(),
                    "step_idx": step_idx,
                })

            # update results
            is_better = clip_scores["ViT-B/32"] > query_attack_results
            query_attack_results = np.where(is_better, clip_scores["ViT-B/32"], query_attack_results)
            best_captions = np.where(is_better, victim_captions, best_captions)
            best_step_info = np.where(is_better, step_idx, best_step_info)
            best_adv_images[is_better] = adv_images[is_better]
            for model_name in clip_models.keys():
                best_clip_scores[model_name] = np.where(is_better, clip_scores[model_name],
                                                        best_clip_scores[model_name])

            torch.cuda.empty_cache()

        print(best_captions)
        print(batch_tgt_texts)

        batch_logs = []
        for img_j in range(batch_size):
            obj_idx = batch_i * batch_size + img_j

            img_info = {
                "img_id": obj_idx,
                "best_caption": best_captions[img_j],
                "best_rgf_step": best_step_info[img_j],
            }
            for model_name in baseline_scores.keys():
                img_info.update({
                    f"{model_name}_baseline_score": baseline_scores[model_name][obj_idx],
                    f"{model_name}_query_attack_score": best_clip_scores[model_name][img_j],
                })
            batch_logs.append(img_info)

        # Append batch logs to the DataFrame
        batch_df = pd.DataFrame(batch_logs)
        log_df = pd.concat([log_df, batch_df], ignore_index=True)
        log_df.to_csv(os.path.join(config.output_path, "experiment_log.csv"), index=False)

        for path_idx in range(len(path)):
            path_parts = path[path_idx].split('/')
            folder, name = path_parts[-2], path_parts[-1]
            folder_to_save = os.path.join(config.output_path, folder)
            os.makedirs(folder_to_save, exist_ok=True)
            save_path = os.path.join(folder_to_save, name.replace('JPEG', 'png') if 'JPEG' in name else name)
            torchvision.utils.save_image(best_adv_images[path_idx], save_path)


if __name__ == "__main__":
    main()
