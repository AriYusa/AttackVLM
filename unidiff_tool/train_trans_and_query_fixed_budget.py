import os
import clip
import numpy as np
import torch
from unidif_gen_captions import generate_captions
import wandb
from omegaconf import OmegaConf

from common_utils import seedEverything, ImageFolderWithPaths, \
    get_main_preprocess, load_models

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_texts_clip_features(texts, clip_model):
    with torch.no_grad():
        text_token = clip.tokenize(texts).to(device)
        text_features = clip_model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.detach()


def main():
    config = OmegaConf.load("trans_and_query_fixed_budget_config.yaml")
    config.unidif = OmegaConf.load("unidif_config.yaml")
    config = OmegaConf.merge(config, OmegaConf.from_cli())

    seedEverything(config.seed)

    nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess = load_models(config.unidif, device)

    num_sub_query, num_query, sigma = config.num_sub_query, config.num_query, config.sigma
    batch_size    = config.batch_size
    alpha         = config.alpha
    epsilon       = config.epsilon

    transform = get_main_preprocess(224)
    vit_adv_data  = ImageFolderWithPaths(config.adv_data_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # clean img, same size as clip img encoder
    clean_data    = ImageFolderWithPaths(config.clean_data_path, transform=transform)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # org text/features
    if not isinstance(config.adv_text_path, str):
        adv_vit_text_path = list(config.adv_text_path)
        adv_vit_text_path=os.path.join(*adv_vit_text_path)
    else:
        adv_vit_text_path=config.adv_text_path
    with open(adv_vit_text_path, 'r') as f:
        unidiff_text_of_adv_vit  = f.readlines()[:config.num_samples]
        f.close()

    adv_vit_text_features = get_texts_clip_features(unidiff_text_of_adv_vit, clip_model)
    
    # tgt text/features
    tgt_text_path = config.tgt_text_path
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:config.num_samples] 
        f.close()
    
    # clip text features of the target
    target_text_features = get_texts_clip_features(tgt_text, clip_model)

    # baseline results
    transfer_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    query_attack_results = np.zeros_like(transfer_attack_results)

    if config.wandb_project_name:
        run = wandb.init(project=config.wandb_project_name, reinit=True, config=OmegaConf.to_container(config, resolve=True))
        table = wandb.Table(columns=[ "obj_idx", "transfer_caption", "transfer_score", "best_query_caption", "best_query_score", "best_step_idx"])

    def get_victim_captions_for_preturbed(perturbed_images):
        texts = []
        for query_idx in range(num_query*batch_size//num_sub_query):
            batch_preturbed_images = perturbed_images[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
            with torch.no_grad():
                batch_captions = generate_captions(config.unidif, batch_preturbed_images, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
            texts.extend(batch_captions)
        return texts
    
    for batch_i, ((adv_images, _, path), (clean_images, _, _)) in enumerate(zip(data_loader, clean_data_loader)):
        if batch_size * (batch_i+1) > config.num_samples:
            break
        adv_images = adv_images.to(device)
        clean_images = clean_images.to(device)
        # obtain all text features (via CLIP text encoder)
        tgt_text_features = target_text_features[batch_size * batch_i: batch_size * (batch_i+1)]

        # ------------------- random gradient-free method
        delta = adv_images - clean_images
        torch.cuda.empty_cache()

        best_captions = [""]*batch_size
        best_step_info = np.zeros(batch_size)

        with torch.no_grad():
            unidiff_captions = generate_captions(config.unidif, adv_images, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
            adv_text_features = get_texts_clip_features(unidiff_captions, clip_model)
            torch.cuda.empty_cache()

        for step_idx in range(config.rgf_steps):
            print(f"------- BATCH {batch_i} / STEP {step_idx} --------")
            # step 1. obtain captions of purturbed images
            images_repeat = adv_images.repeat(num_query, 1, 1, 1) # size = (num_query x batch_size, 3, 224, 224)
            query_noise = torch.randn_like(images_repeat).sign() # Rademacher noise
            perturbed_versions= torch.clamp(images_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, 224, 224)
            text_of_perturbed_imgs = get_victim_captions_for_preturbed(perturbed_versions)
            perturb_text_features = get_texts_clip_features(text_of_perturbed_imgs,
                                                            clip_model)  # should be (num_query x batch_size, 768)

            # step 2. estimate grad
            coefficient = torch.sum((perturb_text_features - adv_text_features.repeat(num_query, 1)) * tgt_text_features.repeat(num_query, 1), dim=-1)
            if config.wandb_project_name:
                wandb.log(
                {
                    "coefficient": coefficient.mean(dim=0).item(),
                    "step_idx": step_idx,
                })
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, 224, 224)
            pseudo_gradient = coefficient * query_noise / sigma
            pseudo_gradient = pseudo_gradient.mean(0)

            # step 3. log metrics
            with torch.no_grad():

                delta = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
                adv_images = torch.clamp(clean_images+delta, 0.0, 255.0)

                wandb.log(
                    {
                        "max delta:":  torch.max(torch.abs(delta)).item(),
                        "mean delta:": torch.mean(torch.abs(delta)).item(),
                        "batch_idx": batch_i,
                        "step_idx": step_idx,
                    }
                )
                unidiff_captions = generate_captions(config.unidif, adv_images, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
                adv_text_features = get_texts_clip_features(unidiff_captions, clip_model)

                clip_scores = torch.sum(adv_text_features * tgt_text_features, dim=1)
                wandb.log(
                    {
                        "clip_scores": clip_scores.mean(dim=0).item(),
                        "step_idx": step_idx,
                    })

                # update results
                for j, score in enumerate(clip_scores):
                    if score > query_attack_results[batch_i*batch_size+j]:
                        query_attack_results[batch_i * batch_size + j] = score.item()
                        best_captions[j] = unidiff_captions[j]
                        best_step_info[j] = step_idx

            torch.cuda.empty_cache()

        # log clip score after query
        if config.wandb_project_name:
            wandb.log(
                {
                    "moving-avg-transfer-vitb32"  : np.mean(transfer_attack_results[:batch_size*(batch_i+1)]),
                    "moving-avg-query-vitb32": np.mean(query_attack_results[:batch_size*(batch_i+1)]),
                }
            )

            # log information to wandb table
            for j in range(batch_size):
                obj_idx = batch_i * batch_size + j
                table.add_data(
                    obj_idx,
                    unidiff_text_of_adv_vit[obj_idx],
                    transfer_attack_results[obj_idx],
                    best_captions[j],
                    query_attack_results[obj_idx],
                    best_step_info[j]
                )

        # log text after query
        print("best captions after query attack:", best_captions)
        os.makedirs(config.output_path, exist_ok=True)
        with open(os.path.join(config.output_path, "captions.txt"), 'a') as f:
            for best_caption in best_captions:
                f.write(f"{best_caption}\n")

        if config.wandb_project_name:
            wandb.log({"results_table": table})

if __name__ == "__main__":
    main()
