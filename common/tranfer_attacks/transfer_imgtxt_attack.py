import os
import clip
import torch
import torchvision
from torch.utils.data import DataLoader
import time
from omegaconf import OmegaConf

from common.utils import seedEverything, ImageFolderWithPaths, get_main_preprocess, \
    get_clip_model_img_preprocess, get_texts_clip_features, get_img_clip_features, prepare_texts

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    if args.wandb_project_name:
        import wandb
        wandb.init(project=args.wandb_project_name, config=OmegaConf.to_container(args, resolve=True))

    seedEverything()

    alpha = args.alpha / 255
    epsilon = args.epsilon / 255

    # Load CLIP model and preprocess
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)
    clip_preprocess = get_clip_model_img_preprocess(preprocess.transforms[0].size)

    transform = get_main_preprocess(args.input_res)
    clean_data = ImageFolderWithPaths(args.cle_data_path, transform=transform)
    clean_data_loader = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    target_texts = prepare_texts(args.tgt_text_path, args.num_samples)

    # Start attack
    for i, (image_org, _, path) in enumerate(clean_data_loader):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        start_time = time.time()


        batch_tgt_texts = target_texts[args.batch_size * i: args.batch_size * (i + 1)]
        tgt_txt_features = get_texts_clip_features(batch_tgt_texts, clip_model, device)

        delta = torch.empty_like(image_org, device=device).uniform_(-epsilon, epsilon)
        delta.requires_grad = True

        image_org = image_org.to(device)
        adv_image = torch.clamp(image_org + delta, 0, 1)

        # PGD Iteration
        for step_idx in range(args.steps):
            adv_image_features = get_img_clip_features(adv_image, clip_model, clip_preprocess)

            cos_sim = torch.mean(torch.sum(adv_image_features * tgt_txt_features, dim=1))
            loss = cos_sim
            loss.backward()

            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()

            adv_image = torch.clamp(image_org + delta, 0, 1)

            if args.wandb_project_name:
                wandb.log({"batch": i,
                        "preturbation_step": step_idx,
                        "mean_embedding_similarity": cos_sim.item(),
                        "max_delta": torch.max(torch.abs(delta)).item(),
                        "mean_delta": torch.mean(torch.abs(delta)).item()
                        })

        # Log
        batch_time = time.time() - start_time
        if args.wandb_project_name:
            wandb.log({"batch": i, "batch_time": batch_time})

        for path_idx in range(len(path)):
            path_parts = path[path_idx].split('/')
            folder, name = path_parts[-2], path_parts[-1]
            folder_to_save = os.path.join(args.output_path, folder)
            os.makedirs(folder_to_save, exist_ok=True)
            save_path = os.path.join(folder_to_save, name.replace('JPEG', 'png') if 'JPEG' in name else name)
            torchvision.utils.save_image(adv_image[path_idx], save_path)

if __name__ == "__main__":
    config = OmegaConf.load("train_ii_transfer_config.yaml")
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    main(config)