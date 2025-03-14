import argparse
import os
import clip
import torch
import torchvision
from torch.utils.data import DataLoader
import time
from omegaconf import OmegaConf

from common_utils import seedEverything, ImageFolderWithPaths, get_main_preprocess, \
    get_clip_model_img_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_img_clip_features(images, clip_model, preprocess):
    """Extract normalized image features using the CLIP model."""
    image_features = clip_model.encode_image(preprocess(images))
    return image_features / image_features.norm(dim=1, keepdim=True)

def main(args):
    if args.wandb_project_name:
        import wandb
        wandb.init(project=args.wandb_project_name, config=OmegaConf.to_container(args, resolve=True))

    seedEverything()

    # Load CLIP model and preprocess
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)
    clip_preprocess = get_clip_model_img_preprocess(preprocess.transforms[0].size)

    transform = get_main_preprocess(args.input_res)
    clean_data = ImageFolderWithPaths(args.cle_data_path, transform=transform)
    clean_data_loader = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    target_data = ImageFolderWithPaths(args.tgt_data_path, transform=transform)
    target_data_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Start attack
    for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(clean_data_loader, target_data_loader)):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        start_time = time.time()

        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)

        with torch.no_grad():
            tgt_image_features = get_img_clip_features(image_tgt, clip_model, clip_preprocess)

        delta = torch.zeros_like(image_org, requires_grad=True)
        for step_idx in range(args.steps):
            adv_image = image_org + delta
            adv_image_features = get_img_clip_features(adv_image, clip_model, clip_preprocess)

            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))
            embedding_sim.backward()

            grad = delta.grad.detach()
            d = torch.clamp(delta + args.alpha * torch.sign(grad), min=-args.epsilon, max=args.epsilon)
            delta.data = d
            delta.grad.zero_()

            # Log progress
            print(f"iter {i}/{args.num_samples // args.batch_size} step:{step_idx:3d}, "
                  f"mean embedding similarity={embedding_sim.item():.5f}, "
                  f"max delta={torch.max(torch.abs(delta)).item():.3f}, "
                  f"mean delta={torch.mean(torch.abs(delta)).item():.3f}")
            if args.wandb_project_name:
                wandb.log({"batch": i,
                        "preturbation_step": step_idx,
                        "mean_embedding_similarity": embedding_sim.item(),
                        "max_delta": torch.max(torch.abs(delta)).item(),
                        "mean_delta": torch.mean(torch.abs(delta)).item()
                        })

        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

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

