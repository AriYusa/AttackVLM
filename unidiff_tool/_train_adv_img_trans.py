import argparse
import os
import clip
import torch
import torchvision

from unidiff_tool.images_adv_ii.utils import seedEverything, ImageFolderWithPaths, get_main_preprocess, \
    get_clip_model_img_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_img_clip_features(images, clip_model):
    image_features = clip_model.encode_image(clip_preprocess(images))
    return image_features / image_features.norm(dim=1, keepdim=True)

if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="temp", type=str, help='the folder name of output')
    
    parser.add_argument("--cle_data_path", default=None, type=str, help='path of the clean images')
    parser.add_argument("--tgt_data_path", default=None, type=str, help='path of the target images')
    args = parser.parse_args()
    
    # load clip_model params
    alpha = args.alpha
    epsilon = args.epsilon
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)

    # preprocess images
    transform = get_main_preprocess(224)
    clean_data    = ImageFolderWithPaths(args.cle_data_path, transform=transform)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    target_data   = ImageFolderWithPaths(args.tgt_data_path, transform=transform)
    target_data_loader = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clip_preprocess = get_clip_model_img_preprocess(preprocess.transforms[0].size)

    normalize = clip_preprocess.transforms[-1]
    inverse_normalize = torchvision.transforms.Normalize(mean=-normalize.mean/normalize.std, std=1/normalize.std)

    # start attack
    for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(clean_data_loader, target_data_loader)):
        if args.batch_size * (i+1) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)

        with torch.no_grad():
            tgt_image_features = get_img_clip_features(image_tgt, clip_model)

        delta = torch.zeros_like(image_org, requires_grad=True)
        for step_idx in range(args.steps):
            adv_image = image_org + delta
            adv_image_features = get_img_clip_features(image_tgt, clip_model)

            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))
            embedding_sim.backward()

            with torch.no_grad():
                grad = delta.grad
                delta += alpha * torch.sign(grad)
                delta = torch.clamp(delta, min=-epsilon, max=epsilon)
                delta.grad.zero_()

            # Log progress
            print(f"iter {i}/{args.num_samples // args.batch_size} step:{step_idx:3d}, "
                  f"mean embedding similarity={embedding_sim.item():.5f}, "
                  f"max delta={torch.max(torch.abs(delta)).item():.3f}, "
                  f"mean delta={torch.mean(torch.abs(delta)).item():.3f}"
            )

        # save imgs
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            if 'JPEG' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')
            elif 'png' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name))

