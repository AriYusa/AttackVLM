import torch
import os
from omegaconf import OmegaConf

from common_utils import seedEverything, load_models, get_main_preprocess, ImageFolderWithPaths
from unidif_gen_captions import generate_captions

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    config = OmegaConf.from_cli()
    config.unidif = OmegaConf.load("unidif_config.yaml")
    seedEverything(config.seed)

    nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess = load_models(config.unidif, device)

    transform = get_main_preprocess(224)
    dataset  = ImageFolderWithPaths(config.images_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    all_captions = []
    for i, (images, _, _) in enumerate(data_loader):
        if config.batch_size * (i + 1) > config.num_samples:
            break

        images = images.to(device)
        batch_captions = generate_captions(config.unidif, images, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
        all_captions.extend(batch_captions)

    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, f'captions.txt'), 'w') as f:
        f.write('\n'.join(all_captions))

if __name__ == "__main__":
    main()
