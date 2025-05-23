import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm

from common.utils import seedEverything, get_main_preprocess, ImageFolderWithPaths
from img2txt import generate_captions, load_models

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    config = OmegaConf.from_cli()
    config.unidif = OmegaConf.load("unidif_config.yaml")
    seedEverything(config.seed)

    nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess = load_models(config.unidif, device)

    transform = get_main_preprocess(clip_model_img_preprocess.transforms[0].size)
    dataset  = ImageFolderWithPaths(config.images_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    os.makedirs(config.output_path, exist_ok=True)
    for i, (images, _, _) in enumerate(tqdm(data_loader)):
        if config.batch_size * (i + 1) > config.num_samples:
            break

        images = images.to(device)
        batch_captions = generate_captions(config.unidif, images, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)

        with open(os.path.join(config.output_path, f'unidiff_captions.txt'), 'a') as f:
            f.write('\n'.join(batch_captions))
            f.write('\n')

if __name__ == "__main__":
    main()
