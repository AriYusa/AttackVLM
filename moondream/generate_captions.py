import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm

from common.utils import seedEverything, get_main_preprocess, ImageFolderWithPaths

from moondream.img2txt import load_model, generate_captions

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    config = OmegaConf.from_cli()
    seedEverything(config.seed)

    model = load_model()

    transform = get_main_preprocess(config.resolution)
    dataset  = ImageFolderWithPaths(config.images_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    os.makedirs(config.output_path, exist_ok=True)
    for i, (images, _, _) in enumerate(tqdm(data_loader)):
        if config.batch_size * (i + 1) > config.num_samples:
            break

        batch_captions = generate_captions(model, images)

        with open(os.path.join(config.output_path, 'moondream_captions.txt'), 'a') as f:
            f.write('\n'.join(batch_captions))
            f.write('\n')

if __name__ == "__main__":
    main()
