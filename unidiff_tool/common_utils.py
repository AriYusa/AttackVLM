import os
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode

# unidiff_imports
import clip
import utils
import libs.autoencoder
import libs.clip
from libs.caption_decoder import CaptionDecoder

BICUBIC = InterpolationMode.BICUBIC

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def get_main_preprocess(resolution):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=resolution, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(size=resolution),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

def get_clip_model_img_preprocess(resolution):
    # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # original clip preprocess expects PIL or np.array,but this preprocessor works with input torch.Tensor in range [0, 255]
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=resolution, interpolation=BICUBIC),
        torchvision.transforms.CenterCrop(size=resolution),
        torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255),
        torchvision.transforms.Normalize(
            mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        ),
    ])


def load_models(config, device):
    config.z_shape = tuple(config.z_shape)
    nnet = utils.get_nnet(**config.nnet)
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()

    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    clip_model, orig_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_img_size = orig_preprocess.transforms[0].size
    clip_model_img_preprocess = get_clip_model_img_preprocess(clip_img_size)

    return nnet, caption_decoder, autoencoder, clip_model, clip_model_img_preprocess
