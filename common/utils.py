import os
import random
from os import truncate

import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode
import clip

BICUBIC = InterpolationMode.BICUBIC

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 31

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
            torchvision.transforms.ToTensor(),
        ]
    )

def get_clip_model_img_preprocess(resolution):
    # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # original clip preprocess expects PIL or np.array,but this preprocessor works with input torch.Tensor in range [0, 255]
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=resolution, interpolation=BICUBIC),
        torchvision.transforms.CenterCrop(size=resolution),
        torchvision.transforms.Normalize(
            mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        ),
    ])

def get_img_clip_features(images, clip_model, preprocess):
    """Extract normalized image features using the CLIP model."""
    image_features = clip_model.encode_image(preprocess(images))
    return image_features / image_features.norm(dim=1, keepdim=True)

def compute_clip_scores(adv_texts, tgt_texts, clip_models, device):
    results = {}
    for model_name, model in clip_models.items():
        adv_features = get_texts_clip_features(adv_texts, model, device)
        tgt_features = get_texts_clip_features(tgt_texts, model, device)
        results[model_name] = torch.sum(adv_features * tgt_features, dim=1).squeeze().detach().cpu().numpy()
    return results

def get_texts_clip_features(texts, clip_model, device):
    with torch.no_grad():
        text_token = clip.tokenize(texts, truncate=True).to(device)
        text_features = clip_model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.detach()

def prepare_texts(text_path, num_samples):
    """Preprocess texts from file"""
    with open(text_path, 'r') as f:
        texts  = f.readlines()[:num_samples]
        texts = [text.strip().strip(".") for text in texts]
    return texts