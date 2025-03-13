import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from unet_gen_captions import generate_captions
import wandb

### unidiff lib
import ml_collections
import utils
from absl import logging
import libs.autoencoder
import libs.clip
from libs.caption_decoder import CaptionDecoder
from torchvision.transforms import InterpolationMode
### ###########

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use device:", device)
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
# ------------------------------------------------------------------ #  

def d(**kwconfig):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwconfig)

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
    ]
)
BICUBIC = InterpolationMode.BICUBIC

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)

def find_max(arr):
    max_value = float('-inf')
    for num in arr:
        if num > max_value:
            max_value = num
    return max_value

def load_models(config):
    nnet = utils.get_nnet(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()

    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    clip_model, orig_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_img_size = orig_preprocess.transforms[0].size

    # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # orig_preprocess expects PIL or np.array, clip_model_img_preprocess works with input torch.Tensor in range [0, 255]
    clip_model_img_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=clip_img_size, interpolation=torchvision.transforms.BICUBIC),
        torchvision.transforms.CenterCrop(size=clip_img_size),
        torchvision.transforms.Lambda(lambda img: img / 255),
        torchvision.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    return nnet, caption_decoder, autoencoder, clip_model, clip_model_img_preprocess

def compute_clip_features(texts, clip_model):
    with torch.no_grad():
        text_token = clip.tokenize(texts).to(device)
        text_features = clip_model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.detach()


def main():
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnet_path", default="models/uvit_v1.pth", type=str)
    parser.add_argument("--mode", default="i2t", type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--output", default="temp", type=str)
    parser.add_argument("--adv_data_path", default="temp", type=str)
    parser.add_argument("--clean_data_path", default="temp", type=str)
    parser.add_argument("--adv_text_path", default="temp", type=str)
    parser.add_argument("--tgt_text_path", default="temp", type=str)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--save_img", action='store_true')
    parser.add_argument("--num_query", default=5, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=16, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    
    config = parser.parse_args()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.text_dim
    )

    config.nnet = d(
        name='uvit_multi_post_ln_v1',
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.text_dim,
        num_text_tokens=77,
        clip_img_dim=config.clip_img_dim,
        use_checkpoint=True
    )
    config.sample = d(
        sample_steps=50,
        scale=7.,
        t2i_cfg_mode='true_uncond'
    )
    
    nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess = load_models(config)

    num_sub_query, num_query, sigma = config.num_sub_query, config.num_query, config.sigma
    batch_size    = config.batch_size
    alpha         = config.alpha
    epsilon       = config.epsilon
    vit_adv_data  = ImageFolderWithPaths(config.adv_data_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=24)

    # clean img, same size as clip img encoder
    clean_data    = ImageFolderWithPaths(config.clean_data_path, transform=transform)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=24)

    # org text/features
    adv_vit_text_path = config.adv_text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        unidiff_text_of_adv_vit  = f.readlines()[:config.num_samples]
        f.close()

    adv_vit_text_features = compute_clip_features(unidiff_text_of_adv_vit, clip_model)
    
    # tgt text/features
    tgt_text_path = config.tgt_text_path
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:config.num_samples] 
        f.close()
    
    # clip text features of the target
    target_text_features = compute_clip_features(tgt_text, clip_model)

    # baseline results
    vit_attack_results   = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    query_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    assert (vit_attack_results == query_attack_results).all()

    if config.wandb:
        run = wandb.init(project=config.wandb_project_name, reinit=True)
    
    for i, ((image, _, path), (image_clean, _, _)) in enumerate(zip(data_loader, clean_data_loader)):
        if batch_size * (i+1) > config.num_samples:
            break

        image = image.to(device)  # size=(10, 3, 224, 224)
        print(image.shape)
        image_clean = image_clean.to(device)  # size=(10, 3, 224, 224)
        print("image", image.shape, image_clean.shape)
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        print("txt features shape", adv_text_features.shape, tgt_text_features.shape)
        
        # ------------------- random gradient-free method
        print("init delta with diff(adv-clean)")
        delta = torch.tensor((image - image_clean))
        torch.cuda.empty_cache()
        print("delta.shape", delta.shape)
        
        best_caption = unidiff_text_of_adv_vit[i]
        better_flag = 0
        for step_idx in range(config.steps):
            print(f"{i}-th image / {step_idx}-th step")
            # step 1. obtain purturbed images
            
            ##########
            ##########
            with torch.no_grad():
                if step_idx == 0:
                    image_repeat      = image.repeat(num_query, 1, 1, 1)
                else:
                    image_repeat      = adv_image_in_current_step.repeat(num_query, 1, 1, 1)             

                    unidiff_text_of_adv_vit_in_current_step = generate_captions(config, image_repeat, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
                    adv_vit_text_features_in_current_step = compute_clip_features(unidiff_text_of_adv_vit_in_current_step, clip_model)
                    adv_text_features     = adv_vit_text_features_in_current_step
                    torch.cuda.empty_cache()
            ###########
            ###########
            print("image_repeat.shape", image_repeat.shape)
            
            query_noise            = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, 224, 224)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                print("sub_perturbed_image_repeat", sub_perturbed_image_repeat.shape)
                with torch.no_grad():
                    text_of_sub_perturbed_imgs = generate_captions(config, sub_perturbed_image_repeat, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
            
            # step 2. estimate grad
            perturb_text_features = compute_clip_features(text_of_perturbed_imgs, clip_model)
            coefficient     = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)
            coefficient     = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise     = query_noise.reshape(num_query, batch_size, 3, 224, 224)
            pseudo_gradient = coefficient * query_noise / sigma
            pseudo_gradient = pseudo_gradient.mean(0)
            
            # step 3. log metrics
            with torch.no_grad():

                delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
                delta.data = delta_data
                adv_image_in_current_step = torch.clamp(image_clean+delta, 0.0, 255.0)
                
                print(f"{i}-th img // {step_idx}-th step // max  delta", torch.max(torch.abs(delta)).item())
                print(f"{i}-th img // {step_idx}-th step // mean delta", torch.mean(torch.abs(delta)).item())
                
                wandb.log(
                    {
                        "max delta:":  torch.max(torch.abs(delta)).item(),
                        "mean delta:": torch.mean(torch.abs(delta)).item()
                    }
                )
                unidiff_text_of_adv_image_in_current_step = generate_captions(config, adv_image_in_current_step, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess)
                unidiff_text_features_of_adv_image_in_current_step = compute_clip_features(unidiff_text_of_adv_image_in_current_step, clip_model)

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(unidiff_text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                    best_caption = unidiff_text_of_adv_image_in_current_step[0]
                    better_flag  = 1

                torch.cuda.empty_cache()

        # log clip score after query
        if config.wandb:
            wandb.log(
                {
                    "moving-avg-adv-vitb32"  : np.mean(vit_attack_results[:(i+1)]),
                    "moving-avg-query-vitb32": np.mean(query_attack_results[:(i+1)]),
                }
            )

        # log text after query
        print("best caption of current image:", best_caption)
        with open(os.path.join(config.output + '.txt'), 'a') as f:
            if better_flag:
                f.write(best_caption+'\n')
            else:
                f.write(best_caption)
        f.close()

if __name__ == "__main__":
    main()