import torch
import time
from torchvision.transforms.functional import center_crop
import clip

# unififf imports
import utils
import libs.autoencoder
import libs.clip
from libs.caption_decoder import CaptionDecoder
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver

from common.utils import get_clip_model_img_preprocess


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

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, images_batch, clip_img_model, clip_img_model_preprocess, autoencoder):
    # image [batch, 3, 224, 224]
    resolution = config.z_shape[-1] * 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_img_feature(images_batch):
        images_batch = center_crop(images_batch, resolution)
        clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(images_batch).to(device)).unsqueeze(1)   # [batch_size, 1, 512]

        images_batch = images_batch * 2 - 1.0  # [0, 1] to [-1, 1]
        moments = autoencoder.encode_moments(images_batch)  # [batch_size, 2*4, 64, 64]

        return clip_img_feature, moments

    clip_img, img_context = get_img_feature(images_batch)
    return clip_img, img_context

def generate_captions(config, images_batch, nnet, caption_decoder,autoencoder, clip_model, clip_model_img_preprocess):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    utils.set_logger(log_level='info')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    def i2t_nnet(x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        _, _, text_out = nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=device, dtype=torch.int) + config.data_type)
        if config.sample.scale == 0.:
            return text_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        _, _, text_out_uncond = nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        return text_out + config.sample.scale * (text_out - text_out_uncond)

    clip_imgs, img_contexts = prepare_contexts(config, images_batch, clip_model, clip_model_img_preprocess, autoencoder)

    z_img = autoencoder.sample(img_contexts)

    def sample_fn(z, clip_img):
        batch_size = clip_img.shape[0]
        _text_init = torch.randn(batch_size, 77, config.text_dim, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return i2t_nnet(x, t, z, clip_img)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                start_time = time.time()
                x = dpm_solver.sample(_text_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {batch_size} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

        return x

    _text = sample_fn(z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
    samples = caption_decoder.generate_captions(_text)

    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')

    with torch.no_grad():
        # Rest of your operations that might need clearing
        torch.cuda.empty_cache()

    return samples
