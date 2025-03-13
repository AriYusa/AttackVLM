import torch
import utils
import os
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import time
from torchvision.transforms.functional import center_crop

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, images_batch, clip_img_model, clip_img_model_preprocess, autoencoder):
    # image [batch, 3, 224, 224]
    resolution = config.z_shape[-1] * 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # img_contexts = torch.randn(batch_size, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    # clip_imgs = torch.randn(batch_size, 1, config.clip_img_dim)

    def get_img_feature(images_batch):
        images_batch = center_crop(images_batch, resolution)
        clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(images_batch).to(device)).unsqueeze(1)   # [batch_size, 1, 512]

        images_batch = (images_batch / 127.5 - 1.0)  # [0, 255] to [-1, 1]
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

    logging.info(config.sample)
    logging.info(f'N={N}')

    clip_imgs, img_contexts = prepare_contexts(config, images_batch, clip_model, clip_model_img_preprocess, autoencoder)
    logging.info(f'img_contexts: {img_contexts.shape}')
    logging.info(f'clip_imgs: {clip_imgs.shape}')

    z_img = autoencoder.sample(img_contexts)
    logging.info(f'z_img: {z_img.shape}')

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
                logging.info(f' sample_fn x: {x.shape}')
                end_time = time.time()
                print(f'\ngenerate {batch_size} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

        os.makedirs(config.output_path, exist_ok=True)
        return x

    _text = sample_fn(z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
    samples = caption_decoder.generate_captions(_text)
    logging.info(samples)
    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, f'{config.mode}.txt'), 'w') as f:
        print('\n'.join(samples), file=f)

    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
    print(f'\nresults are saved in {os.path.join(config.output_path)} :)')

    with torch.no_grad():
        # Rest of your operations that might need clearing
        torch.cuda.empty_cache()

    return samples