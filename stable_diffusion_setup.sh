cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

cd AttackVLM
git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt

conda env create -f environment.yaml
source /root/miniconda3/bin/activate ldm

cd AttackVLM/stable-diffusion

mv ../txt2img_coco.py .
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev

python txt2img_coco.py \
        --ddim_eta 0.0 \
        --n_samples 3000 \
        --n_iter 1 \
        --scale 7.5 \
        --ddim_steps 50 \
        --plms \
        --skip_grid \
        --ckpt ./sd-v1-4-full-ema.ckpt \
        --from-file '../coco_captions.txt' \
        --outdir '../gen_images'