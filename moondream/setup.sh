cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
conda init
source ~/.bashrc

cd AttackVLM
export PYTHONPATH=$(pwd)
tar -xvzf selected_imagenet_images.tar.gz -C .
tar -xvzf img_text_transfer_images.tar.gz -C .
tar -xvzf img_img_transfer_images.tar.gz -C .
tar -xvzf gen_images.tar.gz -C .

sudo apt update -y
sudo apt install libvips42

cd moondream
conda create -n moondream python=3.11 -y
conda activate moondream
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow einops omegaconf pandas
pip install 'accelerate>=0.26.0'
pip install pyvips-binary pyvips
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

pip install wandb
export $(grep -v '^#' .env | xargs)
wandb login $WANDB_API_KEY
