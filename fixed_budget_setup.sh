cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/root/miniconda3/bin:$PATH"
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
conda init
source ~/.bashrc

cd AttackVLM/
tar -xvzf selected_imagenet_images.tar.gz -C .
tar -xvzf img_text_transfer_images.tar.gz -C .

git clone https://github.com/thu-ml/unidiffuser.git
cp -r ./unidiff_tool/* ./unidiffuser
cd unidiffuser

conda create -n unidiffuser python=3.9 -y
conda activate unidiffuser
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py einops pandas omegaconf ftfy==6.1.1 transformers==4.23.1
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install -U --pre triton

mkdir models
cd models
wget -c https://huggingface.co/thu-ml/unidiffuser-v1/resolve/main/autoencoder_kl.pth
wget -c https://huggingface.co/thu-ml/unidiffuser-v1/resolve/main/caption_decoder.pth
wget -c https://huggingface.co/thu-ml/unidiffuser-v1/resolve/main/uvit_v1.pth
cd ..

pip install wandb
export $(grep -v '^#' .env | xargs)
wandb login $WANDB_API_KEY