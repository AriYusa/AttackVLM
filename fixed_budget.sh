cd AttackVLM/
git clone https://github.com/thu-ml/unidiffuser.git
cp ./unidiff_tool/* ./unidiffuser
cd unidiffuser

conda create -n unidiffuser python=3.9
conda activate unidiffuser
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops ftfy==6.1.1 transformers==4.23.1
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