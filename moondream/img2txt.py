import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms.functional import to_pil_image

VERSION = "2025-01-09"
def load_model():
    """Load Moondream model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=VERSION, trust_remote_code=True, device_map={"": "cuda"}
    )
    model.eval()
    return model

def generate_captions(model, images:torch.Tensor):
    captions = []
    for image in images:
        c = model.caption(to_pil_image(image), length="short")["caption"]
        captions.append(c.strip().strip('.'))
    return captions

