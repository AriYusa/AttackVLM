import os
import random
import requests
import json
import zipfile
import io
import pandas as pd

random.seed(31)

def download_coco_captions():
    """
    Download COCO captions for 2017 dataset.
    """
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    print(f"Downloading captions from {url}...")
    response = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(response.content))

    if not os.path.exists('coco_data'):
        os.makedirs('coco_data')

    for file in z.namelist():
        if file in ['annotations/captions_train2017.json', 'annotations/captions_val2017.json']:
            z.extract(file, 'coco_data')
            print(f"Extracted {file}")

    return os.path.join('coco_data', 'annotations', 'captions_val2017.json')

def load_coco_captions(captions_file):
    """
    Load COCO captions from JSON file.
    """
    with open(captions_file, 'r') as f:
        return json.load(f)

def get_image_caption_pairs(captions_data, selected_image_info):
    """
    Get image-caption pairs for selected images.
    """
    selected_images = {img["id"]: img['coco_url'] for img in selected_image_info}

    found_captions ={}
    # Go through all captions and add those for our selected images
    for caption in captions_data['annotations']:
        if caption['image_id'] in set(selected_images.keys()) - set(found_captions.keys()):
            img_id = caption['image_id']
            found_captions[img_id] = {"coco_url": selected_images[img_id], "caption": caption['caption']}
    return found_captions

def save_captions_to_csv(caption_data):
    """Saves image captions"""

    rows = []
    for image_id, data in caption_data.items():
        rows.append([image_id, data['coco_url'], data['caption']])

    df = pd.DataFrame(rows, columns=["image_id", "coco_url", "caption"])
    filename = "coco_captions_info.csv"
    df.to_csv(filename, index=False)

    with open("coco_captions.txt", "w") as f:
        for caption in caption_data.values():
            f.write(caption['caption'] + "\n")


if __name__ == "__main__":
    captions_file = download_coco_captions()
    captions_data = load_coco_captions(captions_file)
    selected_image_info = random.sample(captions_data['images'], 5000)
    caption_data = get_image_caption_pairs(captions_data, selected_image_info)
    save_captions_to_csv(caption_data)