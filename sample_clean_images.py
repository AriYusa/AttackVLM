import argparse
import os
import random
import shutil


def copy_selected_images(source_dir, target_dir, num_images=10):
    """
    Copy randomly selected images from source directory to target directory.

    Args:
        source_dir: Directory containing the extracted ImageNet validation images
        target_dir: Directory to copy the selected images to
        num_images: Number of images to randomly select
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Randomly select images from the source directory
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.JPEG', '.jpeg', '.jpg'))]

    # Check if we have enough images
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images found in source directory.")
        num_images = len(image_files)

    # Randomly select images
    selected_files = random.sample(image_files, num_images)

    print(f"Randomly selected {len(selected_files)} images. Copying to {target_dir}...")

    # Copy selected images
    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(source_path, target_path)
    print(f"Copied {len(selected_files)} images to {target_dir}")

    # Save the list of selected files
    selected_paths_file = os.path.join(target_dir, "selected_images_paths.txt")
    with open(selected_paths_file, 'w') as f:
        for filename in selected_files:
            full_path = os.path.abspath(os.path.join(source_dir, filename))
            f.write(f"{full_path}\n")

    print(f"Saved paths of selected images to {selected_paths_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source-path',
                        default='./ILSVRC2012_img_val')

    parser.add_argument('--target-path',
                        default = './selected_imagenet_images')

    parser.add_argument('--num-samples',
                        type=int,
                        default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_selected_images(args.source, args.target, args.num)