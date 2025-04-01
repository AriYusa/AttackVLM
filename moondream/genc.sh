python generate_captions.py seed=31 resolution=224 batch_size=10 images_path="../selected_imagenet_images" num_samples=500 output_path="../selected_imagenet_images"
python generate_captions.py seed=31 resolution=224 batch_size=10 images_path="../gen_images" num_samples=500 output_path="../gen_images"

tar -czvf selected_imagenet_images.tar.gz selected_imagenet_images/
tar -czvf gen_images.tar.gz gen_images/
