python gen_and_save_captions.py \
      seed=31 \
      images_path='../img_text_transfer_images' \
      output_path='../img_text_transfer_images2' \
      batch_size=10 \
      num_samples=100

python gen_and_save_captions.py \
      seed=31 \
      images_path='../img_img_transfer_images' \
      output_path='../img_img_transfer_images2' \
      batch_size=10 \
      num_samples=100

python gen_and_save_captions.py \
      seed=31 \
      images_path='../selected_imagenet_images' \
      output_path='../selected_imagenet_images2' \
      batch_size=10 \
      num_samples=100

python gen_and_save_captions.py \
      seed=31 \
      images_path='../gen_images' \
      output_path='../gen_images2' \
      batch_size=10 \
      num_samples=100

python ../common/baseline_victim_scores.py \
      seed=31 \
      num_samples=100