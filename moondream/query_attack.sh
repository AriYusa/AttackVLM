python query_attack.py \
    seed=31 \
    adv_data_path='../img_img_transfer_images' \
    adv_text_path="../img_img_transfer_images/moondream_captions.txt" \
    clean_data_path='../selected_imagenet_images' \
    tgt_text_path='../coco_captions.txt' \
    output_path='../moondream_query_images' \
    resolution=224 \
    batch_size=1 \
    num_samples=500\
    rgf_steps=8 \
    epsilon=8 \
    sigma=8 \
    alpha=1 \
    num_query=100 \
    num_sub_query=10 \
    wandb_project_name='moondream-attack'