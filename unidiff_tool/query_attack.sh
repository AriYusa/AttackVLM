python query_attack.py \
    adv_data_path='../img_img_transfer_images' \
    adv_text_path="../img_img_transfer_images/unidiff_captions.txt" \
    clean_data_path='../selected_imagenet_images' \
    tgt_text_path='../coco_captions.txt' \
    output_path='../unififf_query_images' \
    batch_size=2 \
    num_samples=500 \
    rgf_steps=4 \
    epsilon=8 \
    sigma=8 \
    num_query=100 \
    num_sub_query=10 \
    wandb_project_name='unidiff-attack'