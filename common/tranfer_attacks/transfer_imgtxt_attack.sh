python transfer_imgtxt_attack.py \
    output_path="../img_text_transfer_images" \
    cle_data_path="../selected_imagenet_images" \
    tgt_text_path="../coco_captions.txt" \
    wandb_project_name="transfer_attack" \
    batch_size=10 \
    num_samples=500 \
    steps=100
