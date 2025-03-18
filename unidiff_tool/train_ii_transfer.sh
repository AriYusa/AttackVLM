python train_ii_transfer.py \
    output_path="../ii_transfer_images" \
    cle_data_path="../selected_imagenet_images" \
    tgt_data_path="../gen_images" \
    wandb_project_name="transfer_attack" \
    batch_size=10 \
    num_samples=1000 \
    steps=100
