# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
python _train_adv_img_trans.py \
    --output 'images_adv_ii' \
    --cle_data_path 'clean_images' \
    --tgt_data_path 'gen_images' \
    --wandb_project_name 'transfer_attack' \
    --batch_size 2 \
    --num_samples 6 \
    --steps 300