# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
python _train_adv_img_query_fixed_budget.py \
    --output 'results' \
    --data_path '../ii_transfer_images' \
    --text_path '../ii_transfer_images/captions.txt' \
    --batch_size 1 \
    --num_samples 100 \
    --steps 8 \
    --epsilon 8 \
    --sigma 8 \
    --delta 'zero' \
    --num_query 100 \
    --num_sub_query 10 \\