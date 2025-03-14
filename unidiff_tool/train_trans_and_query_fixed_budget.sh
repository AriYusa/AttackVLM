python train_trans_and_query_fixed_budget.py \
    output_path='query_resutls' \
    adv_data_path='ii_transfer_results' \
    clean_data_path='clean_images' \
    adv_text_path=['ii_transfer_results','captions.txt'] \
    tgt_text_path='coco_captions_10000.txt' \
    batch_size=2 \
    num_samples=6 \
    rgf_steps=3 \
    epsilon=8 \
    sigma=8 \
    num_query=2 \
    num_sub_query=2 \
    wandb_project_name='unidiff-attack'
