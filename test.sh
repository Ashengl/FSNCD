#!/bin/bash

set -e
set -x
export CUDA_VISIBLE_DEVICES=2

python test.py \
            --dataset_name 'aircraft' \
            --grad_from_block 11 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.005 \
            --eval_funcs 'v2'\
            --n_way 5 \
            --n_shot 5 \
            --n_nc 5 \
            --n_query 15 \
            --n_episode 50 \
            --epochs 50 \
            --alpha 1.4 \
            --method 'rank'\
            --task 'realtime'\
