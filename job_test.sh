#!/bin/bash
#SBATCH -J test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2

module load shared
module add anaconda3
module add cudnn8.1-cuda11.2/8.1.1.33

source activate 
source activate wrl

gpus=0,1

# python demo.py --output_folder results/LEVIR-CD/CD_base_transformer_pos_s4fpn_diff_dd8_e6d2_sk_LEVIR_b16_lr0.0001_adam_train_val_100_linear_nw4 \
#                --data_name LEVIR \
#                --img_size 256 \
#                --project_name CD_base_transformer_pos_s4fpn_diff_dd8_e6d2_sk_LEVIR_b16_lr0.0001_adam_train_val_100_linear_nw4 \
#                --net_G base_transformer_pos_s4fpn_diff_dd8_e6d2_sk \
#                --gpu_ids ${gpus}

python demo.py --output_folder results/DSIFN-CD/MSFT \
               --data_name DSIFN \
               --img_size 256 \
               --project_name MSFT \
               --net_G base_transformer_pos_s4fpn_diff_dd8_e2d6 \
               --gpu_ids ${gpus}

