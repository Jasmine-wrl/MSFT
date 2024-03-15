#!/bin/bash
#SBATCH -J test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w node04
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2

module load shared
module add anaconda3
module add cudnn8.1-cuda11.2/8.1.1.33

source activate 
source activate wrl

gpus=0,1
# gpus=0
checkpoint_root=checkpoints 

# data_name=LEVIR  # dataset name
data_name=DSIFN  # dataset name 


img_size=256
batch_size=16
num_workers=4


# lr=0.1
lr=0.01
# lr=0.001
# lr=0.0001

# optimizer=adam
optimizer=sgd
# optimizer=adamw

max_epochs=150  #training epochs
# max_epochs=100  #training epochs

net_G=base_transformer_pos_s4fpn_diff_dd8_e2d6 # model name

lr_policy=linear
# lr_policy=exp

split=train  # training txt
split_val=val  #validation txt
# project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_nw${num_workers}
project_name = MSFT

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name} --lr ${lr} --num_workers ${num_workers} --optimizer ${optimizer}