#!/bin/sh
env="StarCraft2"
map="3m"
algo="mat"
exp="single"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_smac_offline.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 16 --batch_size 3200 --num_mini_batch 1 --log_interval 1 --eval_interval 1 --use_eval
