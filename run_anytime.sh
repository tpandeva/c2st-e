#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate c2st


python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 7236 --settings 2 --seed None --test_type anytime --num_skip_rounds -1 --cold_start True \
     --volumes_per_batch 10 --do_online_learning False --patience 3 \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/anytime_offline_exp_jan24

python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 7236 --settings 2 --seed None --test_type anytime --num_skip_rounds -1 --cold_start True \
     --volumes_per_batch 2 --do_online_learning False --patience 3 \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/anytime_offline_exp_jan24