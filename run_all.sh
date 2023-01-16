#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate c2st

## Meta-analysis experiments
#for i in {1..100}; do
#    python trainMRI_all.py --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
#     --dataset_sizes 200 400 1000 --settings 1a 1b 2 --seed None \
#     --save_dir /home/timsey/Projects/c2st-e/results/mri/meta_analysis_oct10
#done

# Base experiments
python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 1000 2000 3000 4000 5000 200 400 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/base_exp_oct10

python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 800 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/base_exp_oct12