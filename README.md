# E-C2ST for MRI data

Code for the E-C2ST MRI experiments. All experiments are run from the same script `trainMRI_all.py`. Example commands are given below.

## Commands

#### Base experiment (100 samples per size)
`num_dataset_samples` sets the number of runs to do (multiple runs are necessary for computing Type I and II errors).

NOTE: `SAVE_DIR` not necessary, but useful to find the results back later.

> python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 3000 4000 5000 --settings 1a 1b 2 --seed None \
     --save_dir SAVE_DIR


#### Meta-analysis experiment (3 partitions). 
We overload the `num_dataset_samples` loop to do partitions instead: set to 0 to activate partition behaviour.

To get multiple meta-analysis experiments to analyse for Type-I/II error (e.g. 100), run this command multiple times (e.g. using a bash loop). Set `seed` to None when doing this, because otherwise the same data split is used every run. It is recommended to set a specific `SAVE_DIR` for ease of combining multiple meta-analysis experiments. 

> python trainMRI_all.py --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
    --dataset_sizes 200 400 1000 2000 --settings 1a 1b 2 --seed None \
    --save_dir SAVE_DIR


#### Anytime testing offline:
> python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
    --dataset_sizes 7236 --settings 1a 2 --seed None \
    --test_type anytime --num_skip_rounds 0 --volumes_per_batch 10 --do_online_learning False --patience 3  \
    --save_dir SAVE_DIR 

#### Anytime testing online:
> python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
    --dataset_sizes 7236 --settings 1a 2 --seed None \
    --test_type anytime --num_skip_rounds 0 --volumes_per_batch 10 --do_online_learning True --patience 1  \
    --save_dir SAVE_DIR 