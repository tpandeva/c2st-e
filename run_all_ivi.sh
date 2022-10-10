#!/bin/sh

# Original code folder is here
MAIN_DIR=/home/tbbakke/c2st-e
# Launch dir
LAUNCH_DIR=/home/tbbakke/c2st-e/launch/
mkdir -p "${LAUNCH_DIR}"


META_DATA_SIZES=(
    400
    800
    1000
    2000
)
# --------------------------------------------
# -------------- SOME IMBALANCE --------------
# --------------------------------------------
for meta_size in "${META_DATA_SIZES[@]}"; do
    # Create dir for specific experiment run
    dt=$(date '+%F_%T.%3N')
    LOGS_DIR=${LAUNCH_DIR}/${dt}
    mkdir -p "${LOGS_DIR}"
    # Copy code to experiment folder
    rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
    JOB_NAME=meta_${meta_size}
    SLURM=${LOGS_DIR}/run.slrm
    # Make SLURM file
    echo "${SLURM}"
    echo "#!/bin/bash" > ${SLURM}
    echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
    echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
    echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
    echo "#SBATCH --gres=gpu:1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "#SBATCH --mem=8G" >> ${SLURM}
    echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
    for i in {1..100}; do
        {
            echo CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
                --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
                --dataset_sizes meta_size --settings 1a 1b 2 --seed None \
                --save_dir /home/tbbakke/c2st-e/results/mri/meta_analysis_oct10
            echo
        } >> ${SLURM}
    done
    sbatch ${SLURM}
    sleep 1
done


# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_400_800
SLURM=${LOGS_DIR}/run.slrm
# Make SLURM file
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}

{
    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 400 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10

    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 800 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10
} >> ${SLURM}

sbatch ${SLURM}
sleep 1



# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_1000_2000
SLURM=${LOGS_DIR}/run.slrm
# Make SLURM file
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
{
    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 1000 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10

    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 2000 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10
} >> ${SLURM}

sbatch ${SLURM}
sleep 1



# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_3000_4000
SLURM=${LOGS_DIR}/run.slrm
# Make SLURM file
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
{
    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 3000 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10

    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 4000 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10
} >> ${SLURM}

sbatch ${SLURM}
sleep 1



# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_5000
SLURM=${LOGS_DIR}/run.slrm
# Make SLURM file
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
{
    CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 5000 --settings 1a 1b 2 --seed None \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct10
} >> ${SLURM}

sbatch ${SLURM}
sleep 1