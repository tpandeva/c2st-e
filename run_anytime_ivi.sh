#!/bin/sh

# Necessary because /home/tbbakke/anaconda3/anaconda3/lib/ is missing libstdc++.so.6 and /lib64/ has old CXXABI.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tbbakke/anaconda3/envs/c2st/lib

# Original code folder is here
MAIN_DIR=/home/tbbakke/c2st-e
# Launch dir
LAUNCH_DIR=/home/tbbakke/c2st-e/launch/
mkdir -p "${LAUNCH_DIR}"



# TYPE 1A

# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=anytime_1a
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
    echo CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/c2st-e/trainMRI_all.py \
        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
        --dataset_sizes 7236 --settings 1a --seed None --num_workers 12 \
        --test_type anytime --num_skip_rounds 0 --cold_start True --volumes_per_batch 10 \
        --data_dir /home/tbbakke/data/fastMRI/singlecoil/singlecoil_all \
        --pathology_path /home/tbbakke/fastmri-plus/Annotations/knee.csv \
        --checked_path /home/tbbakke/fastmri-plus/Annotations/knee_file_list.csv \
        --save_dir /home/tbbakke/c2st-e/results/mri/anytime_exp_jan17

    echo
} >> ${SLURM}

sbatch ${SLURM}
sleep 1



# TYPE 1B

## Create dir for specific experiment run
#dt=$(date '+%F_%T.%3N')
#LOGS_DIR=${LAUNCH_DIR}/${dt}
#mkdir -p "${LOGS_DIR}"
## Copy code to experiment folder
#rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
#JOB_NAME=anytime_1b
#SLURM=${LOGS_DIR}/run.slrm
## Make SLURM file
#echo "${SLURM}"
#echo "#!/bin/bash" > ${SLURM}
#echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
#echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
#echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
#echo "#SBATCH --gres=gpu:1" >> ${SLURM}
#echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
#echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
#echo "#SBATCH --mem=8G" >> ${SLURM}
#echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
#echo "#SBATCH --nodes=1" >> ${SLURM}
#echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
#{
#    echo CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/c2st-e/trainMRI_all.py \
#        --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
#        --dataset_sizes 7236 --settings 1b --seed None --num_workers 12 \
#        --test_type anytime --num_skip_rounds 0 --cold_start True --volumes_per_batch 10 \
#        --data_dir /home/tbbakke/data/fastMRI/singlecoil/singlecoil_all \
#        --pathology_path /home/tbbakke/fastmri-plus/Annotations/knee.csv \
#        --checked_path /home/tbbakke/fastmri-plus/Annotations/knee_file_list.csv \
#        --save_dir /home/tbbakke/c2st-e/results/mri/anytime_exp_jan17
#
#    echo
#} >> ${SLURM}
#
#sbatch ${SLURM}
#sleep 1