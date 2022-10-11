#!/bin/sh

# Necessary because /home/tbbakke/anaconda3/anaconda3/lib/ is missing libstdc++.so.6 and /lib64/ has old CXXABI.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tbbakke/anaconda3/envs/c2st/lib

# Original code folder is here
MAIN_DIR=/home/tbbakke/c2st-e
# Launch dir
LAUNCH_DIR=/home/tbbakke/c2st-e/launch/
mkdir -p "${LAUNCH_DIR}"


#META_DATA_SIZES=(
#    200
#)
## --------------------------------------------
## -------------- SOME IMBALANCE --------------
## --------------------------------------------
#for meta_size in "${META_DATA_SIZES[@]}"; do
#    # Create dir for specific experiment run
#    dt=$(date '+%F_%T.%3N')
#    LOGS_DIR=${LAUNCH_DIR}/${dt}
#    mkdir -p "${LOGS_DIR}"
#    # Copy code to experiment folder
#    rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
#    JOB_NAME=meta_${meta_size}
#    SLURM=${LOGS_DIR}/run.slrm
#    # Make SLURM file
#    echo "${SLURM}"
#    echo "#!/bin/bash" > ${SLURM}
#    echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
#    echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
#    echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
#    echo "#SBATCH --gres=gpu:1" >> ${SLURM}
#    echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
#    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
#    echo "#SBATCH --mem=8G" >> ${SLURM}
#    echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
#    echo "#SBATCH --nodes=1" >> ${SLURM}
#    echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
#    for i in {1..100}; do
#        {
#            echo CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/c2st/bin/python ${LOGS_DIR}/c2st-e/trainMRI_all.py \
#                --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
#                --dataset_sizes ${meta_size} --settings 1a 1b 2 --seed None \
#                --data_dir /home/tbbakke/data/fastMRI/singlecoil/singlecoil_all \
#                --pathology_path /home/tbbakke/fastmri-plus/Annotations/knee.csv \
#                --checked_path /home/tbbakke/fastmri-plus/Annotations/knee_file_list.csv \
#                --save_dir /home/tbbakke/c2st-e/results/mri/meta_analysis_oct10_ivi
#            echo
#        } >> ${SLURM}
#    done
#    sbatch ${SLURM}
#    sleep 1
#done


# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_400
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
        --dataset_sizes 400 --settings 1a 1b 2 --seed None \
        --data_dir /home/tbbakke/data/fastMRI/singlecoil/singlecoil_all \
        --pathology_path /home/tbbakke/fastmri-plus/Annotations/knee.csv \
        --checked_path /home/tbbakke/fastmri-plus/Annotations/knee_file_list.csv \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct11_ivi
    echo
} >> ${SLURM}

sbatch ${SLURM}
sleep 1


# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=base_800
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
        --dataset_sizes 800 --settings 1a 1b 2 --seed None \
        --data_dir /home/tbbakke/data/fastMRI/singlecoil/singlecoil_all \
        --pathology_path /home/tbbakke/fastmri-plus/Annotations/knee.csv \
        --checked_path /home/tbbakke/fastmri-plus/Annotations/knee_file_list.csv \
        --save_dir /home/tbbakke/c2st-e/results/mri/base_exp_oct11_ivi
    echo
} >> ${SLURM}

sbatch ${SLURM}
sleep 1