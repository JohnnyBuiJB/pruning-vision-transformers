#!/bin/bash
#SBATCH -J DynVit						  # name of job
#SBATCH -A cs479-579	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p class								  # name of partition or queue
#SBATCH -o DynVit.out				  # name of output file for this submission script
#SBATCH -e DynVit.err				  # name of error file for this submission script
#SBATCH -t 18:30:00                         # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
#SBATCH --gres=gpu:2                         # number of GPUs to request (default 0)
#SBATCH --mem=50G                          # request 10 gigabytes memory (per node, default depends on node)

# load any software environment module required for app (e.g. matlab, gcc, cuda)
source /nfs/hpc/share/buivy/trustworthy-machine-learning/venv/bin/activate
cd /nfs/hpc/share/buivy/pruning-vision-transformers/DynamicViT
# module load software/version

# run my job (e.g. matlab, python)
torchrun main.py --output_dir logs/dynamicvit_deit-b --model deit-b --input_size 224 --batch_size 128 --data_path ../ --epochs 10 --base_rate 0.7 --lr 1e-3 --warmup_epochs 5 --drop_path 0.2 --ratio_weight 5.0

