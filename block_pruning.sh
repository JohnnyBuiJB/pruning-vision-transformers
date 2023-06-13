#!/bin/bash
#SBATCH -J BlockPruning						  # name of job
#SBATCH -A ai535	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p class								  # name of partition or queue
#SBATCH -o BlockPruning.out				  # name of output file for this submission script
#SBATCH -e BlockPruning.err				  # name of error file for this submission script
#SBATCH -t 18:30:00                         # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
#SBATCH --gres=gpu:2                         # number of GPUs to request (default 0)
#SBATCH --mem=50G                          # request 10 gigabytes memory (per node, default depends on node)

# load any software environment module required for app (e.g. matlab, gcc, cuda)
source /nfs/hpc/share/chenchiu/venv/bin/activate
cd /nfs/hpc/share/chenchiu/pruning-vision-transformers/EdgeVisionTransformer/deit_pruning
# module load software/version

# run my job (e.g. matlab, python)
# Prune
# torchrun src/train_main.py --deit_model_name facebook/deit-base-patch16-224 --output_dir ../../models --data_path /nfs/hpc/share/buivy/pruning-vision-transformers/ --sparse_preset topk-hybrid-struct-layerwise-tiny --layerwise_thresholds h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3 --nn_pruning --do_eval --micro_batch_size 32 --scale_lr --epoch 6

# Finetune
torchrun src/train_main.py --deit_model_name ../../models/final/ --output_dir ../../block_pruning_results --data_path /nfs/hpc/share/buivy/pruning-vision-transformers/ --final_finetune --micro_batch_size 32 --scale_lr --epoch 5