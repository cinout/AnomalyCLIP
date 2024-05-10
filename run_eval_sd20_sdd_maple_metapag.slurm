#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

### FIXME: update job-name
#SBATCH --job-name="e_sdd"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

SEED=20

# FIXME: update save_path, checkpoint_path

python test.py \
  --dataset sdd \
  --data_path data/sdd \
  --save_path ./results/sdd/0shot_maple_metapag_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_maple_metapag_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --maple \
  --meta_net \
  --metanet_patch_and_global \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##