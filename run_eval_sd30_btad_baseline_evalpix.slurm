#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

### FIXME: update job-name
#SBATCH --job-name="e_btad"
#SBATCH --account=punim1623
#SBATCH --time=0-02:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

SEED=30

# FIXME: update save_path, checkpoint_path

python test.py \
  --dataset btad \
  --data_path data/btad \
  --save_path ./results/btad/0shot_metanetpag_lr1e4_nctx6_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_metanetpag_lr1e4_nctx6_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --meta_net \
  --metanet_patch_and_global \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##