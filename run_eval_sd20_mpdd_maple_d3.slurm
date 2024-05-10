#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

### FIXME: update job-name
#SBATCH --job-name="e_mpdd"
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
  --dataset mpdd \
  --data_path data/mpdd \
  --save_path ./results/mpdd/0shot_maple_d3_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_maple_d3_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 3 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --maple \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##