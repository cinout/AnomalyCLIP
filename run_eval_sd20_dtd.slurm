#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

### FIXME: update job-name
#SBATCH --job-name="e_dtd"
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

SEED=20

# FIXME: update save_path, checkpoint_path

python test.py \
  --dataset dtd \
  --data_path data/dtd \
  --save_path ./results/dtd/0shot_metanetponly_lr1e4_musc09_cluster_bcmatch_notext_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_metanetponly_lr1e4_musc09_cluster_bcmatch_notext_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --meta_net \
  --metanet_patch_only \
  --musc \
  --musc_cluster \
  --bias_ctx_match \
  --no_text_template \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##