#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="t_mvtec"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

#### export WORLD_SIZE=2   ### FIXME: update world size: nodes x ntasks-per-node
#### export MASTER_PORT=28400
#### echo ">>> NODELIST="${SLURM_NODELIST}
#### master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#### export MASTER_ADDR=$master_addr
#### echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

SEED=20

# FIXME: update save_path

python train.py \
  --dataset mvtec \
  --train_data_path data/mvtec \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --batch_size 8 \
  --print_freq 1 \
  --epoch 15 \
  --save_freq 1 \
  --save_path ./checkpoints/pretrained_mvtec_metanetpag_lr1e4_nctx6_$SEED/ \
  --depth 9 \
  --n_ctx 6 \
  --t_n_ctx 4 \
  --seed $SEED \
  --meta_net \
  --metanet_patch_and_global \
  --learning_rate 0.0001 \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##