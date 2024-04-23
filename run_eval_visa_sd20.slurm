#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="ev_vis"
#SBATCH --account=punim1623
#SBATCH --time=0-03:00:00

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

# FIXME: update save_path, checkpoint_path

python test.py \
  --dataset visa \
  --data_path data/visa \
  --save_path ./results/visa/0shot_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --meta_net \
  --seed $SEED \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##