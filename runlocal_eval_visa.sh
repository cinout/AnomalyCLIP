SEED=10

python test.py \
  --dataset visa \
  --data_path data/visa \
  --save_path ./results/visa/0shot_bs8_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_bs8_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --meta_net \
  --meta_split \
  --seed $SEED \