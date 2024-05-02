SEED=10

python test.py \
  --dataset mpdd \
  --data_path data/mpdd \
  --save_path ./results/mpdd/0shot_baseline_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_baseline_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --debug_mode \