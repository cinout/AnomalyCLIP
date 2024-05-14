SEED=10

python test.py \
  --dataset sdd \
  --data_path data/sdd \
  --save_path ./results/sdd/0shot_baseline_$SEED \
  --checkpoint_path ./checkpoints/pretrained_mvtec_visualaetighter_$SEED/epoch_15.pth \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --visual_ae \