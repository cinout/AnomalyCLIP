SEED=10

python test.py \
  --dataset sdd \
  --data_path data/sdd \
  --save_path ./results/sdd/0shot_baseline_nodice_$SEED \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --visual_ae \
  --checkpoint_path ./checkpoints/pretrained_mvtec_visualaetanh_$SEED/epoch_15.pth \
  # --checkpoint_path ./checkpoints/pretrained_mvtec_baseline_nodice_$SEED/epoch_15.pth \