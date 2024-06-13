SEED=10

python test.py \
  --dataset visa \
  --data_path data/visa \
  --save_path ./results/visa/0shot_baseline_nodice_$SEED \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --seed $SEED \
  --meta_net \
  --metanet_patch_only \
  --musc \
  
  --checkpoint_path ./checkpoints/pretrained_mvtec_metanetponly_lr1e4_$SEED/epoch_15.pth \
  # --checkpoint_path ./checkpoints/pretrained_mvtec_baseline_$SEED/epoch_15.pth \
  # --measure_image_by_pixel \
  # --checkpoint_path ./checkpoints/pretrained_mvtec_visualaetanh_$SEED/epoch_15.pth \
  # --visual_ae \