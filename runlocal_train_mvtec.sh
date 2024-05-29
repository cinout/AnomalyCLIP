SEED=20

python train.py \
  --dataset mvtec \
  --train_data_path data/mvtec \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --batch_size 8 \
  --print_freq 1 \
  --epoch 1 \
  --save_freq 1 \
  --save_path ./checkpoints/writer_pretrained_mvtec_baseline_$SEED/ \
  --depth 9 \
  --n_ctx 12\
  --t_n_ctx 4 \
  --seed $SEED \
  --meta_net \
  --metanet_patch_only \
  --musc \
  # --no_imageloss \
  # --meta_net \
  # --metanet_patch_and_global \