SEED=10

python train.py \
  --dataset mvtec \
  --train_data_path data/mvtec \
  --features_list 6 12 18 24 \
  --image_size 518 \
  --batch_size 1 \
  --print_freq 1 \
  --epoch 15 \
  --save_freq 1 \
  --save_path ./checkpoints/pretrained_mvtec_$SEED/ \
  --depth 9 \
  --n_ctx 12\
  --t_n_ctx 4 \
  --meta_net \
  --seed $SEED \