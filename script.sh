python main_moco_ttt.py \
  -a resnet_ttt \
  --print-freq 20 \
  --save-freq 40 \
  --svm-freq 20 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --model_path ./results/model \
  --tb_path ./results/tb \
  --mlp --aug-plus --cos \
  --lr 0.03 \
  --batch-size 256 \
  --width 1 \
  --epochs 1000 \
  --moco-k 4096 \
  --moco-t 0.2 \
  ../data/myCIFAR-10-C/ \
  --shared layer2 \
  --rotation_type rand \
  --aug original
  # --group_norm 32
  # --val gaussian_noise,5 \
  # --ttt \
  # --resume ./results/model/moco_ttt_gn8_w4_resnet_ttt_lr_0.03_bsz_256_k_4096_t_0.2_fog,5/checkpoint_0240.pth.tar