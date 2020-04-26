# CUDA_VISIBLE_DEVICES=6,7,8,9 \
python main_moco_ttt.py \
  -a resnet_ttt \
  --print-freq 30 \
  --save-freq 80 \
  --svm-freq 80 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --model_path ./results/model/ \
  --tb_path ./results/tb/ \
  --mlp --aug-plus --cos \
  --lr 0.1 \
  --batch-size 256 \
  --width 4 \
  --epochs 1000 \
  --moco-k 4096 \
  --moco-t 0.3 \
  --moco-m 0.99 \
  ../data/myCIFAR-10-C/ \
  --shared layer2 \
  --rotation_type rand \
  --aug original \
  --norm sn \
  --oracle brightness,5
  # --val original \
  # --bn_update \
  # --resume ./results/model/moco_ttt_sn_w4_resnet_ttt_lr_0.1_bsz_256_k_4096_t_0.3_m_0.99_original_oracle_gaussian_noise,5/checkpoint_1000.pth.tar
  # --val gaussian_noise,5 \
  # --ttt \
  # --frozen \
  # --resume ./results/model/moco_ttt_bn_w4_resnet_ttt_lr_0.1_bsz_256_k_4096_t_0.3_m_0.99_original/checkpoint_1000.pth.tar
