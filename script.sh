python main_moco.py \
  -a resnet_ttt \
  --print-freq 30 \
  --save-freq 40 \
  --dist-url 'tcp://localhost:10006' --multiprocessing-distributed --world-size 1 --rank 0 \
  --model_path ./results/model \
  --tb_path ./results/tb \
  --mlp --aug-plus --cos \
  --lr 0.09 \
  --batch-size 256 \
  --width 4 \
  --epochs 1000 \
  --moco-k 4096 \
  --moco-t 0.3 \
  ../data/myCIFAR-10-C/
  # --resume ./results/model/moco_w8_resnet_ttt_lr_0.03_bsz_256_k_4096_t_0.2/checkpoint_0180.pth.tar