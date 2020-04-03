python main_moco.py \
  -a resnet_ttt \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --model_path ./results/model \
  --tb_path ./results/tb \
  --width 1 \
  ../data/myCIFAR-10-C/