python main_moco.py \
  -a resnet_ttt \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  --results ./results \
  --width 8 \
  ../data/myCIFAR-10-C/