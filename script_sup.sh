common_corruptions=("original" "gaussian_noise,5" "shot_noise,5" "impulse_noise,5" "defocus_blur,5" "glass_blur,5" \
            "motion_blur,5" "zoom_blur,5" "snow,5" "frost,5" "fog,5" \
            "brightness,5" "contrast,5" "elastic_transform,5" "pixelate,5" "jpeg_compression,5" "upsample,5")

python main_supervised.py \
    ../data/myCIFAR-10-C/ \
    --dist-url 'tcp://localhost:10012' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a resnet_ttt \
    --width 4 \
    -b 256 \
    --lr 0.01 \
    --corruption original \
    # --resume results/model/supervised/supervised_w4_resnet_ttt_lr_0.01_bsz_256_aug_original/checkpoint_0240.pth.tar