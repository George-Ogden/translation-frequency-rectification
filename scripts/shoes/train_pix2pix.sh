set -ex
python train.py --dataroot ./datasets/edges2shoes --name edges2shoes_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --n_epochs 4 --n_epochs_decay 1
