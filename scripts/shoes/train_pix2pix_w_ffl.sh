set -ex
python train.py --dataroot ./datasets/edges2shoes --name edges2shoes_pix2pix_w_ffl --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --ffl_w 100.0 --n_epochs 4 --n_epochs_decay 1
