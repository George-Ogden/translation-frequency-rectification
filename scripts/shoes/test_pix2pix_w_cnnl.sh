set -ex
python test.py --dataroot ./datasets/edges2shoes --name edges2shoes_pix2pix_w_cnnl --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch --phase val
mv results/edges2shoes_pix2pix_w_cnnl/{val_latest,test_latest}