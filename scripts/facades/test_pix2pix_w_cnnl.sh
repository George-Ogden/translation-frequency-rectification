set -ex
python test.py --dataroot ./datasets/facades --name facades_pix2pix_w_cnnl --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
