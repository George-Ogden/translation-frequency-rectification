set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix_w_cnnl --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --cnn_loss_w0 0.03 --cnn_loss_w1 0.001 --cnn_loss_model resnet18
