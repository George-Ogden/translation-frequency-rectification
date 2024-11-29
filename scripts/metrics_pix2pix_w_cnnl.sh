set -ex
python ./metrics.py  --metrics psnr ssim lpips fid lfd --name facades_pix2pix_w_cnnl
