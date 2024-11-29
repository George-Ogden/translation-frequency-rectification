import argparse
from glob import glob
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm

from metric_utils import psnr, ssim, lpips, fid, lfd, mse, print_and_write_log


parser = argparse.ArgumentParser()
# basic
parser.add_argument('--metrics', type=str, default=[], nargs='+', help='space separated metrics list: psnr | ssim | lpips | fid | lfd | mse')
parser.add_argument('--name', type=str, default=None, help='experiment name')
parser.add_argument('--direction', type=str, default='BtoA', help='experiment name', choices={"BtoA", "AtoB"})

opt = parser.parse_args()

for i in range(len(opt.metrics)):
    opt.metrics[i] = opt.metrics[i].lower()
    assert opt.metrics[i] in ['psnr', 'ssim', 'lpips', 'fid', 'lfd', 'mse'], (
        f'Current supported metrics are: psnr | ssim | lpips | fid | lfd | mse, but got {opt.metrics[i]}')

test_folder = os.path.join("results", opt.name, "test_latest")
image_folder = os.path.join(test_folder, "images")
logs_file = os.path.join(test_folder, "metrics.txt")
target = opt.direction[0]

fake_path = os.path.join(image_folder, f"*_fake_{target}*")
real_path = os.path.join(image_folder, f"*_real_{target}*")

with open(logs_file, "x"): ...

results = dict()
for metric in opt.metrics:
    results[metric] = list()
metrics = opt.metrics.copy()
if 'fid' in metrics:
    results['fid'].append(fid([fake_path, real_path]))
    metrics.remove('fid')
if len(metrics) > 0:
    fake_paths = sorted(glob(fake_path))
    real_paths = sorted(glob(real_path))
    assert len(fake_paths) == len(real_paths)
    for fake_path, real_path in tqdm(list(zip(fake_paths, real_paths))):
        assert os.path.basename(fake_path).replace("fake", "real") == os.path.basename(real_path)
        assert os.path.basename(fake_path) == os.path.basename(real_path).replace("real", "fake")
        img_fake = cv2.imread(fake_path).astype(np.float64)
        img_real = cv2.imread(real_path).astype(np.float64)
        for metric in metrics:
            result = eval(metric + '(img_fake, img_real)')
            results[metric].append(result)

for metric in opt.metrics:
    result_mean = np.mean(results[metric])
    result = f"{metric.upper()}: {result_mean:.10f}\n"
    print(result, end="")
    with open(logs_file, "a") as f:
        f.write(result)
