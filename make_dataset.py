import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

# Add noise: As done in the DIP codebase: https://github.com/DmitryUlyanov/deep-image-prior/blob/de3894fb4fc01d27920e2cc66af24840e4fccea9/utils/denoising_utils.py#L6
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noisy_np


def imread(path):
    img = plt.imread(path).astype(float)
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    if img.max() > 1.0:
        img /= 255.0
    return np.array(img)

def imwrite(path,img):
    img = 255.0 * img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset for DIP denoising experiments with multiple downsampling factors.')
    parser.add_argument(
        '--downsample_factors', required=True, help='Factors to downsample images'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    downsample_factors = list(map(int,args.downsample_factors.split(',')))
    root = 'data'
    for dwnsample in downsample_factors:
        data_dir = os.path.join(root,'denoise-'+str(dwnsample))
        os.system('mkdir '+data_dir)
        for img_name in os.listdir(root):
            ext = '.png'
            if not (img_name.endswith(ext)):
                continue
            clean_img = os.path.join(root,img_name)
            img = imread(clean_img)
            size = (int(img.shape[0]/float(dwnsample)),int(img.shape[1]/float(dwnsample)))
            img = cv2.resize(img,size)
            img_dir = os.path.join(data_dir,img_name[:-len(ext)])
            if not os.path.isdir(img_dir):
                os.system('mkdir '+img_dir)
            for sigma in [0,5,10,20,25,30,35,40,50,60,70,75,80,90,100]:
                if sigma > 0:
                    output_file = os.path.join(img_dir,img_name[:-len(ext)]+'_s'+str(sigma)+ext)
                else:
                    output_file = os.path.join(img_dir,img_name)
                sigma = sigma/255.
                noisy_img = get_noisy_image(img,sigma)
                imwrite(output_file,noisy_img)
    
    # make default output folder
    os.system('mkdir outputs')


