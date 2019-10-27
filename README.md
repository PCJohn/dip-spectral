# dip-spectral

**Usage**

Download images from Table 2 here: http://www.cs.tut.fi/~foi/GCF-BM3D/index.html Save these images in the folder data/

Generate noisy images:
    
    python make_dataset.py --downsample_factors 1,2,4
    
This will add different levels of noise to the images downsampled by the specified factors and save them in the data/ folder

Run the following to denoise images with a DIP model (a convolutional encoder-decoder):

    python dip.py --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png --niter 500 --traj_iter 100
    
Run the following to denoise images with a ReLUNet:

    python relunet.py --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png
    
The denoising results, along with convergence of 2 trajectories will be saved in the outputs/ folder
