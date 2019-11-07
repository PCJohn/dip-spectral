# The Spectral Bias of the Deep Image Prior

Code to reproduce results in the paper.


**Dataset**

Download images from Table 2 here: http://www.cs.tut.fi/~foi/GCF-BM3D/index.html. Save these images in a folder called data/

Generate noisy images:
    
    python make_dataset.py --downsample_factors 1,2,4
    
This will add different levels of noise to the images downsampled by the specified factors and save them in the data/ folder


**Trajectory Experiment**

Run the following to reproduce the trajectory experiment (figure 1 in the paper)

    # Less high frequency components
    python dip.py --noisy_img data/triangle/triangle-0.2.png --niter 1000 --traj_iter 10

    # More high frequency components
    python dip.py --noisy_img data/triangle/triangle-0.9.png --niter 1000 --traj_iter 10


**Denoising Experiment**

Run the following to reproduce the denoising results with different architectures (tables 1 and 2 in the paper):

1. DIP model (a convolutional encoder-decoder):

        python dip.py --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png
    
2. ReLUNet:

        python relunet.py --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png
    
3. DIP-128:

        python dip.py --linear --n_ch_up 128 --n_ch_down 128 --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png

4. DIP-2048:
        
        python dip.py --linear --n_ch_up 2048 --n_ch_down 2048 --noisy_img data/denoise-4/House256/House256_s25.png --clean_img data/denoise-4/House256/House256.png
    

This runs DIP twice to generate 2 trajectories. The denoising results, along with convergence of the 2 trajectories will be saved in the outputs/ folder.
    
    
