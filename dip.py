from __future__ import print_function
import os
import argparse
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr

from skip import skip
import utils


def dip(noisy_img, 
        mask=None,
        lr=0.0001,
        niter=50000,
        traj_iter=1000,
        num_ch=3,
        fixed_start=False,
        reg_noise_std=0.0,
        n_ch_down=128,
        n_ch_up=128,
        skip_conn=4,
        depth=5,
        act_fun='LeakyReLU',
        upsample='bilinear'):
    
    input_depth = num_ch
    output_depth = num_ch

    net = skip(input_depth, output_depth,
            num_channels_down = [n_ch_down]*depth,
            num_channels_up   = [n_ch_up]*depth,
            num_channels_skip = [skip_conn]*depth, 
            upsample_mode=upsample,
            act_fun=act_fun,
            )

    net.cuda()
    net = nn.DataParallel(net)
    weight_decay = 0
    eta = torch.randn(*noisy_img.size())
    reg_noise = Variable(torch.zeros_like(eta)).cuda()
    eta = Variable(eta).cuda()
    fixed_target = noisy_img
    fixed_target = fixed_target.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss().cuda()
    T = []

    for itr in range(niter):
        optim.zero_grad()
        # add reg noise for inpainting
        if reg_noise_std > 0:
            rec = net(eta + reg_noise.normal_() * reg_noise_std)
        else:
            rec = net(eta)
        rec = net(eta)
        loss = mse(rec,fixed_target)
        loss.backward()
        optim.step()
        if (itr%traj_iter == 0):
            out_np = rec[0, :, :, :].transpose(0,2).detach().cpu().data.numpy()
            T.append(out_np)
            print('Iteration '+str(itr)+': '+str(loss.data))
    return T


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    parser.add_argument(
        '--clean_img', required=True, help='Path to clean image file'
    )
    parser.add_argument(
        '--output_dir', default='outputs', help='Folder to save outputs'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate'
    )
    parser.add_argument(
        '--niter', type=int, default=1000, help='Num iters'
    )
    parser.add_argument(
        '--traj_iter', type=float, default=100, help='Traj. logging iter'
    )
    parser.add_argument(
        '--reg_noise_std', type=float, default=0, help='Var. of noise added to the input as a regularizer'
    )
    parser.add_argument(
        '--n_ch_down', type=int, default=256, help='Num channels for downsampling'
    )
    parser.add_argument(
        '--n_ch_up', type=int, default=256, help='Num channels for upsampling'
    )
    parser.add_argument(
        '--skip_conn', type=int, default=4, help='Skip connection indices'
    )
    parser.add_argument(
        '--depth', type=int, default=5, help='Enc-Dec depth'
    )
    parser.add_argument(
        '--act_fun', default='LeakyReLU', help='Activation function'
    )
    parser.add_argument(
        '--upsample', default='bilinear', help='Upsampling method'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    #load img and mask
    noisy_img = utils.imread(args.noisy_img)
    clean_img = utils.imread(args.clean_img)
    output_dir = args.output_dir
    
    if len(noisy_img.shape) == 2:
        num_ch = 1
    else:
        num_ch = noisy_img.shape[-1]

    noisy_img = Variable(utils.preproc(noisy_img))

    # Generate 2 DIP trajectories
    traj_set = []
    for _ in range(2):
        T = dip(noisy_img, 
                lr=args.lr,
                niter=args.niter,
                traj_iter=args.traj_iter,
                num_ch=num_ch,
                reg_noise_std=args.reg_noise_std,
                n_ch_down=args.n_ch_down,
                n_ch_up=args.n_ch_up,
                skip_conn=args.skip_conn,
                depth=args.depth,
                act_fun=args.act_fun,
                upsample=args.upsample)
        traj_set.append(T)
    T1,T2 = traj_set

    # Find the best PSNR in the trajectory
    clean_img = utils.imread(args.clean_img)
    t1_psnr = [compare_psnr(clean_img,t1) for t1 in T1]
    best_psnr = np.max(t1_psnr)
    best_itr = np.argmax(t1_psnr)
    best_psnr_pred = T1[best_itr]
    print('Best PSNR: '+str(best_psnr))

    # Show initial, best and final points on the trajectory
    plt.imshow(T1[0])
    plt.title('Initial output')
    plt.savefig(os.path.join(output_dir,'init.png'))
    plt.close()
    plt.imshow(T1[-1])
    plt.title('Final output')
    plt.savefig(os.path.join(output_dir,'final.png'))
    plt.close()
    plt.imshow(best_psnr_pred)
    plt.title('Best PSNR: '+str(best_psnr)+', Best Iter: '+str(best_itr))
    plt.savefig(os.path.join(output_dir,'best_psnr.png'))
    plt.close()
    
    # Find trajectory intersection
    sse = []
    for t1,t2 in zip(T1,T2):
        sse.append(((t1-t2)**2).sum())
    itrs = [i*args.traj_iter for i in range(len(sse))]
    plt.plot(itrs,sse)
    plt.ylabel('SSE')
    plt.xlabel('Iterations')
    plt.savefig(os.path.join(output_dir,'sse_vs_itr.png'))
    

