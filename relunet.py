import os
import sys
import cv2
import time
import json
import yaml
import argparse
import itertools
import numpy as np
import matplotlib
import collections
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
from skimage.measure import compare_psnr
from scipy.fftpack import fft2

import utils


def gen_dataset(img):
    bw = (len(img.shape) == 2)
    if bw:
        w,h = img.shape
        c = 1
    else:
        w,h,c = img.shape
    xx,yy = np.meshgrid(range(w),range(h),indexing='ij')
    xx,yy = xx.flatten(),yy.flatten()
    if bw:
        y = img[xx,yy][:,np.newaxis]
    else:
        y = img[xx,yy,:]

    xx = xx / float(w-1)
    yy = yy / float(h-1)
    x = np.hstack([xx[:,np.newaxis],yy[:,np.newaxis]])
    
    return np.float32(x),np.float32(y),w,h,c


def parse_args():
    parser = argparse.ArgumentParser(description='ReLUNet for images')
    parser.add_argument(
        '--output_dir', default='outputs', help='Folder to save outputs'
    )
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    parser.add_argument(
        '--clean_img', required=True, help='Path to clean image file'
    )
    parser.add_argument(
        '--traj_iter', default=100, help='Interval to sample trajectory'
    )
    parser.add_argument(
        '--hid_size', default=256, help='Number of units per hidden layer in the ReLUNet'
    )
    parser.add_argument(
        '--depth', default=10, help='Number of hidden layers in the ReLUNet'
    )
    parser.add_argument(
        '--niter', default=30000, help='Number of iterations'
    )
    parser.add_argument(
        '--lr', default=0.001, help='Learning rate'
    )
    parser.add_argument(
        '--bz', default=2048, help='Batch size'
    )
    return parser.parse_args()


class ReLUNet(nn.Module):
    def __init__(self,H=200,d=2,bw=True):
        super(ReLUNet, self).__init__()
        out_dim = (1 if bw else 3)
        self.w = [nn.Linear(2,H,bias=True)] + \
                 [nn.Linear(H,H,bias=True) for _ in range(d-2)] + \
                 [nn.Linear(H,out_dim,bias=True)]
        
        for i,wi in enumerate(self.w):
            self.add_module('w'+str(i),wi)

    def forward(self,x):
        out = x
        d = len(self.w)
        for i in range(d-1):
            out = self._modules['w'+str(i)](out)
            out = F.relu(out,inplace=True)
        out = self._modules['w'+str(d-1)](out)
        out = F.sigmoid(out)
        return out


def train_net(x,y,w,h,c,H=200,d=2,niter=200000,lr=0.001,bz=256,traj_iter=1000):
    bw = (y.shape[-1] == 1)
    net = ReLUNet(H=H,d=d,bw=bw).cuda()

    x = Variable(torch.from_numpy(x)).cuda()
    y = Variable(torch.from_numpy(y)).cuda()

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss().cuda()
    T = []
    for itr in range(niter):
        optim.zero_grad()
        b = np.random.randint(0,x.shape[0],bz)
        y_ = net(x[b])
        loss = mse(y_,y[b])
        loss.backward()
        optim.step()
        if (itr%traj_iter == 0):
            out_np = net(x).detach().cpu().data.numpy().reshape((w,h,c))
            if bw:
                out_np = out_np[:,:,0]
            T.append(out_np)
            print('Iteration '+str(itr)+': '+str(loss.data))
            del out_np
    
    return T


if __name__ == '__main__':
    args = parse_args()
   
    output_dir = args.output_dir 
    img = utils.imread(args.noisy_img)
    x,y,w,h,c = gen_dataset(img)
    
    # Generate 2 denoising trajectories with ReLUNets
    traj_set = []
    for _ in range(2):
        T = train_net(x,y,w,h,c,
            H=args.hid_size,
            d=args.depth,
            niter=args.niter,
            lr=args.lr,
            bz=args.bz,
            traj_iter=args.traj_iter)
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
    plt.close()
    


