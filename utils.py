import os
import cv2
import torch
import errno
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

def imshow(image):
    if len(image.shape) == 2:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)

def imread(path):
    img = plt.imread(path).astype(float)
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    if img.max() > 1.0:
        img /= 255.0
    return np.float32(img)

def imwrite(path,img):
    img = 255.0 * img
    if len(img.shape) == 2:
        cv2.imwrite(path,np.uint8(img))
    else:
        plt.imsave(path,np.uint8(img))

def preproc(img_n):
    if (len(img_n.shape) == 2):
        img_n = torch.FloatTensor(img_n).unsqueeze(0).unsqueeze(0).transpose(2,3)
    elif (len(img_n.shape) == 3):
        img_n = torch.FloatTensor(img_n).transpose(0,2).unsqueeze(0)
    return img_n

def shape(img):
    r,c = img.shape[:2]
    if len(img.shape) > 2:
        return (r,c,img.shape[2])
    else:
        return (r,c,1)

# Noise addition from the original DIP repo -- use to generate dataset
def get_noisy_image(img_np, sigma):
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    return img_noisy_np

# adding noise: 
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def add_noise(noise_typ,image,var=1e-4):
    row,col,ch = shape(image)
    if noise_typ == "gauss":
        mean = 0
        #var = 0.0001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(*image.shape)
        noisy = image + gauss
        # normalize back to [0,1]
        noisy = (noisy-noisy.min())/noisy.max()
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
        out[coords] = 0
        return out

def mask_img(img,mask):
    if len(img.shape) == 2:
        return np.multiply(img.copy(),mask)
    
    masked_img = img.copy()
    for ch in range(img.shape[-1]):
        masked_img[:,:,ch] = np.multiply(masked_img[:,:,ch],mask)
    return masked_img

def add_mask_noise(mask,drop_frac=0.5):
    w,h = mask.shape
    m = np.random.random((w,h))
    m[m < drop_frac] = 0
    m[m >= drop_frac] = 1
    m = np.multiply(mask,m)
    return m


def save_traj(path,traj_arr):
    # uint8 and compressed to save disk space
    traj_arr = np.array(traj_arr, dtype=np.float32)
    traj_arr = np.uint8(traj_arr * 255)
    np.savez_compressed(path, traj_arr)

def load_traj(path):
    traj_arr = np.load(path)['arr_0']
    traj_arr = np.float32(traj_arr) / 255.
    if traj_arr.shape[-1] == 1:
        traj_arr = traj_arr[:,:,:,0]
    return traj_arr

def traj_file_list(folder):
    traj_ext = '.npz'
    return [f for f in os.listdir(folder) if f.endswith(traj_ext)]


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


### FFT Utils ###
def channel_fft(ch):
    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    return mag

def fft(im):
    if len(im.shape) == 2:
        return channel_fft(im)
    else:
        return np.stack([channel_fft(im[:,:,ch]) for ch in range(im.shape[2])],axis=-1)

def band_pass_filter(im_shape,r,s,n_ch):
    w,h = im_shape[0],im_shape[1]
    f1 = np.zeros((w,h))
    f2 = np.zeros((w,h))
    cv2.circle(f1,(h//2,w//2),r+s,1,-1)
    cv2.circle(f2,(h//2,w//2),r,1,-1)
    filt = (f1-f2)
    if n_ch > 1:
        filt = np.stack([filt]*n_ch,axis=-1)
    #plt.imshow(filt,cmap='gray')
    #plt.savefig('filt.png')
    #import pdb; pdb.set_trace();
    return filt
   
def bandpass_set(im_shape,s=1):
    n_ch = 1
    if len(im_shape) == 3:
        n_ch = im_shape[-1]
    R = min(im_shape[0],im_shape[1])
    filt_bank = [band_pass_filter(im_shape,r,s,n_ch) for r in np.arange(s,R,s)]
    filt_bank = [f for f in filt_bank if f.sum() > 0]
    return filt_bank

def power_variation(img_fft,filt_bank):
    pow_var = np.array([(img_fft * filt).sum() for filt in filt_bank])
    return pow_var #/ pow_var.max()

### *** ###




