import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
from sewar.full_ref import mse, rmse, psnr, ssim, sam
import np_metric as img_met
import metrics_glf_cr as metrics_glf_cr
from torchvision.transforms import Resize
import logging
import datetime
import sys
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from basicsr.models.archs.linear_attntion_relu import Restormer
# from basicsr.models.archs.linear_attntion_softmax_conv import Restormer
# from basicsr.models.archs.linear_attn_add_pos_v3 import Restormer
# from basicsr.models.archs.swin_transformer_sparse import Restormer
# from basicsr.models.archs.swin_transformer import Restormer
from basicsr.models.archs.Rwin_DGC_FFT_RestomerFFN_tiny_new import Restormer
#from basicsr.models.archs.swin_transformer_sparse_2 import Restormers
# from basicsr.models.archs.atttn_GDC import Restormer
# from basicsr.models.archs.linear_attntion_softmax_new import Restormer
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import matplotlib.pyplot as plt
import numpy as np

import os

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Get model weights and parameters
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [2, 2, 8,  2, 2],
              'ffn_expansion_factor': 2.66, 'bias': False, 'split_size': [16, 32]}

#load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'ShuntRestormer_arch.py'))
#model = load_arch['ShuntRestormer'](**parameters)
# load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'vision_mamba_multi_scal.py'))
model = Restormer(**parameters)  #load_arch['MambaRestormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


#test param and flops
# Model Size####
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.3fM" % (total / 1e6))
# print(128, 128, model.flops([128, 128]) / 1e9, 'G')
# print(256, 256, model.flops([256, 256]) / 1e9, 'G')

# Test
_input = torch.randn([1, 3, 256, 256]).cuda()
model = model.cuda() #.cpu()
from fvcore.nn import FlopCountAnalysis, flop_count_table

flops = FlopCountAnalysis(model, inputs=(_input))
print(flop_count_table(flops))

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iterations = 100   # 重复计算的轮次
# GPU预热
g= model
a=_input
g.eval()
with torch.no_grad():
    for i in range(50):
        _ = g(a)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = g(a)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

