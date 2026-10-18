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

from basicsr.models.archs.Rwin_DGC_FFT_RestomerFFN_tiny_new import Restormer
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d(%a)%H:%M:%S',
                    filename='test_LGNet_log_' + now + '.txt',
                    filemode='w')

#将大于或等于INFO级别的日志信息输出到StreamHandler(默认为标准错误)
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s')  #屏显实时查看，无需时间
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser(description='Test on your own images')
parser.add_argument('--input_dir', default='/data/huang/datasets/cloud/clound-wy/RICE2/test/cloud/', type=str,
                    help='Directory of input images or path of single image')
parser.add_argument('--input_truth_dir', default='/data/huang/datasets/cloud/clound-wy/RICE2/test/reference/',
                    type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./output/Deraining_rice1_swin_atd_512_1000_train_20241018_512_33000iters_test', type=str,
                    help='Directory for restored results')
parser.add_argument('--weights', default='experiments/rice1_DVPNet/models/net_g_25000.pth', type=str, help='Path to weights')
parser.add_argument('--opt', default='option/rice1-TGPNet.yml', type=str, help='Path to config yml file')
args = parser.parse_args()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir)
inp_truth_dir = args.input_truth_dir

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.' + ext)))
    files = natsorted(files)

if any([inp_truth_dir.endswith(ext) for ext in extensions]):
    files_truth = [inp_truth_dir]
else:
    files_truth = []
    for ext in extensions:
        files_truth.extend(glob(os.path.join(inp_truth_dir, '*.' + ext)))
    files_truth = natsorted(files_truth)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

yaml_file= args.opt

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

model = Restormer(**x['network_g'])

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

######

checkpoint = torch.load(args.weights)
model.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model.cuda()
model.eval()


img_multiple_of = 8

MAE_vs = []
MSE_vs = []
RMSE_vs = []
BRMSE_vs = []
ssim_vs = []
psnr_vs = []
sam_vs = []
logging.critical("start testing...")



with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = load_img(file_)
            truth_file = file_.replace(args.input_dir, args.input_truth_dir)
            img_truth = load_img(truth_file)

        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        input_truth = torch.from_numpy(img_truth).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        restored = model(input_)

        restored = torch.clamp(restored, 0, 1)

        s2img1 = input_truth.clone()  #0-1 value
        fake_img1 = restored.clone()  # convert values from 0-5 to 0-1

        MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
        MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
        RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
        BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
        # ssim_v = img_met.cloud_ssim(s2img1, fake_img1)
        # psnr_v = img_met.cloud_psnr(s2img1, fake_img1) # equal to glf_cr.psnr
        psnr_v = metrics_glf_cr.PSNR(s2img1, fake_img1)
        ssim_v = metrics_glf_cr.SSIM(s2img1, fake_img1) # torch 1.8.1, 2.1.1 more than torch 2.4.1 0.06. very big gap. please test on torch 1.8.1, 2.1.1

        MAE_vs.append(np.asarray(MAE_v.cpu()))
        MSE_vs.append(np.asarray(MSE_v.cpu()))
        RMSE_vs.append(np.asarray(RMSE_v.cpu()))
        BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
        ssim_vs.append(np.asarray(ssim_v.cpu().detach().numpy()))
        # ssim_vs.append(np.asarray(ssim_v))
        psnr_vs.append(np.asarray(psnr_v))

        # spectral angle mapper
        mat = s2img1 * fake_img1
        mat = torch.sum(mat, 1)
        mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
        mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
        sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / np.pi)

        if not torch.isnan(sam_v):
            sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        input_1 = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_save = img_as_ubyte(input_1[0])
        input_tr = input_truth.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_tr_save = img_as_ubyte(input_tr[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]

        # save_img((os.path.join(out_dir, f+'.png')), restored)
        save_img((os.path.join(out_dir, f + '_out.png')), restored)
        save_img((os.path.join(out_dir, f + '_input.png')), input_save)
        save_img((os.path.join(out_dir, f + '_truth.png')), input_tr_save)

        logging.critical(
            "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}, file_name:{}".format(
                MAE_v.cpu(), MSE_v.cpu(),
                RMSE_v.cpu(), BRMSE_v.cpu(),
                psnr_v, ssim_v.cpu(), sam_v, file_))


    MAE_v = np.mean(MAE_vs)
    MSE_v = np.mean(MSE_vs)
    RMSE_v = np.mean(RMSE_vs)
    BRMSE_v = np.mean(BRMSE_vs)
    ssim_v = np.mean(ssim_vs)
    psnr_v = np.mean(psnr_vs)
    sam_v = np.mean(sam_vs)
    # print("MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f}".format(MAE_v, MSE_v, RMSE_v, BRMSE_v,
    #                                                                          psnr_v))
    logging.critical(
        "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
            MAE_v, MSE_v, RMSE_v, BRMSE_v,
            psnr_v, ssim_v, sam_v))

    print(f"\nRestored images are saved at {out_dir}")
