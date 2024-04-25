from Network_Model import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt
import datetime
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# print(torch.cuda.device_count())
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10, data_type="cave")

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
if not os.path.exists(result_path):
    os.makedirs(result_path)

def test(model):
    test_data, label_rgb = LoadTest(opt.test_path, data_type="cave")
    test_gt = test_data.cuda().float()
    label_rgb = label_rgb.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()

    with torch.no_grad():
        model_out = model(input_meas, label_rgb, input_mask)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    L, H, W, C = pred.shape
    for i in range(L):
        HSI = torch.tensor(pred[i, :, :, :])
        gt = torch.tensor(truth[i, :, :, :])
        psnr = torch_psnr(HSI, gt)
        ssim = torch_ssim(HSI, gt)
        sam = SAM_GPU(HSI, gt)
        name = result_path + f'test_{i}.mat'
        print(f'Save reconstructed HSIs as {name}.')
        print(f"psnr:{psnr},ssim:{ssim},sam:{sam}")
        scio.savemat(name, {'truth': gt, 'pred': pred,"psnr": psnr,"ssim":ssim ,"sam":"sam"})

    model.train()
    return pred, truth

def main():
    # Network_Model
    pretrained_model_path=opt.pretrained_model_path
    model = torch.load(pretrained_model_path)
    model.to("cuda:0")
    pred, truth = test(model)
    name = result_path + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()
