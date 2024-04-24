import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
import logging
from ssim_torch import *
from option import *


def _as_floats(im1, im2):
    im1 = im1.cpu().detach()
    im1 = im1.numpy()
    im2 = im2.cpu().detach()
    im2 = im2.numpy()
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

##########################################################################################################
# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

# def batch_SAM_GPU(img, ref):
#     N = img.size()[0]
#     C = img.size()[1]
#     H = img.size()[2]
#     W = img.size()[3]
#     Itrue = img.clone().resize_(N, C, H*W)
#     Ifake = ref.clone().resize_(N, C, H*W)
#     nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H*W)
#     denom1 = torch.pow(Itrue,2).sum(dim=1).sqrt_().resize_(N, H*W)
#     denom2 = torch.pow(Ifake,2).sum(dim=1).sqrt_().resize_(N, H*W)
#     sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H*W)
#     sam = sam / np.pi * 180a
#     sam = torch.sum(sam) / (N*H*W)
#     return sam
#
def SAM_GPU(img, ref):
    C = img.size()[0]
    H = img.size()[1]
    W = img.size()[2]
    esp = 1e-12
    Itrue = img.clone()#.resize_(C, H*W)
    Ifake = ref.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


# def batch_SAM_CPU(img, ref):
#     I_true = img.data.cpu().numpy()
#     I_fake = ref.data.cpu().numpy()
#     N = I_true.shape[0]
#     C = I_true.shape[1]
#     H = I_true.shape[2]
#     W = I_true.shape[3]
#     batch_sam = 0
#     for i in range(N):
#         true = I_true[i,:,:,:].reshape(C, H*W)
#         fake = I_fake[i,:,:,:].reshape(C, H*W)
#         nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
#         denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
#         denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
#         sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
#         sam = sam/np.pi*180
#         # ignore pixels that have zero norm
#         idx = (np.isfinite(sam))
#         batch_sam += np.sum(sam[idx])/np.sum(idx)
#         if np.sum(~idx) != 0:
#             print("waring: some values were ignored when computing SAM")
#     return batch_sam/N
#
# def SAM_CPU(img, ref):
#     img = img.data.cpu().numpy()
#     ref = ref.data.cpu().numpy()
#     N = opt.batch_size
#     C = img.shape[0]
#     H = img.shape[1]
#     W = img.shape[2]
#     batch_sam = 0
#     for i in range(N):
#         true = img[i,:,:,:].reshape(C, H*W)
#         fake = ref[i,:,:,:].reshape(C, H*W)
#         nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
#         denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
#         denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
#         sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
#         sam = sam/np.pi*180
#         # ignore pixels that have zero norm
#         idx = (np.isfinite(sam))
#         batch_sam += np.sum(sam[idx])/np.sum(idx)
#         if np.sum(~idx) != 0:
#             print("waring: some values were ignored when computing SAM")
#     return batch_sam/N
##########################################################################################################

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return torch.tensor((np.mean(np.square(im1 - im2), dtype=np.float64)))


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C * H * W)
    psnr = 10. * np.log((data_range ** 2) / (err.data + esp)) / np.log(10.)
    return psnr

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    print(Phi_batch.shape, Phi_s_batch.shape)  # 256,310,28
    return Phi_batch, Phi_s_batch


def shift_mask(mask_3d, batch_size):
    # mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    # mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)  256,310,28
    return Phi_s_batch  # Phi_batch


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop_all(train_hsi, train_rgb, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        rgb_batch = []
        index_hsi = np.random.choice(range(len(train_hsi)), batch_size)
        processed_data1 = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)

        index_rgb = np.random.choice(range(len(train_rgb)), batch_size)
        processed_data2 = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)

        for i in range(batch_size):
            img_hsi = train_hsi[index_hsi[i]]
            h, w, _ = img_hsi.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data1[i, :, :, :] = img_hsi[x_index:x_index + crop_size, y_index:y_index + crop_size, :].cpu()
            img_rgb = train_rgb[index_rgb[i]]
            processed_data2[i, :, :, :] = img_rgb[x_index:x_index + crop_size, y_index:y_index + crop_size, :]

        gt_batch = torch.from_numpy(np.transpose(processed_data1, (0, 3, 1, 2))).cuda().float()
        rgb_batch = torch.from_numpy(np.transpose(processed_data2, (0, 3, 1, 2))).cuda().float()

        return gt_batch, rgb_batch
    else:
        index_rgb = np.random.choice(range(len(train_rgb)), batch_size)
        processed_data2 = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)

        index_hsi = np.random.choice(range(len(train_hsi)), batch_size)
        processed_data1 = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_rgb[index_rgb[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            img_rgb = train_rgb[index_rgb[i]]
            processed_data2[i, :, :, :] = img_rgb[x_index:x_index + crop_size, y_index:y_index + crop_size,
                                          :].cpu()
            img = train_hsi[index_hsi[i]]
            processed_data1[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data1, (0, 3, 1, 2)))
        rgb_batch = torch.from_numpy(np.transpose(processed_data2, (0, 3, 1, 2)))

        return gt_batch, rgb_batch


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    #
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def init_mask(mask, Phi, Phi_s, mask_type):
    if mask_type == 'Phi':
        input_mask = Phi
    elif mask_type == 'Phi_PhiPhiT':
        input_mask = (Phi, Phi_s)
    elif mask_type == 'Mask':
        input_mask = mask
    elif mask_type == None:
        input_mask = None
    return input_mask


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

#######################################################################################################
def init_mea(gt, mask, input_setting):
    if input_setting == 'H':
        input_mea = gen_mea(gt, mask, Y2H=True)
    return input_mea


def gen_mea(data_batch, mask3d_batch, Y2H=True):
    nC = data_batch.shape[0]
    temp = sift(mask3d_batch * data_batch, 2)
    mea = torch.sum(temp, 1)
    if Y2H:
        mea = mea / nC * 2
        H = sift_data(mea)
        return H
    return mea


def sift(input, step=2):
    [nC, row, col] = input.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i, :, step * i:step * i + col] = input[i, :, :]
    return output


def sift_data(input, nC, step=2):  # input [256,310]  output [256, 256, 28]
    [row, col] = input.shape
    # nC = 31
    output = torch.zeros(nC, row, col - (nC - 1) * step)
    for i in range(nC):
        output[i, :, :] = input[:, step * i:step * i + col - (nC - 1) * step]
    return output


#######################################################################################################
# mask_3d to mask_3d shift

def sift_mask(inputs, step=2):
    [nC, row, col] = inputs.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i ,:, step * i:step * i + col] = inputs[i, :, :]
    return output
#######################################################################################################
