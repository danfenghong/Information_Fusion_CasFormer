import torch
from .HRFT import HRFT
from .HRFT_31 import HRFT as HRFT31


def model_generator(method, pretrained_model_path=None):
    if method == 'casFormer':
        model = HRFT(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    elif method == 'mst_m':
        model = HRFT(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'mst_l':
        model = HRFT(dim=28, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'casFormer_31':
        model = HRFT31(dim=31, stage=2, num_blocks=[1, 1, 1]).cuda()
    return model
