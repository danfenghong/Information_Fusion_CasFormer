import torch
from .HRFT import HRFT

def model_generator(method, pretrained_model_path=None):
    if method =="CasFormer":
        model = HRFT(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.item()}, strict=False)

    return model