import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='HRFT',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0,1,2,3,4,5,6')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/HRFT/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='HRFT', help='method name')
parser.add_argument('--pretrained_model_path', type=str,
                    default="/....../CasFormer/simulation/test_code/model_zoo/cave_model.pth",
                    help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT or None')

opt = parser.parse_args()
template.set_template(opt)

opt.mask_path = f"{opt.data_root}/Test/"
opt.test_path = f"{opt.data_root}/Test/cave_test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
