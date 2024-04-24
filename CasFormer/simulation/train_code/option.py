import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='casFormer',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0,1,2,3,4,5,6')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
#parser.add_argument('--data_path_cave', default='../../datasets/Train/kaist_train/', type=str,
#                        help='path of data')
parser.add_argument('--data_path_cave', default='../../datasets/Train/cave_train/', type=str,
                     help='path of data')
#parser.add_argument('--data_path_ARAD', default='../../datasets/Test/ARAD_50/', type=str,
#                      help='path of data')
# parser.add_argument('--data_path_KAIST', default='../../datasets/Train/kaist_train/', type=str,
#                     help='path of data')
parser.add_argument('--mask_path', default='../../datasets/Train/mask_256.mat', type=str,
                    help='path of mask')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/casFormer/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='casFormer', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--data_path_test",default='../../datasets/Test/cave_test/', type=str,
                        help='path of data')
#parser.add_argument("--data_path_test",default='../../datasets/Test/ARAD/', type=str, help='path of data')
parser.add_argument('--TrainMask_path', default='../../datasets/Train/mask256.mat', type=str,
                    help='path of mask')
parser.add_argument('--TestMask_path', default='../../datasets/Test/mask256.mat', type=str,
                    help='path of mask')

# Training specifications
parser.add_argument("--size", default=256, type=int, help='cropped patch size')
parser.add_argument("--epoch_sam_num", default=5000, type=int, help='total number of trainset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument('--batch_size', type=int, default=16, help='the number of HSIs per batch')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.6, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0007)
# parser.add_argument("--train_patch_per_img", type=int, default=10)
# parser.add_argument("--test_patch_per_img", type=int, default=10)


opt = parser.parse_args()
template.set_template(opt)

opt.trainset_num = 20000 // ((opt.size // 96) ** 2)


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False