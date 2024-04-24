from architecture import *
from utils import *
import scipy.io as scio
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option import opt
# from logging import Logger
import sys
from ssim_torch import *
from kl_loss import *
from tqdm import tqdm
from visualizer import *
import errno

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
txt_path = result_path+"output.txt"
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


# log_path = opt.outf + date_time + '/log.txt'
# model
def calculate_averages(lst, interval):
    averges = []
    for i in range(0, len(lst), interval):
        sublist = lst[i:i + interval]
        avg = sum(sublist) / len(sublist)
        averges.append(avg)
    return averges


model = torch.nn.DataParallel(model_generator(opt.method, opt.pretrained_model_path).cuda())

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
# criterion = nn.MSELoss().cuda()
criterion = nn.L1Loss(reduction="mean").cuda()

#kl = KL_Loss(T=10, method="mean").cuda()


def eval_metrics(epoch, loader, mode):
    # input:epoch test_loader
    # output:None
    psnr_mean_list = []
    ssim_mean_list = []
    mse_mean_list = []
    sam_mean_list = []

    begin = time.time()
    if mode == "test":
        duration = 50
    if mode == "train":
        duration = 300
    for i, (input_meas, gt_test, label_rgb_test, mask3d_test, mask3d_shift_test) in enumerate(loader):
        psnr_list, ssim_list, mse_list, sam_list = [], [], [], []
        gt_test = gt_test.cuda().float()
        model.eval()
        with torch.no_grad():
            out = model(input_meas, label_rgb_test, mask3d_shift_test)

        for k in range(gt_test.shape[0]):
            psnr_val = torch_psnr(out[k, :, :, :], gt_test[k, :, :, :])
            ssim_val = torch_ssim(out[k, :, :, :], gt_test[k, :, :, :])
            mse_val = compare_mse(out[k, :, :, :], gt_test[k, :, :, :])
            sam_val = SAM_GPU(out[k, :, :, :], gt_test[k, :, :, :])

            psnr_mean_list.append(psnr_val.detach().cpu().numpy())
            ssim_mean_list.append(ssim_val.detach().cpu().numpy())
            mse_mean_list.append(mse_val.detach().cpu().numpy())
            sam_mean_list.append(sam_val.detach().cpu().numpy())

    psnr_mean = np.mean(np.asarray(psnr_mean_list))
    ssim_mean = np.mean(np.asarray(ssim_mean_list))
    mse_mean = np.mean(np.asarray(mse_mean_list))
    sam_mean = np.mean(np.asarray(sam_mean_list))


    end = time.time()
    if mode=="test":
        psnr_list = calculate_averages(psnr_mean_list, duration)
        ssim_list = calculate_averages(ssim_mean_list, duration)
        mse_list = calculate_averages(mse_mean_list, duration)
        sam_list = calculate_averages(sam_mean_list, duration)
        print(f"===> Epoch {epoch + 1}: psnr list: {psnr_list}")
        print(f"===> Epoch {epoch + 1}: ssim list: {ssim_list}")
        print(f"===> Epoch {epoch + 1}: mse list: {mse_list}")
        print(f"===> Epoch {epoch + 1}: sam list: {sam_list}")

    print(
        '===> Epoch {}: {} mode  psnr = {:.2f}, ssim = {:.3f}, mse = {:.10f},sam = {:.3f}, time: {:.2f}'
        .format(epoch + 1, mode, psnr_mean, ssim_mean, mse_mean, sam_mean,
                (end - begin)))
    model.train()
    return psnr_mean, ssim_mean, mse_mean, sam_mean

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    # sys.stdout = Logger(log_path)
    sys.stdout=Logger(fpath=txt_path)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    print(opt)
    data_init_begin = time.time()

    # data_path_cave = "/....../CasFormer/datasets/data_test/cave_1028/"
    Dataset = dataset(opt, opt.data_path_cave, opt.mask_path, patch_per_img=300, dataset_type="cave", mode="train")

    # data_path_test = "/....../CasFormer/datasets/Test/all_test/1/"
    Test_Dataset = dataset(opt, opt.data_path_test, opt.TestMask_path, patch_per_img=50, dataset_type="cave",
                           mode="test")

    loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)
    loader_test = tud.DataLoader(Test_Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=False)

    data_init_end = time.time()

    print(f"dataset loading costs {data_init_end - data_init_begin} s\n")

    ## pipline of training
    for epoch in range(0, opt.max_epoch):
        print(f"begin the {epoch + 1}th epoch train:")
        model.train()

        epoch_loss = 0
        epoch_loss_gt = 0
        epoch_loss_kl = 0

        psnr_mean_list = []
        ssim_mean_list = []
        mse_mean_list = []
        sam_mean_list = []

        start_time = time.time()
        # for i, (input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
        for i, (input, gt, label_rgb, mask3d, mask3d_shift) in enumerate(loader_train):
            # print(f"begin {i}th iteration\n")
            iteration_time_begin = time.time()
            input, gt, label_rgb, mask3d, mask3d_shift = Variable(input), Variable(gt), Variable(
                label_rgb), Variable(mask3d), Variable(mask3d_shift)
            input, gt, label_rgb, mask3d, mask3d_shift = input.cuda(), gt.cuda(), label_rgb.cuda(), mask3d.cuda(), mask3d_shift.cuda()
            psnr_list, ssim_list, mse_list, sam_list = [], [], [], []
            model_out = model(input, label_rgb, mask3d_shift)
            #######################################################
            loss_gt = criterion(model_out, gt)
            loss = loss_gt  # + loss_kl
            epoch_loss += loss.item()
            epoch_loss_gt += loss_gt.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_time_end = time.time()

        elapsed_time = time.time() - start_time
        print(
            'epcoh = %4d , loss = %.10f,loss_gt = %.10f, time = %4.2f s' % (
                epoch + 1, epoch_loss / len(Dataset), epoch_loss_gt / len(Dataset), elapsed_time))
        if epoch in list(range(0,opt.max_epoch,10)):
            print(f"begin to calculate the {epoch + 1}th epoch train metrics ")
            psnr_train, ssim_train, mse_train, sam_train = eval_metrics(epoch=epoch, loader=loader_train, mode="train")
        print(f"begin to calculate the {epoch + 1}th epoch test metrics ")
        psnr_test, ssim_test, mse_test, sam_test = eval_metrics(epoch=epoch, loader=loader_test, mode="test")

        scheduler.step()

        torch.save(model, os.path.join(model_path, 'model_%03d.pth' % (epoch + 1)))


