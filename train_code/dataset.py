import os
import torch
from utils import *
import torch.utils.data as tud
import torchvision.transforms as transforms

class dataset(tud.Dataset):
    def __init__(self, opt, data_path, mask_path, patch_per_img, dataset_type, mode):
        super(dataset, self).__init__()
        self.data_path = data_path
        self.mask_path = mask_path
        self.patch_per_img = patch_per_img
        self.mode = mode
        self.dataset_type = dataset_type
        self.filenames = []
        self.mat_list = []

        mask_data = sio.loadmat(opt.mask_path)
        self.mask = mask_data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))
        crop_size = 256

        file_list = os.listdir(self.data_path)
        location = []

        # for idx in range(1):
        if self.dataset_type == "kaist":
            for per_data in file_list:
                if ".mat" in per_data:
                    self.filenames.append(data_path + per_data)
                    mat = sio.loadmat(data_path + per_data)
                    HSI = mat['img']
                    RGB = mat["rgb"]
                    data = np.concatenate((RGB, HSI), axis=2)
                    H, W, _ = HSI.shape
                    x = list(range(crop_size // 2, H - crop_size // 2))
                    y = list(range(crop_size // 2, W - crop_size // 2))
                    for i in x:
                        for j in y:
                            location.append((i, j))
                    for i in range(patch_per_img):
                        sample = np.random.choice(location, size=1, replace=False)
                        x_0 = sample[0]
                        y_0 = sample[1]
                        mat = data[x_0 - 128:x_0 + 128, y_0 - 128:y_0 + 128, :]
                        self.mat_list.append(mat)
        else:
            for per_data in file_list:
                if ".mat" in per_data:
                    mat = sio.loadmat(data_path + per_data)
                    self.filenames.append(data_path + per_data)
                    self.mat_list.append(mat)
        self.isTrain = opt.isTrain
        self.size = opt.size
        
        if self.isTrain == True:
            self.num = opt.trainset_num

    def __getitem__(self, index):

        if self.dataset_type == "kaist":
            img = self.mat_list[index]
            HSI = img[:, :, 0:28]
            RGB = img[:, :, 28:31]
            # mask = img[:,:,31]
            mask = self.mask
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask[np.newaxis, :, :])
            HSI = HSI.astype(np.float32)
            RGB = RGB.astype(np.float32)
            RGB_HSI = np.concatenate((RGB, HSI), axis=2)
            crop_RGB_HSI = self.process_rgb_hsi_kaist(RGB_HSI)
            data = torch.concatenate((crop_RGB_HSI, mask), axis=0).numpy()
            data = np.transpose(data, (1, 2, 0))
            if self.mode == "train":
                data = self.process_data_train(data)
            if self.mode == "test":
                data = self.process_data_test(data)
            RGB_data = data[0:3, :, :]
            HSI_data = data[3:31, :, :]
            mask = data[31, :, :]
            nC = HSI_data.shape[0]
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                mask3d = mask.repeat(31, 1, 1)
            else:
                mask3d = mask.repeat(28, 1, 1)
            temp_1 = mask3d * HSI_data
            temp_2 = sift(temp_1, 2)
            mea = torch.sum(temp_2, 0)
            mea1 = mea / nC * 2
            H = sift_data(mea1, nC=28, step=int(2))
            input = H
            mask3d_shift = sift_mask(mask3d, int(2))
            gt = HSI_data
            label_rgb = RGB_data

            pass
        else:
            img = self.mat_list[int(index / self.patch_per_img)]
            mask = self.mask
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask[np.newaxis, :, :])

            if self.dataset_type == "cave":
               
                HSI = img["cave_data"]
                RGB = img['cave_rgb']
               
            elif self.dataset_type == "CAVE31":
                HSI = img["M"]
                RGB = img['RGB']
            elif self.dataset_type == "ARAD":
                HSI = img['cube']
                RGB = img['rgb']
                HSI[HSI < 0] = 0
                HSI[HSI > 1] = 1
                RGB[RGB < 0] = 0
                RGB[RGB > 1] = 1
            else:
                raise ValueError("no this mode in dataset")

            HSI = HSI.astype(np.float32)
            RGB = RGB.astype(np.float32)

            RGB_HSI = np.concatenate((RGB, HSI), axis=2)
            crop_RGB_HSI = self.process_rgb_hsi(RGB_HSI)

            data = torch.concatenate((crop_RGB_HSI, mask), axis=0).numpy()
            data = np.transpose(data, (1, 2, 0))
            if self.mode == "train":
                data = self.process_data_train(data)
            if self.mode == "test":
                data = self.process_data_test(data)

            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                RGB_data = data[0:3, :, :]
                HSI_data = data[3:34, :, :]  
                mask = data[34, :, :] 

            else:
                RGB_data = data[0:3, :, :]  
                HSI_data = data[3:31, :, :]  
                mask = data[31, :, :]  

            nC = HSI_data.shape[0]
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                mask3d = mask.repeat(31, 1, 1)
            else:
                mask3d = mask.repeat(28, 1, 1)

            temp_1 = mask3d * HSI_data
            temp_2 = sift(temp_1, 2)
            mea = torch.sum(temp_2, 0)

            mea1 = mea / nC * 2
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                H = sift_data(mea1, nC=31, step=int(2))
            else:
                H = sift_data(mea1, nC=28, step=int(2))
            input = H

            mask3d_shift = sift_mask(mask3d, int(2))
            gt = HSI_data
           
        return input, gt, label_rgb, mask3d, mask3d_shift

    def __len__(self):
        return len(self.filenames) * self.patch_per_img

    def process_rgb_hsi(self, data):
        transforms_list1 = [
            transforms.ToTensor(),
            transforms.RandomCrop((256, 256))
        ]
        data_transform_1 = transforms.Compose(transforms_list1)
        return data_transform_1(data)

    def process_data_train(self, data):
        transforms_list2 = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
        data_transform_2 = transforms.Compose(transforms_list2)
        return data_transform_2(data)

    def process_rgb_hsi_kaist(self, data):
        transforms_list3 = [
            transforms.ToTensor()
        ]
        data_transform_3 = transforms.Compose(transforms_list3)
        return data_transform_3(data)

    def process_data_test(self, data):
        transforms_list4 = [
            transforms.ToTensor()
        ]
        data_transform_4 = transforms.Compose(transforms_list4)
        return data_transform_4(data)