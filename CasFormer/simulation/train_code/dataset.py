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
            # input:filepath
            # return:a list of mat (HSI,RGB,mask)
            # 对于每个file，读取得到HSI、RGB和mask，concat起来之后裁剪patch_per_img，放到self.mat_list中
            for per_data in file_list:
                if ".mat" in per_data:
                    # 读取
                    self.filenames.append(data_path + per_data)
                    mat = sio.loadmat(data_path + per_data)
                    HSI = mat['img']
                    RGB = mat["rgb"]
                    # mask = self.mask
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
        #            img = sio.loadmat(data_path + per_data)

        self.isTrain = opt.isTrain
        self.size = opt.size
        # self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.trainset_num
        # else:
        #     self.num = opt.testset_num
        # self.CAVE = CAVE
        # self.KAIST = KAIST, KAIST
        ## load mask

    def __getitem__(self, index):
        # index1 = 0; d=0
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
            RGB_data = data[0:3, :, :]  # 3 256 256
            HSI_data = data[3:31, :, :]  # 28   34
            mask = data[31, :, :]

            nC = HSI_data.shape[0]  # 0
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                mask3d = mask.repeat(31, 1, 1)  # 28 256 256   1, 1, 31
            else:
                mask3d = mask.repeat(28, 1, 1)  # 28 256 256   1, 1, 28

            # mask3d = np.transpose(mask3d, (1, 2, 0))
            temp_1 = mask3d * HSI_data  # 28, 256, 256(After Norm)  meas
            temp_2 = sift(temp_1, 2)  # 28, 256, 310  beta #0
            mea = torch.sum(temp_2, 0)  # 256, 310

            mea1 = mea / nC * 2  # 256, 310(Before Norm)
            H = sift_data(mea1, nC=28, step=int(2))  # 28, 256, 256(Before Norm)
            input = H

            mask3d_shift = sift_mask(mask3d, int(2))
            gt = HSI_data
            # RGB_data = np.transpose(RGB_data, (1, 2, 0))
            label_rgb = RGB_data

            pass
        else:
            img = self.mat_list[int(index / self.patch_per_img)]
            mask = self.mask
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask[np.newaxis, :, :])

            if self.dataset_type == "cave":
                # HSI = img['cave_data'] / 65535.0
                # RGB = img['cave_rgb'] / 65535.0
                HSI = img["cave_data"]
                RGB = img['cave_rgb']
                # for i in range(28):
                #     HSI[:,:,i]=(HSI[:,:,i]-np.min(HSI[:,:,i]))/(np.max(HSI[:,:,i])-np.min(HSI[:,:,i]))
                # HSI = (HSI-HSI.min())/(HSI.max()-HSI.min())
                # for i in range(3):
                #     RGB[:, :, i] = (RGB[:, :, i] - np.min(RGB[:, :, i])) / (np.max(RGB[:, :, i]) - np.min(RGB[:, :, i]))
                # HSI[HSI < 0] = 0
                # HSI[HSI > 1] = 1
                # RGB[RGB < 0] = 0
                # RGB[RGB > 1] = 1
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
                RGB_data = data[0:3, :, :]  # 3 256 256
                HSI_data = data[3:34, :, :]  # 28   34
                mask = data[34, :, :]  # 31
            #########################################################################################################33
            # RGB_data = np.transpose(RGB_data, (1, 2, 0))
            # HSI_data = np.transpose(HSI_data, (1, 2, 0))
            # h, w, nC, beta = 256, 256, 31, 0.9
            # RGB_data_1 = torch.unsqueeze(RGB_data[:, :, 2], 2)
            # RGB_data_2 = torch.unsqueeze(RGB_data[:, :, 1], 2)
            # RGB_data_3 = torch.unsqueeze(RGB_data[:, :, 0], 2)
            # RGB_data = torch.cat((RGB_data_1, RGB_data_2, RGB_data_3), dim=2)
            # RGB_data = F.interpolate(RGB_data, size=[nC], mode='linear')
            # RGB_data = RGB_data * (1 - beta)
            #####################################################################################################################

            else:
                RGB_data = data[0:3, :, :]  # 3 256 256
                HSI_data = data[3:31, :, :]  # 28   34
                mask = data[31, :, :]  # 31
            #####################################################################################################################
            # RGB_data = np.transpose(RGB_data, (1, 2, 0))
            # HSI_data = np.transpose(HSI_data, (1, 2, 0))
            # h, w, nC, beta = 256, 256, 28, 0.9
            # RGB_data_1 = torch.unsqueeze(RGB_data[:, :, 2], 2)
            # RGB_data_2 = torch.unsqueeze(RGB_data[:, :, 1], 2)
            # RGB_data_3 = torch.unsqueeze(RGB_data[:, :, 0], 2)
            # RGB_data = torch.cat((RGB_data_1, RGB_data_2, RGB_data_3), dim=2)
            # RGB_data = F.interpolate(RGB_data, size=[nC], mode='linear')
            # RGB_data = RGB_data * (1 - beta)
            #####################################################################################################################
            # mask_3d
            nC = HSI_data.shape[0]  # 0
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                mask3d = mask.repeat(31, 1, 1)  # 28 256 256   1, 1, 31
            else:
                mask3d = mask.repeat(28, 1, 1)  # 28 256 256   1, 1, 28

            # mask3d = np.transpose(mask3d, (1, 2, 0))
            temp_1 = mask3d * HSI_data  # 28, 256, 256(After Norm)  meas
            temp_2 = sift(temp_1, 2)  # 28, 256, 310  beta #0
            mea = torch.sum(temp_2, 0)  # 256, 310

            mea1 = mea / nC * 2  # 256, 310(Before Norm)
            if self.dataset_type == "CAVE31" or self.dataset_type == "ARAD":
                H = sift_data(mea1, nC=31, step=int(2))  # 28, 256, 256(Before Norm)
            else:
                H = sift_data(mea1, nC=28, step=int(2))  # 28, 256, 256(Before Norm)
            input = H

            mask3d_shift = sift_mask(mask3d, int(2))
            gt = HSI_data
            # RGB_data = np.transpose(RGB_data, (1, 2, 0))
            label_rgb = RGB_data
            # h w  c --->c h w
            # input = np.transpose(input, (2, 0, 1))
            # gt = np.transpose(gt, (2, 0, 1))
            # label_rgb = np.transpose(label_rgb, (2, 0, 1))
            # mask3d = np.transpose(mask3d, (2, 0, 1))
            # mask3d_shift = np.transpose(mask3d_shift, (2, 0, 1))

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
            # transforms.RandomRotation(degrees=45, center=(128, 128), expand=False)

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