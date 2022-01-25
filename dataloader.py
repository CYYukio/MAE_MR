import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import glob
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt


class DatasetTrain(Dataset):
    def __init__(self):
        super(DatasetTrain, self).__init__()
        self.dataList = sorted(glob.glob('D:/database/MR/mat_T1/' + '*.mat'))
        self.shape = 256

        self.len = len(self.dataList)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len
        img = scio.loadmat(self.dataList[i])['img']
        img = torch.from_numpy(img).float()
        img = torch.reshape(img, (1, self.shape, self.shape))

        return img


if __name__ == '__main__':
    d = DatasetTrain()
    img = d.__getitem__(100)
    shape = 256
    img = np.reshape(img, (shape, shape))

    plt.subplot(1, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.show()
