import torch
import torch.nn.functional as F
from glob import glob
import cv2
import numpy as np

# 构建训练集
class MyData(torch.utils.data.Dataset):

    def __init__(self, datapath, isTrain=True):
        trainFileList = sorted(glob(datapath + "train/*.jpg"))
        testFileList = sorted(glob(datapath + "test/*.jpg"))
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList
            
        self.nHeight = 28
        self.nWidth = 28

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return torch.from_numpy(data.reshape(1, self.nHeight, self.nWidth).astype(np.float32)), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)