import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, pm_folder, temp_folder, num_images, mask, transform=None, h_ele=2597):
        self.pm_files = sorted([os.path.join(pm_folder, f) for f in os.listdir(pm_folder) if f.endswith('.png')])
        self.temp_folder = temp_folder
        self.num_images = num_images
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.transform = transform
        self.h_ele = h_ele

        # 预读取站点位置（假设所有CSV站点顺序一致）
        sample_temp = pd.read_csv(os.path.join(temp_folder, os.listdir(temp_folder)[0]), header=None)
        self.locations = sample_temp.iloc[:, :2].values  # [Lon, Lat]

    def __len__(self):
        return len(self.pm_files) - self.num_images + 1

    def __getitem__(self, idx):
        # 1. 获取 PM2.5 序列
        seq_files = self.pm_files[idx: idx + self.num_images]
        images = []
        for img_path in seq_files:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / self.h_ele
            if self.transform: image = self.transform(image)
            images.append(image)
        pm_seq = torch.stack(images)  # (12, 1, 64, 64)

        # 2. 获取对应日期的温度数据
        # 从最后一张图的文件名提取日期，如 "CN-Reanalysis2013010100.png" -> "20130101"
        last_file_name = os.path.basename(seq_files[-1])
        date_str = last_file_name[13:21]
        temp_path = os.path.join(self.temp_folder, f"{date_str}.csv")

        # 加载温度值并归一化 (简单示例：假设温度在-20到40度)
        temp_df = pd.read_csv(temp_path, header=None)
        temp_values = torch.tensor(temp_df.iloc[:, 2].values, dtype=torch.float32).unsqueeze(-1)
        temp_values = (temp_values - 10.0) / 20.0  # 自定义归一化

        # 3. 掩膜处理
        last_pm_real = pm_seq[-1].clone()
        masked_last_pm = last_pm_real * self.mask

        return pm_seq, masked_last_pm, temp_values, last_pm_real


def compute_adj(locations, sigma=0.1):
    """根据经纬度计算高斯核邻接矩阵"""
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(locations))
    adj = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    # 归一化
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return torch.from_numpy(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).float()