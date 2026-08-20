from __future__ import print_function

import os
import time
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.utils as vutils
import torchvision.transforms as transforms

from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, DownDS, UpDS
from sam_unet.unet_parts import OutConv
from sam_unet.layers import CBAM


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        return torch.from_numpy(img).float()


def extract_date_str(filename):
    import re
    m = re.search(r'(\d{10})', filename)
    if m is None:
        raise ValueError(f"Cannot extract 10-digit datetime from filename: {filename}")
    return m.group(1)


def denorm_to_physical(x, h_ele):
    """
    x: 模型输出，范围一般是 [-1, 1]
    返回物理量尺度
    """
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * h_ele


def batch_mae(pred, target):
    """
    pred, target: [B, 1, H, W]
    返回每个样本的 MAE，shape [B]
    """
    return (pred - target).abs().mean(dim=(1, 2, 3))


def batch_high_pm_recall(pred, target, threshold=75.0, eps=1e-8):
    """
    High-PM Recall
    以 target 中大于等于 threshold 的像素作为高浓度区域，
    计算 pred 对这些区域的召回率。

    pred, target: [B, 1, H, W]
    返回每个样本的 recall，shape [B]
    """
    target_high = target >= threshold
    pred_high = pred >= threshold

    tp = (pred_high & target_high).sum(dim=(1, 2, 3)).float()
    fn = ((~pred_high) & target_high).sum(dim=(1, 2, 3)).float()

    return tp / (tp + fn + eps)


def batch_gradient_mae(pred, target):
    """
    Gradient MAE
    计算水平和垂直梯度差的 MAE，再取平均。
    pred, target: [B, 1, H, W]
    返回每个样本的梯度 MAE，shape [B]
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    mae_dx = (pred_dx - target_dx).abs().mean(dim=(1, 2, 3))
    mae_dy = (pred_dy - target_dy).abs().mean(dim=(1, 2, 3))

    return (mae_dx + mae_dy) / 2.0


# ---------------------------------------------------------------------------
# Dataset: PM only
# ---------------------------------------------------------------------------
class PMDataset(Dataset):
    def __init__(self, pm_folder, num_images, mask, transform=None, H_ELE=990):
        self.pm_files = sorted([
            os.path.join(pm_folder, f)
            for f in os.listdir(pm_folder)
            if f.endswith('.png')
        ])
        self.num_images = num_images
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.transform = transform
        self.h_ele = H_ELE

    def __len__(self):
        return len(self.pm_files) - self.num_images + 1

    def __getitem__(self, idx):
        seq_files = self.pm_files[idx: idx + self.num_images]

        pm_images = []
        for img_path in seq_files:
            pm_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if pm_img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")

            pm_img = pm_img.astype(np.float32) / self.h_ele
            if self.transform is not None:
                pm_img = self.transform(pm_img)
            pm_images.append(pm_img)

        pm_seq = torch.stack(pm_images)            # [T, 1, H, W]
        last_pm_real = pm_seq[-1].unsqueeze(0)     # [1, 1, H, W]
        masked_last_pm = last_pm_real * self.mask  # [1, 1, H, W]

        return pm_seq, masked_last_pm, last_pm_real


# ---------------------------------------------------------------------------
# PM encoder
# ---------------------------------------------------------------------------
class PMEncoder(nn.Module):
    def __init__(self, in_channels=12, sequence_length=11,
                 kernels_per_layer=2, reduction_ratio=16):
        super().__init__()
        self.sequence_length = sequence_length
        self.inc = DoubleConvDS(in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)

    def forward(self, pm_seq, cp_layer):
        x = torch.cat((pm_seq, cp_layer), dim=1)  # [B, 12, H, W]
        x1 = self.cbam1(self.inc(x))
        x2 = self.cbam2(self.down1(x1))
        x3 = self.cbam3(self.down2(x2))
        x4 = self.down3(x3)
        x4 = F.dropout2d(x4, p=0.3, training=self.training)
        return x4, [x1, x2, x3]


class UNetDecoder(nn.Module):
    def __init__(self, n_classes=1, bilinear=True, kernels_per_layer=2):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = UpDS(768, 256 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(256, 128 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(128, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, n_classes)
        self.tanh = nn.Tanh()

    def forward(self, x, skips):
        x1, x2, x3 = skips
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return self.tanh(x)


# ---------------------------------------------------------------------------
# Generator: PM only
# ---------------------------------------------------------------------------
class PMGenerator(nn.Module):
    def __init__(self, n_channels=12, n_classes=1):
        super().__init__()
        pm_seq_len = n_channels - 1
        self.pm_encoder = PMEncoder(in_channels=pm_seq_len + 1, sequence_length=pm_seq_len)
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pm_seq, cp_layer):
        pm_feat, skips = self.pm_encoder(pm_seq, cp_layer)
        return self.decoder(pm_feat, skips)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test PM-only model')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=12)
    parser.add_argument('--outf', default=r'D:\MMSTGAN\experiment\1')
    parser.add_argument('--pm_path', default=r'D:\dataset\utc_2019_huabei')
    parser.add_argument('--mask_path', default=r'D:\mamba_GAN\code\mask\mask_huabei.txt')
    parser.add_argument('--netG', default='', help='指定权重路径（可选）')
    parser.add_argument('--test_epoch', type=int, default=200)
    parser.add_argument('--high_pm_threshold', type=float, default=75.0)
    parser.add_argument('--save_images', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    opt = parser.parse_args()

    print(opt)

    torch.manual_seed(3407)
    if opt.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

    H_ELE = 990

    mask_ts = torch.tensor(np.loadtxt(opt.mask_path, dtype=np.float32))
    transform = transforms.Compose([
        ResizeTransform(opt.image_size),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = PMDataset(
        pm_folder=opt.pm_path,
        num_images=opt.num_images,
        mask=mask_ts,
        transform=transform,
        H_ELE=H_ELE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    netG = PMGenerator(n_channels=opt.num_images, n_classes=1).to(device)

    epoch = opt.test_epoch
    netG_path = opt.netG if opt.netG else os.path.join(opt.outf, "nets", f"netG_epoch_{epoch:03d}.pth")

    if not os.path.exists(netG_path):
        print(f"Model not found: {netG_path}")
        return

    state = torch.load(netG_path, map_location=device)
    netG.load_state_dict(state, strict=True)
    netG.eval()
    print(f"Loaded model: {netG_path}")

    test_img_dir = os.path.join(opt.outf, "test_images")
    test_err_dir = os.path.join(opt.outf, "error")
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_err_dir, exist_ok=True)

    logfile = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}.txt")
    logfile_rmse = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}_rmse.txt")
    logfile_mae = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}_mae.txt")
    logfile_recall = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}_high_pm_recall.txt")
    logfile_gradmae = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}_gradient_mae.txt")

    def log_print(msg):
        print(msg)
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    rmse_list, mae_list, recall_list, gradmae_list = [], [], [], []

    start_time = time.time()

    with torch.no_grad():
        for i, (imgs, masked_last, last_pm_real) in enumerate(dataloader):
            real_pm = last_pm_real.to(device)
            if real_pm.dim() == 5:
                real_pm = real_pm.squeeze(1)  # [B, 1, H, W]

            input_seq = imgs[:, :-1].squeeze(2).to(device)   # [B, T-1, H, W]
            cp_layer = masked_last.squeeze(2).to(device)     # [B, 1, H, W]

            if not torch.isfinite(input_seq).all():
                raise RuntimeError(f"pm contains NaN/Inf at batch {i+1}")
            if not torch.isfinite(cp_layer).all():
                raise RuntimeError(f"cp contains NaN/Inf at batch {i+1}")

            fake = netG(input_seq, cp_layer)

            real_denorm = denorm_to_physical(real_pm, H_ELE)
            fake_denorm = denorm_to_physical(fake, H_ELE)

            mse_val = F.mse_loss(fake_denorm, real_denorm, reduction='mean').item()
            rmse = math.sqrt(mse_val)

            mae_vals = batch_mae(fake_denorm, real_denorm)
            recall_vals = batch_high_pm_recall(
                fake_denorm, real_denorm, threshold=opt.high_pm_threshold
            )
            grad_mae_vals = batch_gradient_mae(fake_denorm, real_denorm)

            mae = mae_vals.mean().item()
            high_pm_recall = recall_vals.mean().item()
            grad_mae = grad_mae_vals.mean().item()

            rmse_list.append(rmse)
            mae_list.append(mae)
            recall_list.append(high_pm_recall)
            gradmae_list.append(grad_mae)

            msg = (f'[Epoch {epoch}][Batch {i+1}/{len(dataloader)}] '
                   f'RMSE: {rmse:.4f}  MAE: {mae:.4f}  '
                   f'High-PM Recall: {high_pm_recall:.4f}  Gradient MAE: {grad_mae:.4f}')
            log_print(msg)

            with open(logfile_rmse, 'a', encoding='utf-8') as f:
                f.write(f'{rmse:.4f}\n')
            with open(logfile_mae, 'a', encoding='utf-8') as f:
                f.write(f'{mae:.4f}\n')
            with open(logfile_recall, 'a', encoding='utf-8') as f:
                f.write(f'{high_pm_recall:.4f}\n')
            with open(logfile_gradmae, 'a', encoding='utf-8') as f:
                f.write(f'{grad_mae:.4f}\n')

            if opt.save_images:
                vutils.save_image(
                    fake.detach().cpu(),
                    os.path.join(test_img_dir, f'test_epoch_{epoch:03d}_batch_{i+1:03d}_fake.png'),
                    normalize=True
                )
                vutils.save_image(
                    real_pm.detach().cpu(),
                    os.path.join(test_img_dir, f'test_epoch_{epoch:03d}_batch_{i+1:03d}_real.png'),
                    normalize=True
                )

    avg_rmse = float(np.mean(rmse_list)) if rmse_list else float('nan')
    avg_mae = float(np.mean(mae_list)) if mae_list else float('nan')
    avg_recall = float(np.mean(recall_list)) if recall_list else float('nan')
    avg_gradmae = float(np.mean(gradmae_list)) if gradmae_list else float('nan')

    summary = (
        f'\n[Epoch {epoch}] Batches: {len(rmse_list)}  '
        f'Avg RMSE: {avg_rmse:.6f}  Avg MAE: {avg_mae:.6f}  '
        f'Avg High-PM Recall: {avg_recall:.6f}  Avg Gradient MAE: {avg_gradmae:.6f}'
    )
    log_print(summary)

    with open(logfile_rmse, 'a', encoding='utf-8') as f:
        f.write(f'# Avg RMSE: {avg_rmse:.6f}\n')
    with open(logfile_mae, 'a', encoding='utf-8') as f:
        f.write(f'# Avg MAE: {avg_mae:.6f}\n')
    with open(logfile_recall, 'a', encoding='utf-8') as f:
        f.write(f'# Avg High-PM Recall: {avg_recall:.6f}\n')
    with open(logfile_gradmae, 'a', encoding='utf-8') as f:
        f.write(f'# Avg Gradient MAE: {avg_gradmae:.6f}\n')

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Testing completed. Time: {int(h):02d}:{int(m):02d}:{int(s):02d}")


if __name__ == "__main__":
    main()