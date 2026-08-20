from __future__ import print_function

import os
import re
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
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


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

def batch_high_pm_recall(pred, target, threshold, eps=1e-8):
    """
    pred, target: [B, 1, H, W]，物理尺度
    threshold: 高 PM 阈值（物理尺度）
    返回: [B]，每个样本的 recall；若该样本没有高 PM 像素，则为 nan
    """
    pred_pos = pred >= threshold
    target_pos = target >= threshold

    tp = (pred_pos & target_pos).reshape(pred.size(0), -1).sum(dim=1).float()
    fn = ((~pred_pos) & target_pos).reshape(pred.size(0), -1).sum(dim=1).float()
    positives = target_pos.reshape(pred.size(0), -1).sum(dim=1).float()

    recall = tp / (tp + fn + eps)
    recall = torch.where(
        positives > 0,
        recall,
        torch.full_like(recall, float("nan"))
    )
    return recall


def batch_gradient_mae(pred, target):
    """
    梯度 MAE：比较水平和垂直梯度差异
    pred, target: [B, 1, H, W]
    返回: [B]
    """
    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_tgt  = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_tgt  = target[:, :, :, 1:] - target[:, :, :, :-1]

    gx = torch.abs(dx_pred - dx_tgt).mean(dim=(1, 2, 3))
    gy = torch.abs(dy_pred - dy_tgt).mean(dim=(1, 2, 3))
    return 0.5 * (gx + gy)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, pm_folder, press_folder,
                 num_images, mask, transform=None, H_ELE=990):
        self.pm_files = sorted([
            os.path.join(pm_folder, f)
            for f in os.listdir(pm_folder)
            if f.endswith('.png')
        ])
        self.press_folder = press_folder
        self.num_images = num_images
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.transform = transform
        self.h_ele = H_ELE

        self.press_map = {}
        for f in os.listdir(press_folder):
            if f.endswith('.png'):
                date_str = extract_date_str(f)
                self.press_map[date_str] = os.path.join(press_folder, f)

        print("Computing dataset statistics for normalization (press)...")
        self.press_min, self.press_max = self._compute_press_stats()
        print(f"  Press : min={self.press_min:.4f}, max={self.press_max:.4f}")

    def _compute_press_stats(self):
        press_min, press_max = float('inf'), float('-inf')

        for path in self.press_map.values():
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read pressure image: {path}")
            img = img.astype(np.float32)
            press_min = min(press_min, float(img.min()))
            press_max = max(press_max, float(img.max()))

        return press_min, press_max

    def _minmax_norm(self, x, vmin, vmax):
        denom = vmax - vmin
        if denom < 1e-8:
            denom = 1e-8
        return (x - vmin) / denom * 2.0 - 1.0

    def __len__(self):
        return len(self.pm_files) - self.num_images + 1

    def __getitem__(self, idx):
        seq_files = self.pm_files[idx: idx + self.num_images]

        pm_images = []
        press_images = []

        for img_path in seq_files:
            fname = os.path.basename(img_path)
            dt_str = extract_date_str(fname)

            press_path = self.press_map.get(dt_str, None)
            if press_path is None:
                raise FileNotFoundError(f"Pressure image not found for time: {dt_str}")

            # ---------- PM2.5 ----------
            pm_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if pm_img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            pm_img = pm_img.astype(np.float32) / self.h_ele
            if self.transform is not None:
                pm_img = self.transform(pm_img)
            pm_images.append(pm_img)

            # ---------- 气压 ----------
            press_img = cv2.imread(press_path, cv2.IMREAD_UNCHANGED)
            if press_img is None:
                raise FileNotFoundError(f"Cannot read pressure image: {press_path}")
            press_img = press_img.astype(np.float32)
            press_img = self._minmax_norm(press_img, self.press_min, self.press_max)

            if self.transform is not None:
                press_img = ResizeTransform(self.transform.transforms[0].size)(press_img)
                press_img = ToTensor16Bit()(press_img)
            press_images.append(press_img)

        pm_seq = torch.stack(pm_images)          # [T, 1, H, W]
        press_seq = torch.stack(press_images)    # [T, 1, H, W]

        last_pm_real = pm_seq[-1].unsqueeze(0)   # [1, 1, H, W]
        masked_last_pm = last_pm_real * self.mask

        return pm_seq, press_seq, masked_last_pm, last_pm_real


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Pressure encoder
# ---------------------------------------------------------------------------
class PressViT1DEncoder(nn.Module):
    def __init__(self, image_size=64, num_frames=12, patch_size=8,
                 embed_dim=256, depth_spatial=2, num_heads=8,
                 out_channels=512, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        grid = image_size // patch_size
        self.num_patches = grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth_spatial)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, press_seq):
        # press_seq: [B, T, 1, H, W]
        B, T, C, H, W = press_seq.shape
        assert T == self.num_frames, f"Expected T={self.num_frames}, got {T}"

        x = press_seq.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        Hp, Wp = x.shape[-2], x.shape[-1]
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2)   # [B*T, N, embed_dim]
        x = x + self.pos_embed[:, :N, :]

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.norm(x)

        x = x.reshape(B, T, N, -1).permute(0, 2, 1, 3).contiguous()  # [B, N, T, embed_dim]
        x = x.reshape(B * N, T, -1).transpose(1, 2).contiguous()      # [B*N, embed_dim, T]
        x = self.temporal_conv(x)
        x = x.mean(dim=2)                                             # [B*N, embed_dim]

        x = x.reshape(B, N, -1).transpose(1, 2).contiguous()          # [B, embed_dim, N]
        x = x.reshape(B, -1, Hp, Wp)                                  # [B, embed_dim, Hp, Wp]
        x = self.out_proj(x)                                          # [B, 512, Hp, Wp]
        return x


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
        x = torch.cat((pm_seq, cp_layer), dim=1)
        x1 = self.cbam1(self.inc(x))
        x2 = self.cbam2(self.down1(x1))
        x3 = self.cbam3(self.down2(x2))
        x4 = self.down3(x3)
        x4 = F.dropout2d(x4, p=0.3, training=self.training)
        return x4, [x1, x2, x3]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class PressureConcatFusion(nn.Module):
    def __init__(self, pm_dim=512, press_dim=512, out_dim=512, dropout=0.1):
        super().__init__()
        self.fuse_in = nn.Sequential(
            nn.Conv2d(pm_dim + press_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResidualConvBlock(out_dim, dropout=dropout)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)
        )

    def forward(self, pm_feat, press_feat):
        if press_feat.shape[-2:] != pm_feat.shape[-2:]:
            press_feat = F.interpolate(
                press_feat,
                size=pm_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        x = torch.cat([pm_feat, press_feat], dim=1)  # [B, 1024, H, W]
        x = self.fuse_in(x)                          # [B, 512, H, W]
        x = self.res_block1(x)                       # [B, 512, H, W]
        x = self.out_proj(x)                         # [B, 512, H, W]
        fused = x + pm_feat
        return fused


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
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
# Generator
# ---------------------------------------------------------------------------
class MultiModalGenerator(nn.Module):
    def __init__(self, n_channels=12, n_classes=1):
        super().__init__()
        pm_seq_len = n_channels - 1  # 前 11 帧作为输入，第 12 帧作为监督目标
        self.pm_encoder = PMEncoder(in_channels=pm_seq_len + 1, sequence_length=pm_seq_len)
        self.press_encoder = PressViT1DEncoder(
            image_size=64,
            num_frames=n_channels,
            patch_size=8,
            embed_dim=256,
            depth_spatial=2,
            num_heads=8,
            out_channels=512
        )
        self.fusion = PressureConcatFusion(
            pm_dim=512, press_dim=512, out_dim=512, dropout=0.1
        )
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pm_seq, cp_layer, press_seq):
        pm_feat, skips = self.pm_encoder(pm_seq, cp_layer)
        press_feat = self.press_encoder(press_seq)
        fused_feat = self.fusion(pm_feat, press_feat)
        return self.decoder(fused_feat, skips)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test MMSTGAN (PM + Pressure only)')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=12)
    parser.add_argument('--outf', default=r'D:\MMSTGAN\experiment\3')
    parser.add_argument('--pm_path', default=r'D:\dataset\utc_2019_huabei')
    parser.add_argument('--press_path', default=r'D:\dataset\press')
    parser.add_argument('--mask_path', default=r'D:\mamba_GAN\code\mask\mask_huabei.txt')
    parser.add_argument('--netG', default='', help='指定权重路径（可选）')
    parser.add_argument('--test_epoch', type=int, default=200)
    parser.add_argument('--save_images', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--high_pm_threshold', type=float, default=75.0)
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

    dataset = MultiModalDataset(
        pm_folder=opt.pm_path,
        press_folder=opt.press_path,
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

    netG = MultiModalGenerator(n_channels=opt.num_images, n_classes=1).to(device)

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
    logfile_grad_mae = os.path.join(test_err_dir, f"test_epoch_{epoch:03d}_grad_mae.txt")

    mse_list, rmse_list, mae_list = [], [], []
    recall_list, grad_mae_list = [], []

    def log_print(msg):
        print(msg)
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    start_time = time.time()

    with torch.no_grad():
        for i, (imgs, press_imgs, masked_last, last_pm_real) in enumerate(dataloader):
            real_pm = last_pm_real.to(device)
            if real_pm.dim() == 5:
                real_pm = real_pm.squeeze(1)   # [B, 1, H, W]

            input_seq = imgs[:, :-1].squeeze(2).to(device)    # [B, T-1, H, W]
            cp_layer = masked_last.squeeze(2).to(device)      # [B, 1, H, W]
            press_imgs = press_imgs.to(device)                # [B, T, 1, H, W]

            for name, tensor in [
                ("pm", input_seq),
                ("cp", cp_layer),
                ("press", press_imgs),
            ]:
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{name} contains NaN/Inf at batch {i+1}")

            fake = netG(input_seq, cp_layer, press_imgs)

            real_denorm = denorm_to_physical(real_pm, H_ELE)
            fake_denorm = denorm_to_physical(fake, H_ELE)

            rmse = torch.sqrt(F.mse_loss(fake_denorm, real_denorm)).item()
            mae = torch.mean(torch.abs(fake_denorm - real_denorm)).item()

            recall_vals = batch_high_pm_recall(
                fake_denorm, real_denorm, threshold=opt.high_pm_threshold
            )
            recall = torch.nanmean(recall_vals).item() if not torch.isnan(recall_vals).all() else float('nan')

            grad_mae_vals = batch_gradient_mae(fake_denorm, real_denorm)
            grad_mae = grad_mae_vals.mean().item()

            rmse_list.append(rmse)
            mae_list.append(mae)
            recall_list.append(recall)
            grad_mae_list.append(grad_mae)

            msg = (
                f'[Epoch {epoch}][Batch {i + 1}/{len(dataloader)}] '
                f'RMSE: {rmse:.4f}  MAE: {mae:.4f}  '
                f'High-PM Recall(@{opt.high_pm_threshold:g}): {recall:.4f}  '
                f'Gradient MAE: {grad_mae:.4f}'
            )
            log_print(msg)

            with open(logfile_rmse, 'a', encoding='utf-8') as f:
                f.write(f'{rmse:.6f}\n')
            with open(logfile_mae, 'a', encoding='utf-8') as f:
                f.write(f'{mae:.6f}\n')
            with open(logfile_recall, 'a', encoding='utf-8') as f:
                f.write(f'{recall:.6f}\n')
            with open(logfile_grad_mae, 'a', encoding='utf-8') as f:
                f.write(f'{grad_mae:.6f}\n')
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

    avg_rmse = float(np.nanmean(rmse_list)) if rmse_list else float('nan')
    avg_mae = float(np.nanmean(mae_list)) if mae_list else float('nan')
    avg_recall = float(np.nanmean(recall_list)) if recall_list else float('nan')
    avg_grad_mae = float(np.nanmean(grad_mae_list)) if grad_mae_list else float('nan')

    summary = (
        f'\n[Epoch {epoch}] Batches: {len(rmse_list)}  '
        f'Avg RMSE: {avg_rmse:.6f}  Avg MAE: {avg_mae:.6f}  '
        f'Avg High-PM Recall(@{opt.high_pm_threshold:g}): {avg_recall:.6f}  '
        f'Avg Gradient MAE: {avg_grad_mae:.6f}'
    )
    log_print(summary)

    with open(logfile_rmse, 'a', encoding='utf-8') as f:
        f.write(f'# Avg RMSE: {avg_rmse:.6f}\n')
    with open(logfile_mae, 'a', encoding='utf-8') as f:
        f.write(f'# Avg MAE: {avg_mae:.6f}\n')
    with open(logfile_recall, 'a', encoding='utf-8') as f:
        f.write(f'# Avg High-PM Recall(@{opt.high_pm_threshold:g}): {avg_recall:.6f}\n')
    with open(logfile_grad_mae, 'a', encoding='utf-8') as f:
        f.write(f'# Avg Gradient MAE: {avg_grad_mae:.6f}\n')

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Testing completed. Time: {int(h):02d}:{int(m):02d}:{int(s):02d}")


if __name__ == "__main__":
    main()