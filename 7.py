from __future__ import print_function

import os
import re
import time
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.utils as vutils
from torchvision import transforms

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


def gn2d(channels):
    return nn.GroupNorm(num_groups=8, num_channels=channels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, pm_folder, press_folder, dem_path,
                 num_images, mask, transform=None, H_ELE=2597):
        super().__init__()
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

        # 建立 pressure 文件索引：date_str -> path
        self.press_map = {}
        for f in os.listdir(press_folder):
            if f.endswith('.png'):
                date_str = extract_date_str(f)
                self.press_map[date_str] = os.path.join(press_folder, f)

        print("Computing dataset statistics for normalization (press / dem)...")
        self.press_min, self.press_max = self._compute_press_stats()
        print(f"  Press : min={self.press_min:.4f}, max={self.press_max:.4f}")

        # DEM 只加载一次
        dem_img = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        if dem_img is None:
            raise FileNotFoundError(f"Cannot read DEM image: {dem_path}")
        if dem_img.ndim == 3:
            dem_img = cv2.cvtColor(dem_img, cv2.COLOR_BGR2GRAY)

        dem_img = dem_img.astype(np.float32)
        dem_min = float(dem_img.min())
        dem_max = float(dem_img.max())
        dem_denom = max(dem_max - dem_min, 1e-8)
        dem_img = (dem_img - dem_min) / dem_denom * 2.0 - 1.0  # [-1, 1]

        self.dem_img = torch.from_numpy(dem_img).unsqueeze(0).float()  # [1, H, W]

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

            # ---------- PM2.5 ----------
            pm_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if pm_img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            pm_img = pm_img.astype(np.float32) / self.h_ele
            if self.transform is not None:
                pm_img = self.transform(pm_img)
            pm_images.append(pm_img)

            # ---------- Pressure ----------
            if press_path is None:
                raise FileNotFoundError(f"Pressure image not found for time: {dt_str}")
            press_img = cv2.imread(press_path, cv2.IMREAD_UNCHANGED)
            if press_img is None:
                raise FileNotFoundError(f"Cannot read pressure image: {press_path}")
            press_img = press_img.astype(np.float32)
            press_img = self._minmax_norm(press_img, self.press_min, self.press_max)

            if self.transform is not None:
                press_img = ResizeTransform(self.transform.transforms[0].size)(press_img)
                press_img = ToTensor16Bit()(press_img)
            press_images.append(press_img)

        pm_seq = torch.stack(pm_images)         # [T, 1, H, W]
        press_seq = torch.stack(press_images)   # [T, 1, H, W]

        last_pm_real = pm_seq[-1].unsqueeze(0)  # [1, 1, H, W]
        masked_last_pm = last_pm_real * self.mask

        dem_img = self.dem_img.clone()          # [1, H, W]

        return pm_seq, press_seq, dem_img, masked_last_pm, last_pm_real

# ---------------------------------------------------------------------------
# Pressure encoder
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


class PressureViT1DEncoder(nn.Module):
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
            gn2d(out_channels),
            nn.ReLU(inplace=True)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, press_seq):
        B, T, C, H, W = press_seq.shape
        x = press_seq.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        Hp, Wp = x.shape[-2], x.shape[-1]
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :N, :]

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.norm(x)

        x = x.reshape(B, T, N, -1).permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * N, T, -1).transpose(1, 2).contiguous()
        x = self.temporal_conv(x)
        x = x.mean(dim=2)

        x = x.reshape(B, N, -1).transpose(1, 2).contiguous()
        x = x.reshape(B, -1, Hp, Wp)
        x = self.out_proj(x)
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

        # DEM 多尺度门控
        self.dem_fuse1 = DEMGateFusion(64, 64)
        self.dem_fuse2 = DEMGateFusion(128, 128)
        self.dem_fuse3 = DEMGateFusion(256, 256)
        self.dem_fuse4 = DEMGateFusion(512, 512)

    def forward(self, pm_seq, cp_layer, dem_feats=None):
        x = torch.cat((pm_seq, cp_layer), dim=1)
        x1 = self.cbam1(self.inc(x))
        x2 = self.cbam2(self.down1(x1))
        x3 = self.cbam3(self.down2(x2))
        x4 = self.down3(x3)

        if dem_feats is not None:
            d1, d2, d3, d4 = dem_feats
            x1 = self.dem_fuse1(x1, d1)
            x2 = self.dem_fuse2(x2, d2)
            x3 = self.dem_fuse3(x3, d3)
            x4 = self.dem_fuse4(x4, d4)

        x4 = F.dropout2d(x4, p=0.3, training=self.training)
        return x4, [x1, x2, x3]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            gn2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            gn2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class PMPressureFusion(nn.Module):
    def __init__(self, pm_dim=512, press_dim=512, out_dim=512, dropout=0.1):
        super().__init__()
        self.fuse_in = nn.Sequential(
            nn.Conv2d(pm_dim + press_dim, out_dim, kernel_size=3, padding=1, bias=False),
            gn2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResidualConvBlock(out_dim, dropout=dropout)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            gn2d(out_dim),
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

        x = torch.cat([pm_feat, press_feat], dim=1)
        x = self.fuse_in(x)
        x = self.res_block1(x)
        x = self.out_proj(x)
        fused = x + pm_feat
        return fused

class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            gn2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DEMEncoder(nn.Module):
    """
    DEM: [B, 1, 128, 128]
    输出多尺度特征:
      d1: [B, 64, 64, 64]
      d2: [B, 128, 32, 32]
      d3: [B, 256, 16, 16]
      d4: [B, 512, 8, 8]
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = ConvGNAct(in_channels, 32, stride=1)
        self.down1 = ConvGNAct(32, 64, stride=2)
        self.down2 = ConvGNAct(64, 128, stride=2)
        self.down3 = ConvGNAct(128, 256, stride=2)
        self.down4 = ConvGNAct(256, 512, stride=2)

    def forward(self, dem):
        x = self.stem(dem)  # [B, 32, 128, 128]
        d1 = self.down1(x)  # [B, 64, 64, 64]
        d2 = self.down2(d1)  # [B, 128, 32, 32]
        d3 = self.down3(d2)  # [B, 256, 16, 16]
        d4 = self.down4(d3)  # [B, 512, 8, 8]
        return [d1, d2, d3, d4]


class DEMGateFusion(nn.Module):
    """
    用 DEM 特征对 PM 特征做门控调制
    """

    def __init__(self, pm_channels, dem_channels, dropout=0.1):
        super().__init__()
        self.dem_proj = nn.Sequential(
            nn.Conv2d(dem_channels, pm_channels, kernel_size=1, bias=False),
            gn2d(pm_channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(pm_channels + pm_channels, pm_channels, kernel_size=3, padding=1, bias=False),
            gn2d(pm_channels),
            nn.Sigmoid()
        )
        self.refine = ResidualConvBlock(pm_channels, dropout=dropout)

    def forward(self, pm_feat, dem_feat):
        if dem_feat.shape[-2:] != pm_feat.shape[-2:]:
            dem_feat = F.interpolate(
                dem_feat,
                size=pm_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        dem_feat = self.dem_proj(dem_feat)
        g = self.gate(torch.cat([pm_feat, dem_feat], dim=1))
        x = pm_feat + g * dem_feat
        x = self.refine(x)
        return x


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
        pm_seq_len = n_channels - 1
        self.pm_encoder = PMEncoder(in_channels=pm_seq_len + 1, sequence_length=pm_seq_len)

        self.press_encoder = PressureViT1DEncoder(
            image_size=64,
            num_frames=n_channels,
            patch_size=8,
            embed_dim=256,
            depth_spatial=2,
            num_heads=8,
            out_channels=512
        )
        self.dem_encoder = DEMEncoder(in_channels=1)

        self.press_fusion = PMPressureFusion(
            pm_dim=512,
            press_dim=512,
            out_dim=512,
            dropout=0.1
        )
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pm_seq, cp_layer, press_seq, dem_img):
        dem_feats = self.dem_encoder(dem_img)

        pm_feat, skips = self.pm_encoder(
            pm_seq, cp_layer,
            dem_feats=dem_feats
        )

        press_feat = self.press_encoder(press_seq)
        fused_feat = self.press_fusion(pm_feat, press_feat)
        return self.decoder(fused_feat, skips)
# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf):
        super().__init__()
        self.layer1_image = nn.Sequential(
            nn.Conv2d(in_channels, int(ndf / 2), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer1_cp = nn.Sequential(
            nn.Conv2d(in_channels, int(ndf / 2), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, cp, _cpLayer):
        out_1 = self.layer1_image(cp)
        out_2 = self.layer1_cp(_cpLayer)
        out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--pm_path', default=r"dataset/utc_2013-2018_huabei")
    parser.add_argument('--press_path', default=r"dataset/press")
    parser.add_argument('--dem_path', default=r"dataset/dem/dem_128.png")
    parser.add_argument('--mask_path', default=r"mask/mask_huabei.txt")

    parser.add_argument('--outf', default=r'experiment/7')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=12)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--cuda', action='store_true', default=True)

    opt = parser.parse_args()

    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(os.path.join(opt.outf, "image"), exist_ok=True)
    os.makedirs(os.path.join(opt.outf, "nets"), exist_ok=True)
    os.makedirs(os.path.join(opt.outf, "error"), exist_ok=True)
    log_path = os.path.join(opt.outf, "error", "errlog.txt")

    H_ELE = 2597
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

    mask_ts = torch.tensor(np.loadtxt(opt.mask_path, dtype=np.float32))

    transform = transforms.Compose([
        ResizeTransform(opt.image_size),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MultiModalDataset(
        pm_folder=opt.pm_path,
        press_folder=opt.press_path,
        dem_path=opt.dem_path,
        num_images=opt.num_images,
        mask=mask_ts,
        transform=transform,
        H_ELE=H_ELE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    netG = MultiModalGenerator(n_channels=opt.num_images, n_classes=1).to(device)
    netD = Discriminator(in_channels=1, ndf=64).to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    warmup_epochs = 10
    total_epochs = opt.niter
    warmup_steps = warmup_epochs * len(dataloader)
    total_steps = total_epochs * len(dataloader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    schedulerD = LambdaLR(optimizerD, lr_lambda)
    schedulerG = LambdaLR(optimizerG, lr_lambda)

    def log_print(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log_print("Starting training...")
    start_time = time.time()

    save_epochs = (1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300)

    for epoch in range(1, opt.niter + 1):
        netG.train()
        netD.train()

        last_fake_pm = None
        last_real_pm = None

        for i, (imgs, press_imgs, dem_img, masked_last, last_pm_real) in enumerate(dataloader):
            real_pm = last_pm_real.to(device)
            if real_pm.dim() == 5:
                real_pm = real_pm.squeeze(1)

            input_seq = imgs[:, :-1].squeeze(2).to(device)  # [B, 11, H, W]
            cp_layer = masked_last.squeeze(2).to(device)  # [B, 1, H, W]
            press_imgs = press_imgs.to(device)  # [B, 12, 1, H, W]
            dem_img = dem_img.to(device)  # [B, 1, H, W]

            # ---- Discriminator ----
            optimizerD.zero_grad(set_to_none=True)
            out_real = netD(real_pm, cp_layer)
            loss_d_real = criterion_gan(out_real, torch.full_like(out_real, 0.9))

            fake_pm = netG(input_seq, cp_layer, press_imgs, dem_img)
            out_fake = netD(fake_pm.detach(), cp_layer)
            loss_d_fake = criterion_gan(out_fake, torch.zeros_like(out_fake))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            # ---- Generator ----
            optimizerG.zero_grad(set_to_none=True)
            out_g = netD(fake_pm, cp_layer)
            loss_adv = criterion_gan(out_g, torch.ones_like(out_g))
            loss_l1 = criterion_l1(fake_pm, real_pm)
            total_g_loss = loss_adv + opt.lambda_l1 * loss_l1
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            schedulerD.step()
            schedulerG.step()

            last_fake_pm = fake_pm.detach()
            last_real_pm = real_pm.detach()

            if (i + 1) % 40 == 0:
                with torch.no_grad():
                    fake_denorm = (fake_pm.detach() / 2.0 + 0.5) * H_ELE
                    real_denorm = (real_pm.detach() / 2.0 + 0.5) * H_ELE
                    mse_val = criterion_mse(fake_denorm, real_denorm).item()

                log_msg = (
                        '[%d/%d][%d/%d] Loss_D: %.4f (D_real: %.4f, D_fake: %.4f) '
                        'Loss_G: %.4f (G_adv: %.4f, L1: %.4f) MSE: %.2f' % (
                            epoch, opt.niter, i + 1, len(dataloader),
                            loss_d.item(), loss_d_real.item(), loss_d_fake.item(),
                            total_g_loss.item(), loss_adv.item(), loss_l1.item(),
                            mse_val
                        )
                )
                log_print(log_msg)

                vutils.save_image(
                    fake_pm.detach().cpu(),
                    os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_batch_{i + 1:03d}_fake.png'),
                    normalize=True
                )
                vutils.save_image(
                    real_pm.detach().cpu(),
                    os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_batch_{i + 1:03d}_real.png'),
                    normalize=True
                )

        if epoch in save_epochs and last_fake_pm is not None and last_real_pm is not None:
            vutils.save_image(
                last_fake_pm.cpu(),
                os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_fake.png'),
                normalize=True
            )
            vutils.save_image(
                last_real_pm.cpu(),
                os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_real.png'),
                normalize=True
            )
            torch.save(
                netG.state_dict(),
                os.path.join(opt.outf, "nets", f'netG_epoch_{epoch:03d}.pth')
            )
            log_print(f'-> Epoch {epoch} completed. netG saved.')

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        log_print(f"Epoch {epoch} completed. Cumulative training time: "
                  f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()