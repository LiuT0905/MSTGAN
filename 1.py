from __future__ import print_function
import time
import argparse
import os
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torchvision import transforms
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, DownDS, UpDS
from sam_unet.unet_parts import OutConv
from sam_unet.layers import CBAM


# ---------------------------
# Utils
# ---------------------------

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

# ---------------------------
# Dataset: only PM2.5
# ---------------------------
class PMDataset(Dataset):
    def __init__(self, pm_folder, num_images, mask, transform=None, H_ELE=2597):
        self.pm_files = sorted([
            os.path.join(pm_folder, f)
            for f in os.listdir(pm_folder)
            if f.endswith('.png')
        ])
        self.num_images = num_images
        self.mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
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

# ---------------------------
# Transformer block
# ---------------------------
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

# ---------------------------
# PM encoder
# ---------------------------
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

class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))

# ---------------------------
# Decoder
# ---------------------------
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

# ---------------------------
# Generator: PM only
# ---------------------------
class PMGenerator(nn.Module):
    def __init__(self, n_channels=12, n_classes=1):
        super().__init__()
        pm_seq_len = n_channels - 1  # 前11帧作为输入，第12帧为监督目标
        self.pm_encoder = PMEncoder(in_channels=pm_seq_len + 1, sequence_length=pm_seq_len)
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pm_seq, cp_layer):
        pm_feat, skips = self.pm_encoder(pm_seq, cp_layer)
        return self.decoder(pm_feat, skips)

# ---------------------------
# Discriminator
# ---------------------------
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

# ---------------------------
# Train
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--pm_path', default=r"D:\dataset\utc_2013-2018_huabei")
    parser.add_argument('--mask_path', default=r"D:\mamba_GAN\code\mask\mask_huabei.txt")
    parser.add_argument('--outf', default=r'D:\MMSTGAN\experiment\1')
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
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    netG = PMGenerator(n_channels=opt.num_images, n_classes=1).to(device)
    netD = Discriminator(in_channels=1, ndf=64).to(device)

    optimizerG = torch.optim.Adam(
        netG.parameters(),
        lr=opt.lr,
        betas=(0.5, 0.999)
    )

    optimizerD = torch.optim.Adam(
        netD.parameters(),
        lr=opt.lr * 0.3,
        betas=(0.5, 0.999)
    )

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

        for i, (imgs, masked_last, last_pm_real) in enumerate(dataloader):
            real_pm = last_pm_real.to(device)
            if real_pm.dim() == 5:
                real_pm = real_pm.squeeze(1)  # [B,1,H,W]

            input_seq = imgs[:, :-1].squeeze(2).to(device)   # [B,11,H,W]
            cp_layer = masked_last.squeeze(2).to(device)     # [B,1,H,W]

            # ---- Discriminator ----
            optimizerD.zero_grad(set_to_none=True)

            out_real = netD(real_pm, cp_layer)
            loss_d_real = criterion_gan(out_real, torch.full_like(out_real, 0.9))

            fake_pm = netG(input_seq, cp_layer)
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
            optimizerG.step()

            schedulerD.step()
            schedulerG.step()

            last_fake_pm = fake_pm.detach()
            last_real_pm = real_pm.detach()

            if (i + 1) % 40 == 0:
                with torch.no_grad():
                    fake_denorm = (fake_pm.detach() / 2.0 + 0.5) * 2597.0
                    real_denorm = (real_pm.detach() / 2.0 + 0.5) * 2597.0
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
                    os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_batch_{i+1:03d}_fake.png'),
                    normalize=True
                )
                vutils.save_image(
                    real_pm.detach().cpu(),
                    os.path.join(opt.outf, "image", f'epoch_{epoch:03d}_batch_{i+1:03d}_real.png'),
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