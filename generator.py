import torch
import torch.nn as nn
from mamba_ssm import Mamba
from encoders.pm_encoder import PMEncoder
from encoders.temp_gcn import TempGCNEncoder
from sam_unet.unet_parts_depthwise_separable import UpDS
from sam_unet.unet_parts import OutConv

class UNetDecoder(nn.Module):
    def __init__(self, n_classes=1, bilinear=True, kernels_per_layer=2):
        super().__init__()
        factor = 2 if bilinear else 1
        # 输入是拼接后的特征 (512 + 256)
        self.up1 = UpDS(768, 256 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(256, 128 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(128, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, skips):
        x1Att, x2Att, x3Att = skips
        x = self.up1(x, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)
        return self.outc(x)

class MultiModalGenerator(nn.Module):
    def __init__(self, adj, n_channels=12, n_classes=1):
        super().__init__()
        # 1. 初始化各组件
        self.pm_encoder = PMEncoder(n_channels=n_channels + 1)  # PM序列 + Mask图
        self.temp_encoder = TempGCNEncoder(adj=adj)
        self.decoder = UNetDecoder(n_classes=n_classes)

        # 2. Mamba 瓶颈增强
        self.pre_mamba = nn.Conv2d(512, 16 * 12, kernel_size=1)
        self.mamba = Mamba(d_model=768, d_state=16, d_conv=4, expand=2)  # 768 是根据你的 image_size 计算的 D
        self.post_mamba = nn.Conv2d(16 * 12, 512, kernel_size=1)

    def forward(self, pm_seq, cp_layer, temp_data):
        # 模态 1：图像编码
        x_in = torch.cat([pm_seq, cp_layer], dim=1)
        x4, skips = self.pm_encoder(x_in)

        # 模态 2：气象编码
        temp_feat = self.temp_encoder(temp_data)  # (B, 512)

        # --- 跨模态融合 (Cross-Modal Fusion) ---
        # 策略：将温度特征作为全局偏置注入图像瓶颈层
        fusion_feat = x4 + temp_feat.view(temp_feat.size(0), 512, 1, 1)

        # --- Mamba 处理 ---
        x_mamba = self.pre_mamba(fusion_feat)
        B, C, H, W = x_mamba.shape
        x_mamba = x_mamba.view(B, 12, -1)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.view(B, C, H, W)
        x4_enhanced = self.post_mamba(x_mamba)

        # 解码
        logits = self.decoder(x4_enhanced, skips)
        return logits