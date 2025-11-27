import torch
import torch.nn as nn
from config import Config


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        r = x.real
        i = x.imag
        out_r = self.conv_r(r) - self.conv_i(i)
        out_i = self.conv_r(i) + self.conv_i(r)
        return torch.complex(out_r, out_i)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mag = torch.abs(x)
        phase = x / (mag + 1e-8)
        return torch.nn.functional.relu(mag + self.b) * phase


class ComplexBeamformerNet(nn.Module):
    def __init__(self, cfg=Config):
        super().__init__()
        MN = cfg.M * cfg.N

        # === 修正：改回你训练时用的通道数 (64, 128, 256) ===
        # 1. 信号特征提取
        self.layer1 = nn.Sequential(ComplexConv1d(MN, 64), ModReLU(64))
        self.layer2 = nn.Sequential(ComplexConv1d(64, 128), ModReLU(128))
        self.layer3 = nn.Sequential(ComplexConv1d(128, 256), ModReLU(256))
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 2. 导向矢量特征提取
        self.a_stream = nn.Sequential(
            nn.Linear(MN * 2, 512),
            nn.LeakyReLU(0.1),
            # 修正：输出改回 256，匹配你的 .pth
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1)
        )

        # 3. 融合层
        # Signal(256*2=512) + Vector(256) = 768
        self.fc_r = nn.Linear(512 + 256, MN)
        self.fc_i = nn.Linear(512 + 256, MN)

    def forward(self, x, a_tgt):
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.pool(feat).squeeze(-1)
        feat_x = torch.cat([feat.real, feat.imag], dim=1)  # (B, 512)

        a_in = torch.cat([a_tgt.real, a_tgt.imag], dim=1)
        feat_a = self.a_stream(a_in)  # (B, 256)

        feat_cat = torch.cat([feat_x, feat_a], dim=1)  # (B, 768)

        w_r = self.fc_r(feat_cat)
        w_i = self.fc_i(feat_cat)
        w_raw = torch.complex(w_r, w_i)

        # === MVDR 硬约束投影层 ===
        inner_prod = torch.sum(w_raw.conj() * a_tgt, dim=1, keepdim=True)
        norm_a_sq = torch.sum(a_tgt.conj() * a_tgt, dim=1, keepdim=True)
        correction = a_tgt * (inner_prod.conj() - 1.0) / (norm_a_sq + 1e-8)

        w_final = w_raw - correction
        return w_final