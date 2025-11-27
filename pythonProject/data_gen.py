import torch
import numpy as np
from config import Config


class FdaMimoSimulator:
    def __init__(self, cfg=Config):
        self.cfg = cfg
        self.MN = cfg.M * cfg.N

    def get_steering_vector(self, theta_deg, r_meter):
        """生成 FDA-MIMO 导向矢量 (物理核心)"""
        theta = np.deg2rad(theta_deg)

        # 接收导向矢量
        n_idx = np.arange(self.cfg.N)
        b_r = np.exp(-1j * 2 * np.pi * self.cfg.f0 * self.cfg.d_r * n_idx * np.sin(theta) / self.cfg.c)

        # 发射导向矢量 (含距离项)
        m_idx = np.arange(self.cfg.M)
        # 相位 = 角度项 + 距离项
        phi_angle = 2 * np.pi * self.cfg.f0 * self.cfg.d_t * m_idx * np.sin(theta) / self.cfg.c
        phi_range = -2 * np.pi * self.cfg.delta_f * m_idx * r_meter / self.cfg.c
        a_t = np.exp(1j * (phi_angle + phi_range))

        # Kronecker积
        v = np.kron(b_r, a_t)
        return torch.tensor(v, dtype=torch.complex64, device=self.cfg.device)

    def generate_batch(self):
        B = self.cfg.batch_size
        MN = self.MN
        L = self.cfg.L

        # 1. 随机目标参数
        theta_tgt = np.random.uniform(-60, 60, B)
        r_tgt = np.random.uniform(5000, 15000, B)

        # 2. 干扰参数 (定点训练：锁定在目标 +2km 处)
        theta_jam = theta_tgt + np.random.normal(0, 0.5, B)  # 角度主瓣重叠
        r_jam = r_tgt + 2000.0  # <--- 锁定距离差，确保挖坑精准

        snr_db = np.random.uniform(*self.cfg.SNR_range, B)
        jnr_db = np.random.uniform(*self.cfg.JNR_range, B)

        X_batch = torch.zeros(B, MN, L, dtype=torch.complex64, device=self.cfg.device)
        a_tgt_batch = torch.zeros(B, MN, dtype=torch.complex64, device=self.cfg.device)

        for i in range(B):
            v_s = self.get_steering_vector(theta_tgt[i], r_tgt[i])
            v_j = self.get_steering_vector(theta_jam[i], r_jam[i])
            a_tgt_batch[i] = v_s

            # 生成信号
            sig_wave = (torch.randn(1, L, device=self.cfg.device) + 1j * torch.randn(1, L,
                                                                                     device=self.cfg.device)) / np.sqrt(
                2)
            jam_wave = (torch.randn(1, L, device=self.cfg.device) + 1j * torch.randn(1, L,
                                                                                     device=self.cfg.device)) / np.sqrt(
                2)
            noise = (torch.randn(MN, L, device=self.cfg.device) + 1j * torch.randn(MN, L,
                                                                                   device=self.cfg.device)) / np.sqrt(2)

            # 叠加
            signal = 10 ** (snr_db[i] / 20) * v_s.unsqueeze(1) * sig_wave
            jamming = 10 ** (jnr_db[i] / 20) * v_j.unsqueeze(1) * jam_wave

            # 归一化 (防止梯度爆炸)
            raw_sum = signal + jamming + noise
            max_val = torch.max(torch.abs(raw_sum))
            X_batch[i] = raw_sum / (max_val + 1e-8)

        return X_batch, a_tgt_batch