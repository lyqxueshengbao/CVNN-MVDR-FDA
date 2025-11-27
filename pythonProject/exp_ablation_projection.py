import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexConv1d, ModReLU


class ComplexBeamformerNetNoProjection(nn.Module):
    """无投影层版本：用软约束 Loss"""
    def __init__(self, cfg=Config):
        super().__init__()
        MN = cfg.M * cfg.N

        self.layer1 = nn.Sequential(ComplexConv1d(MN, 64), ModReLU(64))
        self.layer2 = nn.Sequential(ComplexConv1d(64, 128), ModReLU(128))
        self.layer3 = nn.Sequential(ComplexConv1d(128, 256), ModReLU(256))
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.a_stream = nn.Sequential(
            nn.Linear(MN * 2, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1)
        )

        self.fc_r = nn.Linear(512 + 256, MN)
        self.fc_i = nn.Linear(512 + 256, MN)

    def forward(self, x, a_tgt):
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.pool(feat).squeeze(-1)
        feat_x = torch.cat([feat.real, feat.imag], dim=1)

        a_in = torch.cat([a_tgt.real, a_tgt.imag], dim=1)
        feat_a = self.a_stream(a_in)

        feat_cat = torch.cat([feat_x, feat_a], dim=1)

        w_r = self.fc_r(feat_cat)
        w_i = self.fc_i(feat_cat)
        w_raw = torch.complex(w_r, w_i)

        # 无投影层，直接输出
        return w_raw


def soft_constraint_loss(w_pred, X_in, a_tgt, lambda_c=10.0):
    """带软约束的损失函数"""
    w_H = w_pred.conj().unsqueeze(1)
    y = torch.matmul(w_H, X_in).squeeze(1)
    
    # 功率最小化
    loss_power = torch.mean(torch.abs(y) ** 2)
    
    # 软约束: |w^H * a - 1|^2
    constraint = torch.sum(w_pred.conj() * a_tgt, dim=1)
    loss_constraint = torch.mean(torch.abs(constraint - 1.0) ** 2)
    
    return loss_power + lambda_c * loss_constraint


def train_no_projection_model():
    """训练无投影层的模型"""
    print("训练无投影层模型 (使用软约束)...")
    cfg = Config()
    cfg.epochs = 100  # 快速训练
    
    sim = FdaMimoSimulator(cfg)
    model = ComplexBeamformerNetNoProjection(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(cfg.epochs):
        model.train()
        X, a_tgt = sim.generate_batch()
        w_pred = model(X, a_tgt)
        
        loss = soft_constraint_loss(w_pred, X, a_tgt, lambda_c=10.0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4e}")
    
    torch.save(model.state_dict(), "fda_no_projection.pth")
    print("无投影层模型训练完成！")
    return model


def calculate_beampattern(w, sim, target_theta, target_range, scan_ranges):
    gains = []
    for r in scan_ranges:
        a = sim.get_steering_vector(target_theta, r).to(w.device)
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


def run_ablation_experiment():
    """实验 2: 投影层消融实验"""
    print("=" * 60)
    print("实验 2: 投影层消融实验")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    # 检查是否存在无投影层模型
    import os
    if not os.path.exists("fda_no_projection.pth"):
        print("未找到无投影层模型，开始训练...")
        train_no_projection_model()
    
    # 加载完整模型 (有投影层)
    from model import ComplexBeamformerNet
    model_with_proj = ComplexBeamformerNet(cfg).to(cfg.device)
    model_with_proj.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model_with_proj.eval()
    
    # 加载无投影层模型
    model_no_proj = ComplexBeamformerNetNoProjection(cfg).to(cfg.device)
    model_no_proj.load_state_dict(torch.load("fda_no_projection.pth"))
    model_no_proj.eval()
    
    # 测试场景
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    r_jam = 12000.0
    
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    L = cfg.L
    
    # 生成测试数据 (50dB 强干扰)
    sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
    
    X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise
    X_input = X_input / torch.max(torch.abs(X_input))
    X_input = X_input.unsqueeze(0)
    a_tgt_test = v_s.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        w_with_proj = model_with_proj(X_input, a_tgt_test).squeeze(0)
        w_no_proj = model_no_proj(X_input, a_tgt_test).squeeze(0)
    
    # 计算目标增益 (验证约束)
    gain_tgt_with = torch.abs(torch.vdot(w_with_proj, v_s)) ** 2
    gain_tgt_no = torch.abs(torch.vdot(w_no_proj, v_s)) ** 2
    
    print(f"\n目标增益 (有投影层): {10 * np.log10(gain_tgt_with.item()):.2f} dB")
    print(f"目标增益 (无投影层): {10 * np.log10(gain_tgt_no.item()):.2f} dB")
    print(f"理论值应为: 0 dB (增益为1)")
    
    # 绘制波束图
    scan_ranges = np.linspace(6000, 14000, 500)
    resp_with = calculate_beampattern(w_with_proj, sim, theta_tgt, r_tgt, scan_ranges)
    resp_no = calculate_beampattern(w_no_proj, sim, theta_tgt, r_tgt, scan_ranges)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(scan_ranges / 1000, resp_with, 'b-', linewidth=2.5, label='With Projection Layer (Hard Constraint)')
    plt.plot(scan_ranges / 1000, resp_no, 'orange', linestyle='--', linewidth=2, label='Without Projection (Soft Constraint)')
    
    plt.axvline(x=r_tgt / 1000, color='green', linestyle=':', linewidth=2, label='Target')
    plt.axvline(x=r_jam / 1000, color='red', linestyle=':', linewidth=2, label='Jammer')
    
    plt.title("Ablation Study: Impact of MVDR Projection Layer", fontsize=14)
    plt.xlabel("Range (km)", fontsize=12)
    plt.ylabel("Response (dB)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig("exp_ablation_projection.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n实验完成！结果已保存至 'exp_ablation_projection.png'")


if __name__ == "__main__":
    run_ablation_experiment()
