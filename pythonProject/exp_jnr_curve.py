import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def calculate_output_sinr(w, X, v_s, v_j, sig_power, jam_power, noise_power):
    """
    计算波束成形后的输出 SINR
    w: 权值向量 (MN,)
    X: 输入信号 (1, MN, L)
    """
    w_H = w.conj().unsqueeze(0).unsqueeze(-1)  # (1, MN, 1)
    
    # 输出信号
    y = torch.matmul(w_H.transpose(1, 2), X).squeeze(1)  # (1, L)
    
    # 目标增益
    gain_s = torch.abs(torch.vdot(w, v_s)) ** 2
    
    # 干扰增益
    gain_j = torch.abs(torch.vdot(w, v_j)) ** 2
    
    # 噪声功率 (假设白噪声)
    noise_out = torch.sum(torch.abs(w) ** 2)
    
    # SINR
    signal_power_out = gain_s * sig_power
    jam_power_out = gain_j * jam_power
    noise_power_out = noise_out * noise_power
    
    sinr = signal_power_out / (jam_power_out + noise_power_out + 1e-10)
    return 10 * torch.log10(sinr + 1e-10).item()


def get_mvdr_weights(X_input, a_tgt):
    """传统 MVDR 算法"""
    X = X_input.squeeze(0)
    L = X.shape[1]
    
    R = torch.matmul(X, X.conj().T) / L
    R = R + 1e-6 * torch.eye(R.shape[0], device=R.device)
    R_inv = torch.linalg.inv(R)
    
    a = a_tgt.T
    numerator = torch.matmul(R_inv, a)
    denominator = torch.matmul(a.conj().T, numerator)
    w = numerator / (denominator + 1e-10)
    return w.squeeze()


def run_jnr_curve_experiment():
    """实验 1: 不同 JNR 下的抗干扰性能曲线"""
    print("=" * 60)
    print("实验 1: JNR vs 干扰抑制深度")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    # 加载训练好的模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model.eval()
    
    # 测试场景
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0  # 主瓣干扰
    r_jam = 12000.0
    
    SNR_test = 5  # 固定 SNR
    JNR_range = np.arange(20, 61, 5)  # 20 到 60 dB
    
    # 存储结果
    suppression_dcvb = []
    suppression_mvdr = []
    output_sinr_dcvb = []
    output_sinr_mvdr = []
    
    L = cfg.L
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    a_tgt_test = v_s.unsqueeze(0)
    
    for JNR in JNR_range:
        print(f"Testing JNR = {JNR} dB...")
        
        # 生成数据
        sig_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        jam_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)
        
        sig_power_linear = 10 ** (SNR_test / 10)
        jam_power_linear = 10 ** (JNR / 10)
        
        signal = np.sqrt(sig_power_linear) * v_s.unsqueeze(1) * sig_wave
        jamming = np.sqrt(jam_power_linear) * v_j.unsqueeze(1) * jam_wave
        
        X_input = signal + jamming + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        
        # DCVB 方法
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        
        # MVDR 方法
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        # 计算干扰抑制深度 (在干扰位置的增益)
        gain_j_dcvb = torch.abs(torch.vdot(w_dcvb, v_j)) ** 2
        gain_j_mvdr = torch.abs(torch.vdot(w_mvdr, v_j)) ** 2
        
        suppression_dcvb.append(10 * np.log10(gain_j_dcvb.item() + 1e-10))
        suppression_mvdr.append(10 * np.log10(gain_j_mvdr.item() + 1e-10))
        
        # 计算输出 SINR
        sinr_dcvb = calculate_output_sinr(w_dcvb, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        sinr_mvdr = calculate_output_sinr(w_mvdr, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        
        output_sinr_dcvb.append(sinr_dcvb)
        output_sinr_mvdr.append(sinr_mvdr)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图 1: 干扰抑制深度
    ax1.plot(JNR_range, suppression_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    ax1.plot(JNR_range, suppression_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB')
    ax1.set_xlabel('JNR (dB)', fontsize=12)
    ax1.set_ylabel('Jamming Suppression (dB)', fontsize=12)
    ax1.set_title('Jamming Suppression vs JNR', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 子图 2: 输出 SINR
    ax2.plot(JNR_range, output_sinr_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    ax2.plot(JNR_range, output_sinr_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB')
    ax2.set_xlabel('JNR (dB)', fontsize=12)
    ax2.set_ylabel('Output SINR (dB)', fontsize=12)
    ax2.set_title('Output SINR vs JNR', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('exp_jnr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n实验完成！结果已保存至 'exp_jnr_curve.png'")
    print(f"平均干扰抑制 (DCVB): {np.mean(suppression_dcvb):.2f} dB")
    print(f"平均干扰抑制 (MVDR): {np.mean(suppression_mvdr):.2f} dB")


if __name__ == "__main__":
    run_jnr_curve_experiment()
