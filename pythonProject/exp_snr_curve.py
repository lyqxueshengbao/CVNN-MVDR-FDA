import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def calculate_output_sinr(w, X, v_s, v_j, sig_power, jam_power, noise_power):
    """计算波束成形后的输出 SINR"""
    # 目标增益
    gain_s = torch.abs(torch.vdot(w, v_s)) ** 2
    
    # 干扰增益
    gain_j = torch.abs(torch.vdot(w, v_j)) ** 2
    
    # 噪声功率
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


def run_snr_curve_experiment():
    """实验 4: 不同 SNR 下的输出 SINR 性能"""
    print("=" * 60)
    print("实验 4: SNR vs 输出 SINR (弱目标检测能力)")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    # 加载模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model.eval()
    
    # 测试场景
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    r_jam = 12000.0
    
    JNR_test = 45  # 固定强干扰
    SNR_range = np.arange(-10, 21, 2)  # -10 到 20 dB
    
    L = cfg.L
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    a_tgt_test = v_s.unsqueeze(0)
    
    output_sinr_dcvb = []
    output_sinr_mvdr = []
    input_sinr = []  # 输入 SINR (作为 baseline)
    
    for SNR in SNR_range:
        print(f"Testing SNR = {SNR} dB...")
        
        # 生成数据
        sig_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        jam_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)
        
        sig_power_linear = 10 ** (SNR / 10)
        jam_power_linear = 10 ** (JNR_test / 10)
        
        signal = np.sqrt(sig_power_linear) * v_s.unsqueeze(1) * sig_wave
        jamming = np.sqrt(jam_power_linear) * v_j.unsqueeze(1) * jam_wave
        
        X_input = signal + jamming + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        
        # 计算输入 SINR
        input_sinr.append(SNR - JNR_test)  # 近似值
        
        # DCVB 方法
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        
        # MVDR 方法
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        # 计算输出 SINR
        sinr_dcvb = calculate_output_sinr(w_dcvb, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        sinr_mvdr = calculate_output_sinr(w_mvdr, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        
        output_sinr_dcvb.append(sinr_dcvb)
        output_sinr_mvdr.append(sinr_mvdr)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(SNR_range, input_sinr, 'k:', linewidth=2, label='Input SINR (No Processing)')
    plt.plot(SNR_range, output_sinr_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    plt.plot(SNR_range, output_sinr_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB')
    
    plt.xlabel('Input SNR (dB)', fontsize=12)
    plt.ylabel('Output SINR (dB)', fontsize=12)
    plt.title(f'Output SINR vs Input SNR (JNR = {JNR_test} dB)', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加 0 dB 参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('exp_snr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n实验完成！结果已保存至 'exp_snr_curve.png'")
    print(f"SNR=-10dB 时输出 SINR: DCVB={output_sinr_dcvb[0]:.2f} dB, MVDR={output_sinr_mvdr[0]:.2f} dB")
    print(f"SNR=20dB 时输出 SINR: DCVB={output_sinr_dcvb[-1]:.2f} dB, MVDR={output_sinr_mvdr[-1]:.2f} dB")


if __name__ == "__main__":
    run_snr_curve_experiment()
