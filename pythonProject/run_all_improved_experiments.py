"""
使用改进版模型运行所有核心对比实验
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen_v2 import FdaMimoSimulatorV2
from model import ComplexBeamformerNet
import os


def calculate_output_sinr(w, X, v_s, v_j, sig_power, jam_power, noise_power):
    """计算波束成形后的输出 SINR"""
    gain_s = torch.abs(torch.vdot(w, v_s)) ** 2
    gain_j = torch.abs(torch.vdot(w, v_j)) ** 2
    noise_out = torch.sum(torch.abs(w) ** 2)
    
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


def calculate_beampattern(w, sim, target_theta, target_range, scan_ranges):
    gains = []
    for r in scan_ranges:
        a = sim.get_steering_vector(target_theta, r).to(w.device)
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


# ============================================================================
# 实验 1: JNR 曲线
# ============================================================================
def run_jnr_curve():
    print("=" * 70)
    print("实验 1: JNR vs 干扰抑制深度/输出SINR（改进版）")
    print("=" * 70)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_improved.pth", weights_only=True))
    model.eval()
    
    theta_tgt = 10.0; r_tgt = 10000.0
    theta_jam = 10.0; r_jam = 12000.0
    SNR_test = 5
    JNR_range = np.arange(20, 61, 5)
    
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
        
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        gain_j_dcvb = torch.abs(torch.vdot(w_dcvb, v_j)) ** 2
        gain_j_mvdr = torch.abs(torch.vdot(w_mvdr, v_j)) ** 2
        
        suppression_dcvb.append(10 * np.log10(gain_j_dcvb.item() + 1e-10))
        suppression_mvdr.append(10 * np.log10(gain_j_mvdr.item() + 1e-10))
        
        sinr_dcvb = calculate_output_sinr(w_dcvb, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        sinr_mvdr = calculate_output_sinr(w_mvdr, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        
        output_sinr_dcvb.append(sinr_dcvb)
        output_sinr_mvdr.append(sinr_mvdr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(JNR_range, suppression_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    ax1.plot(JNR_range, suppression_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB (Improved)')
    ax1.set_xlabel('JNR (dB)', fontsize=12)
    ax1.set_ylabel('Jamming Suppression (dB)', fontsize=12)
    ax1.set_title('Jamming Suppression vs JNR', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    ax2.plot(JNR_range, output_sinr_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    ax2.plot(JNR_range, output_sinr_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB (Improved)')
    ax2.set_xlabel('JNR (dB)', fontsize=12)
    ax2.set_ylabel('Output SINR (dB)', fontsize=12)
    ax2.set_title('Output SINR vs JNR', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('exp_improved_jnr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 完成！平均干扰抑制: DCVB={np.mean(suppression_dcvb):.2f} dB\n")


# ============================================================================
# 实验 2: SNR 曲线
# ============================================================================
def run_snr_curve():
    print("=" * 70)
    print("实验 2: SNR vs 输出SINR（改进版）")
    print("=" * 70)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_improved.pth", weights_only=True))
    model.eval()
    
    theta_tgt = 10.0; r_tgt = 10000.0
    theta_jam = 10.0; r_jam = 12000.0
    JNR_test = 45
    SNR_range = np.arange(-10, 21, 2)
    
    output_sinr_dcvb = []
    output_sinr_mvdr = []
    input_sinr = []
    
    L = cfg.L
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    a_tgt_test = v_s.unsqueeze(0)
    
    for SNR in SNR_range:
        print(f"Testing SNR = {SNR} dB...")
        
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
        
        input_sinr.append(SNR - JNR_test)
        
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        sinr_dcvb = calculate_output_sinr(w_dcvb, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        sinr_mvdr = calculate_output_sinr(w_mvdr, X_input, v_s, v_j, sig_power_linear, jam_power_linear, 1.0)
        
        output_sinr_dcvb.append(sinr_dcvb)
        output_sinr_mvdr.append(sinr_mvdr)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(SNR_range, input_sinr, 'k:', linewidth=2, label='Input SINR (No Processing)')
    plt.plot(SNR_range, output_sinr_mvdr, 'r--o', linewidth=2, markersize=6, label='Traditional MVDR')
    plt.plot(SNR_range, output_sinr_dcvb, 'b-s', linewidth=2, markersize=6, label='Proposed DCVB (Improved)')
    
    plt.xlabel('Input SNR (dB)', fontsize=12)
    plt.ylabel('Output SINR (dB)', fontsize=12)
    plt.title(f'Output SINR vs Input SNR (JNR = {JNR_test} dB)', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('exp_improved_snr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 完成！\n")


# ============================================================================
# 实验 3: 波束图对比
# ============================================================================
def run_beampattern_comparison():
    print("=" * 70)
    print("实验 3: 波束图对比（改进版 vs MVDR）")
    print("=" * 70)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_improved.pth", weights_only=True))
    model.eval()
    
    theta_tgt = 10.0; r_tgt = 10000.0
    theta_jam = 10.0; r_jam = 12000.0
    
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    L = cfg.L
    
    sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
    
    X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise
    X_input = X_input / torch.max(torch.abs(X_input))
    X_input = X_input.unsqueeze(0)
    a_tgt_test = v_s.unsqueeze(0)
    
    with torch.no_grad():
        w_dcvb = model(X_input, a_tgt_test).squeeze(0)
    w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
    
    scan_ranges = np.linspace(6000, 14000, 500)
    resp_dcvb = calculate_beampattern(w_dcvb, sim, theta_tgt, r_tgt, scan_ranges)
    resp_mvdr = calculate_beampattern(w_mvdr, sim, theta_tgt, r_tgt, scan_ranges)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(scan_ranges / 1000, resp_mvdr, 'r--', linewidth=2, label='Traditional MVDR')
    plt.plot(scan_ranges / 1000, resp_dcvb, 'b-', linewidth=2.5, alpha=0.8, label='Proposed DCVB (Improved)')
    
    plt.axvline(x=r_tgt / 1000, color='green', linestyle=':', linewidth=2, label='Target')
    plt.axvline(x=r_jam / 1000, color='orange', linestyle=':', linewidth=2, label='Jammer')
    
    plt.title("Performance Comparison: Improved DCVB vs MVDR", fontsize=14)
    plt.xlabel("Range (km)", fontsize=12)
    plt.ylabel("Response (dB)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("exp_improved_beampattern.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 完成！\n")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("改进版模型 - 完整实验套件")
    print("=" * 70 + "\n")
    
    if not os.path.exists("fda_improved.pth"):
        print("❌ 错误: 未找到改进模型 'fda_improved.pth'")
        print("请先运行: python train_improved.py")
        return
    
    experiments = [
        ("JNR 曲线", run_jnr_curve),
        ("SNR 曲线", run_snr_curve),
        ("波束图对比", run_beampattern_comparison),
    ]
    
    for name, func in experiments:
        try:
            func()
        except Exception as e:
            print(f"❌ {name} 失败: {e}\n")
            continue
    
    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)
    
    print("\n生成的图表文件:")
    result_files = [
        "exp_improved_jnr_curve.png",
        "exp_improved_snr_curve.png",
        "exp_improved_beampattern.png",
        "exp_improved_range_difference.png",  # 之前已生成
        "exp_improved_generalization.png",    # 之前已生成
    ]
    
    for f in result_files:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ⚠️  {f} (未找到)")
    
    print("\n所有实验结果可直接用于论文！")


if __name__ == "__main__":
    main()
