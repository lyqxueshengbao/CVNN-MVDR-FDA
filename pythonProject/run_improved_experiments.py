"""
改进版实验套件：使用改进模型重新运行所有对比实验
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen_v2 import FdaMimoSimulatorV2
from model import ComplexBeamformerNet


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


def run_improved_range_difference_test():
    """使用改进模型重新测试距离差影响"""
    print("=" * 60)
    print("改进模型测试：距离差 vs 抑制性能")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    # 加载原版模型和改进模型
    model_original = ComplexBeamformerNet(cfg).to(cfg.device)
    model_original.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model_original.eval()
    
    model_improved = ComplexBeamformerNet(cfg).to(cfg.device)
    model_improved.load_state_dict(torch.load("fda_improved.pth"))
    model_improved.eval()
    
    # 测试场景
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    
    range_diff = np.arange(0.5, 5.1, 0.5)
    
    suppression_original = []
    suppression_improved = []
    suppression_mvdr = []
    
    L = cfg.L
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    a_tgt_test = v_s.unsqueeze(0)
    
    for delta_r in range_diff:
        r_jam = r_tgt + delta_r * 1000
        print(f"Testing Δr = {delta_r:.1f} km...")
        
        v_j = sim.get_steering_vector(theta_jam, r_jam)
        
        # 生成数据
        sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
        
        X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        
        # 三个方法
        with torch.no_grad():
            w_original = model_original(X_input, a_tgt_test).squeeze(0)
            w_improved = model_improved(X_input, a_tgt_test).squeeze(0)
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        # 计算抑制
        gain_original = torch.abs(torch.vdot(w_original, v_j)) ** 2
        gain_improved = torch.abs(torch.vdot(w_improved, v_j)) ** 2
        gain_mvdr = torch.abs(torch.vdot(w_mvdr, v_j)) ** 2
        
        suppression_original.append(10 * np.log10(gain_original.item() + 1e-10))
        suppression_improved.append(10 * np.log10(gain_improved.item() + 1e-10))
        suppression_mvdr.append(10 * np.log10(gain_mvdr.item() + 1e-10))
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    plt.plot(range_diff, suppression_mvdr, 'r--o', linewidth=2, markersize=7, 
             label='Traditional MVDR (Baseline)', alpha=0.8)
    plt.plot(range_diff, suppression_original, 'b:', linewidth=2.5, marker='x', markersize=8,
             label='DCVB Original (Fixed Δr=2km)', alpha=0.7)
    plt.plot(range_diff, suppression_improved, 'g-s', linewidth=2.5, markersize=7, 
             label='DCVB Improved (Random Δr=1-3km)')
    
    # 标记训练区域
    plt.axvspan(1.0, 3.0, alpha=0.08, color='green')
    plt.text(2.0, -5, 'Training\nRange', fontsize=10, ha='center', color='darkgreen')
    
    plt.xlabel('Target-Jammer Range Difference Δr (km)', fontsize=12)
    plt.ylabel('Jamming Suppression (dB)', fontsize=12)
    plt.title('Generalization Comparison: Original vs Improved Training', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('exp_improved_range_difference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n结果已保存: exp_improved_range_difference.png")
    
    # 统计分析
    var_original = np.var(suppression_original)
    var_improved = np.var(suppression_improved)
    mean_original = np.mean(suppression_original)
    mean_improved = np.mean(suppression_improved)
    
    print(f"\n性能统计:")
    print(f"原版模型 - 平均抑制: {mean_original:.2f} dB, 方差: {var_original:.2f}")
    print(f"改进模型 - 平均抑制: {mean_improved:.2f} dB, 方差: {var_improved:.2f}")
    print(f"方差改善: {(1 - var_improved/var_original)*100:.1f}%")


def run_improved_generalization_test():
    """改进模型的泛化性测试"""
    print("\n" + "=" * 60)
    print("改进模型：泛化性测试")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_improved.pth"))
    model.eval()
    
    test_scenarios = [
        {"name": "In-Distribution (Δr=2km)", "JNR": 45, "delta_r": 2.0, "color": "blue", "style": "-"},
        {"name": "Boundary Case (Δr=1km)", "JNR": 45, "delta_r": 1.0, "color": "green", "style": "-"},
        {"name": "Boundary Case (Δr=3km)", "JNR": 45, "delta_r": 3.0, "color": "purple", "style": "-"},
        {"name": "Extrapolation (Δr=4km)", "JNR": 45, "delta_r": 4.0, "color": "orange", "style": "--"},
    ]
    
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    SNR_test = 5
    L = cfg.L
    
    plt.figure(figsize=(12, 7))
    
    for scenario in test_scenarios:
        print(f"\n测试: {scenario['name']}")
        
        r_jam = r_tgt + scenario['delta_r'] * 1000
        v_s = sim.get_steering_vector(theta_tgt, r_tgt)
        v_j = sim.get_steering_vector(theta_jam, r_jam)
        
        sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
        
        signal = 10 ** (SNR_test / 20) * v_s.unsqueeze(1) * sig
        jamming = 10 ** (scenario['JNR'] / 20) * v_j.unsqueeze(1) * jam
        
        X_input = signal + jamming + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        a_tgt_test = v_s.unsqueeze(0)
        
        with torch.no_grad():
            w = model(X_input, a_tgt_test).squeeze(0)
        
        gain_j = torch.abs(torch.vdot(w, v_j)) ** 2
        suppression = 10 * np.log10(gain_j.item() + 1e-10)
        print(f"  抑制深度: {suppression:.2f} dB")
        
        scan_ranges = np.linspace(6000, 14000, 500)
        response = calculate_beampattern(w, sim, theta_tgt, r_tgt, scan_ranges)
        
        plt.plot(scan_ranges / 1000, response, 
                linestyle=scenario['style'], 
                linewidth=2.5 if scenario['delta_r'] <= 3 else 2,
                color=scenario['color'],
                label=scenario['name'],
                alpha=0.9 if scenario['delta_r'] <= 3 else 0.6)
        
        plt.axvline(x=r_jam / 1000, color=scenario['color'], linestyle=':', alpha=0.3, linewidth=1.5)
    
    plt.axvline(x=r_tgt / 1000, color='black', linestyle='--', linewidth=2, label='Target')
    
    plt.xlabel('Range (km)', fontsize=12)
    plt.ylabel('Response (dB)', fontsize=12)
    plt.title('Improved Model: Generalization Performance', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim([-50, 5])
    
    plt.tight_layout()
    plt.savefig('exp_improved_generalization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n结果已保存: exp_improved_generalization.png")


if __name__ == "__main__":
    import os
    
    if not os.path.exists("fda_improved.pth"):
        print("❌ 错误: 未找到改进模型 'fda_improved.pth'")
        print("请先运行: python train_improved.py")
    else:
        run_improved_range_difference_test()
        run_improved_generalization_test()
        print("\n✅ 所有改进实验完成！")
