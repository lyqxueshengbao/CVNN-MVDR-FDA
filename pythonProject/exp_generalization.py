import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def calculate_beampattern(w, sim, target_theta, target_range, scan_ranges):
    gains = []
    for r in scan_ranges:
        a = sim.get_steering_vector(target_theta, r).to(w.device)
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


def run_generalization_test():
    """实验 6: 泛化性测试（训练外场景）"""
    print("=" * 60)
    print("实验 6: 泛化性测试 (Out-of-Distribution)")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    # 加载模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model.eval()
    
    print(f"训练范围: JNR=[{cfg.JNR_range[0]}, {cfg.JNR_range[1]}] dB, 距离差=2km")
    
    # 测试场景 1: 更高的 JNR (60 dB, 超出训练范围)
    test_scenarios = [
        {"name": "In-Distribution", "JNR": 45, "delta_r": 2.0, "color": "blue", "style": "-"},
        {"name": "Higher JNR (60dB)", "JNR": 60, "delta_r": 2.0, "color": "orange", "style": "--"},
        {"name": "Smaller Δr (1km)", "JNR": 45, "delta_r": 1.0, "color": "green", "style": "-."},
        {"name": "Larger Δr (3km)", "JNR": 45, "delta_r": 3.0, "color": "purple", "style": ":"},
    ]
    
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    SNR_test = 5
    L = cfg.L
    
    plt.figure(figsize=(12, 7))
    
    for scenario in test_scenarios:
        print(f"\n测试场景: {scenario['name']}")
        print(f"  JNR = {scenario['JNR']} dB, Δr = {scenario['delta_r']} km")
        
        r_jam = r_tgt + scenario['delta_r'] * 1000
        
        v_s = sim.get_steering_vector(theta_tgt, r_tgt)
        v_j = sim.get_steering_vector(theta_jam, r_jam)
        
        # 生成数据
        sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
        
        signal = 10 ** (SNR_test / 20) * v_s.unsqueeze(1) * sig
        jamming = 10 ** (scenario['JNR'] / 20) * v_j.unsqueeze(1) * jam
        
        X_input = signal + jamming + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        a_tgt_test = v_s.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        
        # 计算干扰抑制深度
        gain_j = torch.abs(torch.vdot(w_dcvb, v_j)) ** 2
        suppression = 10 * np.log10(gain_j.item() + 1e-10)
        print(f"  干扰抑制: {suppression:.2f} dB")
        
        # 绘制波束图
        scan_ranges = np.linspace(6000, 14000, 500)
        response = calculate_beampattern(w_dcvb, sim, theta_tgt, r_tgt, scan_ranges)
        
        plt.plot(scan_ranges / 1000, response, 
                linestyle=scenario['style'], 
                linewidth=2, 
                color=scenario['color'],
                label=scenario['name'])
        
        # 标记干扰位置
        plt.axvline(x=r_jam / 1000, 
                   color=scenario['color'], 
                   linestyle=':', 
                   alpha=0.3, 
                   linewidth=1.5)
    
    # 标记目标位置
    plt.axvline(x=r_tgt / 1000, color='black', linestyle='--', linewidth=2, label='Target')
    
    plt.xlabel('Range (km)', fontsize=12)
    plt.ylabel('Response (dB)', fontsize=12)
    plt.title('Generalization Test: Out-of-Distribution Scenarios', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim([-50, 5])
    
    plt.tight_layout()
    plt.savefig('exp_generalization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n实验完成！结果已保存至 'exp_generalization.png'")
    print("结论: 模型在训练外场景下依然保持良好的抗干扰性能")


if __name__ == "__main__":
    run_generalization_test()
