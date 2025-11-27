import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
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


def run_range_difference_experiment():
    """实验 3: 距离差对抑制性能的影响"""
    print("=" * 60)
    print("实验 3: 距离差 vs 干扰抑制深度 (FDA 特性验证)")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    # 加载模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model.eval()
    
    # 固定目标位置
    theta_tgt = 10.0
    r_tgt = 10000.0
    
    # 固定干扰角度（主瓣干扰）
    theta_jam = 10.0
    
    # 测试不同的距离差
    range_diff = np.arange(0.5, 5.1, 0.5)  # 0.5km 到 5km
    
    JNR_test = 50  # 固定 50dB 强干扰
    SNR_test = 5
    L = cfg.L
    
    suppression_dcvb = []
    suppression_mvdr = []
    
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    a_tgt_test = v_s.unsqueeze(0)
    
    for delta_r in range_diff:
        r_jam = r_tgt + delta_r * 1000  # 转换为米
        print(f"Testing Δr = {delta_r:.1f} km (Jammer at {r_jam/1000:.1f} km)...")
        
        v_j = sim.get_steering_vector(theta_jam, r_jam)
        
        # 生成数据
        sig_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        jam_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)
        
        signal = 10 ** (SNR_test / 20) * v_s.unsqueeze(1) * sig_wave
        jamming = 10 ** (JNR_test / 20) * v_j.unsqueeze(1) * jam_wave
        
        X_input = signal + jamming + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        
        # DCVB 方法
        with torch.no_grad():
            w_dcvb = model(X_input, a_tgt_test).squeeze(0)
        
        # MVDR 方法
        w_mvdr = get_mvdr_weights(X_input, a_tgt_test)
        
        # 计算干扰抑制深度
        gain_j_dcvb = torch.abs(torch.vdot(w_dcvb, v_j)) ** 2
        gain_j_mvdr = torch.abs(torch.vdot(w_mvdr, v_j)) ** 2
        
        suppression_dcvb.append(10 * np.log10(gain_j_dcvb.item() + 1e-10))
        suppression_mvdr.append(10 * np.log10(gain_j_mvdr.item() + 1e-10))
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(range_diff, suppression_mvdr, 'r--o', linewidth=2, markersize=7, label='Traditional MVDR')
    plt.plot(range_diff, suppression_dcvb, 'b-s', linewidth=2, markersize=7, label='Proposed DCVB')
    
    plt.xlabel('Target-Jammer Range Difference Δr (km)', fontsize=12)
    plt.ylabel('Jamming Suppression (dB)', fontsize=12)
    plt.title('Impact of Range Separation on Jamming Suppression\n(FDA-MIMO Range Resolution)', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加注释
    plt.axhline(y=-30, color='gray', linestyle=':', alpha=0.5)
    plt.text(range_diff[-1] * 0.7, -28, '-30 dB Threshold', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig('exp_range_difference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n实验完成！结果已保存至 'exp_range_difference.png'")
    print(f"最小距离差 (0.5km) 抑制: DCVB={suppression_dcvb[0]:.2f} dB, MVDR={suppression_mvdr[0]:.2f} dB")
    print(f"最大距离差 (5.0km) 抑制: DCVB={suppression_dcvb[-1]:.2f} dB, MVDR={suppression_mvdr[-1]:.2f} dB")


if __name__ == "__main__":
    run_range_difference_experiment()
