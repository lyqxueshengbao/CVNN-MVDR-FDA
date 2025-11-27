"""
综合鲁棒性测试：少快拍 + 指向误差 双重挑战
(Combined Robustness Test: Low Snapshots + Pointing Error)

核心逻辑：
- 实战中，两种问题经常同时出现
- MVDR 在单一条件下可能还能撑住，但双重恶劣条件下会崩溃
- DCVB 的鲁棒性在恶劣组合下更加突出

这是真正的"实战场景"！
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def get_mvdr_weights(X_input, a_tgt):
    X = X_input.squeeze(0)
    MN, L = X.shape
    R = torch.matmul(X, X.conj().T) / L
    R = R + 1e-6 * torch.eye(MN, device=R.device, dtype=R.dtype)
    try:
        R_inv = torch.linalg.inv(R)
    except:
        R_inv = torch.linalg.pinv(R)
    a = a_tgt.T
    numerator = torch.matmul(R_inv, a)
    denominator = torch.matmul(a.conj().T, numerator)
    return (numerator / (denominator + 1e-10)).squeeze()


def calculate_target_gain(w, v_s):
    """目标方向增益 (希望保持在 0dB 附近)"""
    gain = torch.abs(torch.vdot(w, v_s))**2
    return 10 * np.log10(gain.item() + 1e-10)


def run_combined_test():
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth", 
                                      map_location=cfg.device, weights_only=True))
    model.eval()
    
    print("=" * 70)
    print("Combined Robustness Test: Low Snapshots + Pointing Error")
    print("测试 少快拍 + 指向误差 双重恶劣条件")
    print("=" * 70)
    
    # 真实目标位置
    true_theta, r_tgt = 10.0, 10000.0
    theta_jam, r_jam = 10.0, 12000.0
    
    v_s_true = sim.get_steering_vector(true_theta, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    
    amp_signal = 1.0
    amp_jammer = 316.0  # 50dB
    
    # 测试条件
    snapshots_list = [5, 10, 20, 50, 100]
    errors_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    num_trials = 30
    
    # 结果存储: [snapshots][errors] -> (mean, std)
    dcvb_results = np.zeros((len(snapshots_list), len(errors_list)))
    mvdr_results = np.zeros((len(snapshots_list), len(errors_list)))
    
    for i, L in enumerate(snapshots_list):
        for j, err in enumerate(errors_list):
            print(f"\nTesting L={L}, Error={err}°...")
            
            # 假设的目标位置 (带误差)
            assumed_theta = true_theta + err
            v_assumed = sim.get_steering_vector(assumed_theta, r_tgt)
            a_wrong = v_assumed.unsqueeze(0)
            
            dcvb_trials = []
            mvdr_trials = []
            
            for _ in range(num_trials):
                sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
                jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
                noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)
                
                X_raw = amp_signal * v_s_true.unsqueeze(1) * sig + \
                        amp_jammer * v_j.unsqueeze(1) * jam + noise
                X_input = (X_raw / torch.max(torch.abs(X_raw))).unsqueeze(0)
                
                # DCVB
                with torch.no_grad():
                    w_deep = model(X_input, a_wrong).squeeze(0)
                
                # MVDR
                w_mvdr = get_mvdr_weights(X_input, a_wrong)
                
                # 真实目标增益 (关键！用错误导向矢量计算权值，但检查真实目标增益)
                dcvb_trials.append(calculate_target_gain(w_deep, v_s_true))
                mvdr_trials.append(calculate_target_gain(w_mvdr, v_s_true))
            
            dcvb_results[i, j] = np.mean(dcvb_trials)
            mvdr_results[i, j] = np.mean(mvdr_trials)
            
            print(f"  DCVB: {dcvb_results[i,j]:.2f} dB, MVDR: {mvdr_results[i,j]:.2f} dB")
    
    # ==================== 绘图：热力图 ====================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 子图1: MVDR 热力图
    ax1 = axes[0]
    im1 = ax1.imshow(mvdr_results, cmap='RdYlGn', aspect='auto', 
                     vmin=-25, vmax=5)
    ax1.set_xticks(range(len(errors_list)))
    ax1.set_xticklabels([f'{e}°' for e in errors_list])
    ax1.set_yticks(range(len(snapshots_list)))
    ax1.set_yticklabels(snapshots_list)
    ax1.set_xlabel('Pointing Error', fontsize=12)
    ax1.set_ylabel('Number of Snapshots (L)', fontsize=12)
    ax1.set_title('(a) MVDR Target Gain (dB)', fontsize=13)
    plt.colorbar(im1, ax=ax1)
    
    # 添加数值标注
    for i in range(len(snapshots_list)):
        for j in range(len(errors_list)):
            ax1.text(j, i, f'{mvdr_results[i,j]:.1f}', ha='center', va='center', 
                    color='white' if mvdr_results[i,j] < -10 else 'black', fontsize=9)
    
    # 子图2: DCVB 热力图
    ax2 = axes[1]
    im2 = ax2.imshow(dcvb_results, cmap='RdYlGn', aspect='auto', 
                     vmin=-25, vmax=5)
    ax2.set_xticks(range(len(errors_list)))
    ax2.set_xticklabels([f'{e}°' for e in errors_list])
    ax2.set_yticks(range(len(snapshots_list)))
    ax2.set_yticklabels(snapshots_list)
    ax2.set_xlabel('Pointing Error', fontsize=12)
    ax2.set_ylabel('Number of Snapshots (L)', fontsize=12)
    ax2.set_title('(b) DCVB Target Gain (dB)', fontsize=13)
    plt.colorbar(im2, ax=ax2)
    
    for i in range(len(snapshots_list)):
        for j in range(len(errors_list)):
            ax2.text(j, i, f'{dcvb_results[i,j]:.1f}', ha='center', va='center', 
                    color='white' if dcvb_results[i,j] < -10 else 'black', fontsize=9)
    
    # 子图3: DCVB 相对 MVDR 的优势
    advantage = dcvb_results - mvdr_results  # 正数表示 DCVB 更好
    ax3 = axes[2]
    im3 = ax3.imshow(advantage, cmap='coolwarm', aspect='auto', 
                     vmin=-10, vmax=25)
    ax3.set_xticks(range(len(errors_list)))
    ax3.set_xticklabels([f'{e}°' for e in errors_list])
    ax3.set_yticks(range(len(snapshots_list)))
    ax3.set_yticklabels(snapshots_list)
    ax3.set_xlabel('Pointing Error', fontsize=12)
    ax3.set_ylabel('Number of Snapshots (L)', fontsize=12)
    ax3.set_title('(c) DCVB Advantage over MVDR (dB)', fontsize=13)
    plt.colorbar(im3, ax=ax3)
    
    for i in range(len(snapshots_list)):
        for j in range(len(errors_list)):
            ax3.text(j, i, f'{advantage[i,j]:+.1f}', ha='center', va='center', 
                    color='white' if abs(advantage[i,j]) > 10 else 'black', fontsize=9)
    
    plt.suptitle('Combined Robustness Test: Low Snapshots + Pointing Error\n(Target Gain: 0 dB = Ideal, Negative = Signal Cancellation)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('exp_combined_robustness.png', dpi=300, bbox_inches='tight')
    print("\n\n热力图已保存到 exp_combined_robustness.png")
    
    # ==================== 关键发现 ====================
    print("\n" + "=" * 70)
    print("Key Findings for Paper")
    print("=" * 70)
    
    # 找到 DCVB 优势最大的条件
    max_advantage = np.max(advantage)
    max_idx = np.unravel_index(np.argmax(advantage), advantage.shape)
    
    # 找到 MVDR 最差的条件
    min_mvdr = np.min(mvdr_results)
    min_mvdr_idx = np.unravel_index(np.argmin(mvdr_results), mvdr_results.shape)
    
    print(f"""
1. 最大 DCVB 优势：
   - 条件：L={snapshots_list[max_idx[0]]}, Error={errors_list[max_idx[1]]}°
   - DCVB: {dcvb_results[max_idx]:.2f} dB
   - MVDR: {mvdr_results[max_idx]:.2f} dB
   - DCVB 领先: {max_advantage:.2f} dB

2. MVDR 最差表现：
   - 条件：L={snapshots_list[min_mvdr_idx[0]]}, Error={errors_list[min_mvdr_idx[1]]}°
   - MVDR Target Gain: {min_mvdr:.2f} dB (严重自削!)

3. 论文亮点：
   "Under combined adverse conditions (limited snapshots AND pointing 
    error), MVDR suffers severe signal cancellation. At L={snapshots_list[min_mvdr_idx[0]]} 
    snapshots with {errors_list[min_mvdr_idx[1]]}° error, MVDR target gain drops to 
    {min_mvdr:.1f} dB. The proposed DCVB maintains {dcvb_results[min_mvdr_idx]:.1f} dB 
    gain under the same conditions, demonstrating superior practical 
    robustness for real-world radar systems."
""")


if __name__ == "__main__":
    run_combined_test()
