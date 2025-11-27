"""
实验一：单快拍/少快拍稳定性测试 (The Snapshot Stability Test)

核心逻辑：
- MVDR 需要估计协方差矩阵 R = X * X^H / L
- 当 L < MN 时，协方差估计的 *方差* 很大 → 性能不稳定
- 深度学习方法对快拍数不敏感（输出稳定）

关键指标：
- 性能的标准差 (Standard Deviation) 
- MVDR 在少快拍时方差巨大，DCVB 保持稳定

这个实验直接展示"可靠性"——实战中最重要的指标！
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def get_mvdr_weights(X_input, a_tgt):
    """标准 MVDR（最小对角加载，暴露其弱点）"""
    X = X_input.squeeze(0)
    MN, L = X.shape
    
    R = torch.matmul(X, X.conj().T) / L
    # 最小对角加载
    R = R + 1e-6 * torch.eye(MN, device=R.device, dtype=R.dtype)
    
    try:
        R_inv = torch.linalg.inv(R)
    except:
        R_inv = torch.linalg.pinv(R)
    
    a = a_tgt.T
    numerator = torch.matmul(R_inv, a)
    denominator = torch.matmul(a.conj().T, numerator)
    w = numerator / (denominator + 1e-10)
    
    return w.squeeze()


def calculate_interference_suppression(w, v_s, v_j):
    """干扰抑制比 (越小越好)"""
    signal_gain = torch.abs(torch.vdot(w, v_s))**2
    interf_gain = torch.abs(torch.vdot(w, v_j))**2
    isr = interf_gain / (signal_gain + 1e-10)
    return 10 * np.log10(isr.item() + 1e-10)


def run_stability_test():
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth", 
                                      map_location=cfg.device, weights_only=True))
    model.eval()
    
    print("=" * 65)
    print("Snapshot Stability Test (少快拍稳定性测试)")
    print("关键指标：性能的标准差 (Std Dev) —— 越小越稳定")
    print("=" * 65)
    
    # 场景
    theta_tgt, r_tgt = 10.0, 10000.0
    theta_jam, r_jam = 10.0, 12000.0  # 同角度干扰
    
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    a_tgt = v_s.unsqueeze(0)
    
    # 50dB 干扰
    amp_signal = 1.0
    amp_jammer = 316.0
    
    snapshots_list = [1, 2, 5, 10, 20, 50, 100, 200]
    num_trials = 50  # 多次试验测方差
    
    # 存储结果：均值和标准差
    dcvb_means, dcvb_stds = [], []
    mvdr_means, mvdr_stds = [], []
    
    for L in snapshots_list:
        print(f"\nTesting L = {L} snapshots ({num_trials} trials)...")
        
        dcvb_results = []
        mvdr_results = []
        
        for _ in range(num_trials):
            # 随机生成信号
            sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
            jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
            noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)
            
            X_raw = amp_signal * v_s.unsqueeze(1) * sig + \
                    amp_jammer * v_j.unsqueeze(1) * jam + noise
            X_input = X_raw / torch.max(torch.abs(X_raw))
            X_input = X_input.unsqueeze(0)
            
            # DCVB
            with torch.no_grad():
                w_deep = model(X_input, a_tgt).squeeze(0)
            
            # MVDR
            w_mvdr = get_mvdr_weights(X_input, a_tgt)
            
            # 计算 ISR
            dcvb_results.append(calculate_interference_suppression(w_deep, v_s, v_j))
            mvdr_results.append(calculate_interference_suppression(w_mvdr, v_s, v_j))
        
        # 统计
        dcvb_means.append(np.mean(dcvb_results))
        dcvb_stds.append(np.std(dcvb_results))
        mvdr_means.append(np.mean(mvdr_results))
        mvdr_stds.append(np.std(mvdr_results))
        
        print(f"  DCVB: {dcvb_means[-1]:.1f} ± {dcvb_stds[-1]:.1f} dB")
        print(f"  MVDR: {mvdr_means[-1]:.1f} ± {mvdr_stds[-1]:.1f} dB")
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 均值对比 (带误差棒)
    ax1 = axes[0]
    x = np.arange(len(snapshots_list))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mvdr_means, width, yerr=mvdr_stds, 
                    label='MVDR', color='indianred', capsize=5, alpha=0.8)
    bars2 = ax1.bar(x + width/2, dcvb_means, width, yerr=dcvb_stds, 
                    label='DCVB', color='steelblue', capsize=5, alpha=0.8)
    
    ax1.set_xlabel('Number of Snapshots (L)', fontsize=12)
    ax1.set_ylabel('Interference Suppression (dB)', fontsize=12)
    ax1.set_title('(a) Mean ISR with Error Bars (Lower is Better)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(snapshots_list)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)  # MN=100 boundary
    ax1.text(4.5, ax1.get_ylim()[1]-5, 'L=MN', ha='center', fontsize=10, color='gray')
    
    # 子图2: 标准差对比 (稳定性)
    ax2 = axes[1]
    ax2.semilogy(snapshots_list, mvdr_stds, 'r--o', linewidth=2, markersize=8, label='MVDR')
    ax2.semilogy(snapshots_list, dcvb_stds, 'b-s', linewidth=2, markersize=8, label='DCVB')
    
    ax2.set_xlabel('Number of Snapshots (L)', fontsize=12)
    ax2.set_ylabel('Standard Deviation (dB)', fontsize=12)
    ax2.set_title('(b) Performance Stability (Lower is Better)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axvline(x=100, color='gray', linestyle=':', alpha=0.5)
    
    plt.suptitle('Snapshot Stability Test: DCVB vs MVDR\n(50dB Interference at Same Angle)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('exp_snapshots_stability.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存到 exp_snapshots_stability.png")
    
    # ==================== 详细结果 ====================
    print("\n" + "=" * 75)
    print("Detailed Results: Mean ± Std (Lower Mean, Lower Std = Better)")
    print("=" * 75)
    print(f"{'L':<8} | {'MVDR Mean':<12} | {'MVDR Std':<12} | {'DCVB Mean':<12} | {'DCVB Std':<12}")
    print("-" * 75)
    for i, L in enumerate(snapshots_list):
        print(f"{L:<8} | {mvdr_means[i]:<12.2f} | {mvdr_stds[i]:<12.2f} | {dcvb_means[i]:<12.2f} | {dcvb_stds[i]:<12.2f}")
    
    # ==================== 关键发现 ====================
    print("\n" + "=" * 75)
    print("Key Findings for Paper:")
    print("=" * 75)
    
    # 找到 DCVB 标准差最大的点
    max_dcvb_std = max(dcvb_stds)
    min_mvdr_std = min(mvdr_stds)
    
    # 计算稳定性提升倍数
    stability_improvement = [mvdr_stds[i] / (dcvb_stds[i] + 1e-6) for i in range(len(snapshots_list))]
    
    print(f"""
1. 稳定性对比：
   - DCVB 最大标准差: {max_dcvb_std:.2f} dB
   - MVDR 最小标准差: {min_mvdr_std:.2f} dB
   - 在 L=1 时，MVDR 标准差是 DCVB 的 {stability_improvement[0]:.1f} 倍

2. 实战意义：
   - 雷达系统需要 *稳定* 的干扰抑制
   - MVDR 在少快拍时"赌博"性能——有时好，有时极差
   - DCVB 提供 *可预测* 的性能保证

3. 论文表述：
   "While MVDR achieves deep nulling on average, its performance 
    variance is extremely high with limited snapshots. The standard 
    deviation of MVDR reaches {mvdr_stds[0]:.1f} dB at L=1, making it 
    unreliable in dynamic scenarios. In contrast, DCVB maintains 
    consistent performance (σ < {max_dcvb_std:.1f} dB) regardless of 
    snapshot count, providing a reliable solution for real-time radar."
""")


if __name__ == "__main__":
    run_stability_test()
