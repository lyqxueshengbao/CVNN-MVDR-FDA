"""
改进版混合预热启动实验：DCVB初始化 + 快速微调
验证深度网络权值作为传统算法初始值的有效性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen_v2 import FdaMimoSimulatorV2
from model import ComplexBeamformerNet
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def apply_mvdr_constraint(w, a_target):
    """单步MVDR约束投影（快速微调）"""
    inner_prod = torch.sum(w.conj() * a_target, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a_target.conj() * a_target, dim=-1, keepdim=True)
    correction = a_target * (inner_prod.conj() - 1.0) / (norm_a_sq + 1e-8)
    return w - correction


def power_iteration_refine(R, a_target, w_init, n_iters=5):
    """
    从 w_init 开始，使用共轭梯度法快速微调
    每次迭代：w ← w - α*(R*w), 然后投影到约束流形
    """
    w = w_init.clone()
    power_history = []
    
    for it in range(n_iters):
        # 计算当前功率
        P = torch.real(w.conj() @ R @ w).item()
        power_history.append(P)
        
        # 梯度下降：最小化 w^H * R * w
        grad = R @ w
        
        # 自适应步长（基于当前功率）
        alpha = 0.5 / (torch.norm(grad).item() + 1e-8)
        
        # 更新权值
        w_new = w - alpha * grad
        
        # 投影到 MVDR 约束流形：w^H * a = 1
        w = apply_mvdr_constraint(w_new, a_target)
    
    # 最后计算一次功率
    P_final = torch.real(w.conj() @ R @ w).item()
    power_history.append(P_final)
    
    return w, np.array(power_history)


def compare_hybrid_methods():
    """
    对比四种方法：
    1. 纯 DCVB（快速但精度有限）
    2. DCVB + 3次迭代微调（混合方法）
    3. DCVB + 10次迭代微调（混合方法，更精细）
    4. 直接 MVDR（最优但慢）
    """
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载 DCVB 模型
    model = ComplexBeamformerNet(cfg=cfg).to(device)
    model.load_state_dict(torch.load('fda_improved.pth', map_location=device, weights_only=False))
    model.eval()
    
    # 预热CUDA（关键！）
    print("=== CUDA 预热 ===")
    simulator = FdaMimoSimulatorV2(cfg)
    for _ in range(3):
        X_warm, a_warm = simulator.generate_batch(range_diff_mode='fixed')
        with torch.no_grad():
            _ = model(X_warm, a_warm)
    print("预热完成\n")
    
    # 生成100个测试样本，取平均
    num_tests = 50
    times_dcvb = []
    times_hybrid_3 = []
    times_hybrid_10 = []
    times_mvdr = []
    
    perfs_dcvb = []
    perfs_hybrid_3 = []
    perfs_hybrid_10 = []
    perfs_mvdr = []
    
    print("开始测试（50个样本）...")
    for idx in range(num_tests):
        # 生成测试数据
        X, a_tgt = simulator.generate_batch(range_diff_mode='random')
        
        # 计算协方差矩阵
        with torch.no_grad():
            R = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L
            R = R[0]  # (MN, MN)
            a_tgt_vec = a_tgt[0]  # (MN,)
        
        # ===== 方法1：纯 DCVB =====
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        with torch.no_grad():
            w_dcvb = model(X, a_tgt)[0]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_dcvb = time.time() - t0
        
        P_dcvb = torch.real(w_dcvb.conj() @ R @ w_dcvb).item()
        times_dcvb.append(t_dcvb)
        perfs_dcvb.append(10 * np.log10(P_dcvb))
        
        # ===== 方法2：DCVB + 3次迭代 =====
        t0 = time.time()
        w_hybrid_3, power_hist_3 = power_iteration_refine(R, a_tgt_vec, w_dcvb, n_iters=3)
        t_refine_3 = time.time() - t0
        
        times_hybrid_3.append(t_dcvb + t_refine_3)
        perfs_hybrid_3.append(10 * np.log10(power_hist_3[-1]))
        
        # ===== 方法3：DCVB + 10次迭代 =====
        t0 = time.time()
        w_hybrid_10, power_hist_10 = power_iteration_refine(R, a_tgt_vec, w_dcvb, n_iters=10)
        t_refine_10 = time.time() - t0
        
        times_hybrid_10.append(t_dcvb + t_refine_10)
        perfs_hybrid_10.append(10 * np.log10(power_hist_10[-1]))
        
        # ===== 方法4：直接 MVDR =====
        t0 = time.time()
        MN = cfg.M * cfg.N
        R_inv = torch.linalg.inv(R + 1e-6 * torch.eye(MN, device=device))
        numerator = R_inv @ a_tgt_vec
        denominator = a_tgt_vec.conj() @ R_inv @ a_tgt_vec
        w_mvdr = numerator / denominator
        t_mvdr = time.time() - t0
        
        P_mvdr = torch.real(w_mvdr.conj() @ R @ w_mvdr).item()
        times_mvdr.append(t_mvdr)
        perfs_mvdr.append(10 * np.log10(P_mvdr))
        
        if (idx + 1) % 10 == 0:
            print(f"  已完成 {idx+1}/{num_tests}")
    
    # 计算统计量
    def stats(arr):
        return np.mean(arr), np.std(arr)
    
    print("\n" + "="*70)
    print("实验结果统计（50个样本平均）")
    print("="*70)
    
    t_dcvb_avg, t_dcvb_std = stats([t*1000 for t in times_dcvb])
    t_h3_avg, t_h3_std = stats([t*1000 for t in times_hybrid_3])
    t_h10_avg, t_h10_std = stats([t*1000 for t in times_hybrid_10])
    t_mvdr_avg, t_mvdr_std = stats([t*1000 for t in times_mvdr])
    
    p_dcvb_avg, p_dcvb_std = stats(perfs_dcvb)
    p_h3_avg, p_h3_std = stats(perfs_hybrid_3)
    p_h10_avg, p_h10_std = stats(perfs_hybrid_10)
    p_mvdr_avg, p_mvdr_std = stats(perfs_mvdr)
    
    print("\n【方法1】纯 DCVB（无微调）")
    print(f"  时间: {t_dcvb_avg:.2f} ± {t_dcvb_std:.2f} ms")
    print(f"  抑制: {p_dcvb_avg:.2f} ± {p_dcvb_std:.2f} dB")
    
    print("\n【方法2】混合方法（DCVB + 3次微调）")
    print(f"  时间: {t_h3_avg:.2f} ± {t_h3_std:.2f} ms")
    print(f"  抑制: {p_h3_avg:.2f} ± {p_h3_std:.2f} dB")
    print(f"  相比纯DCVB改进: {p_h3_avg - p_dcvb_avg:.2f} dB")
    print(f"  速度优势 vs MVDR: {t_mvdr_avg / t_h3_avg:.1f}× 更快")
    
    print("\n【方法3】混合方法（DCVB + 10次微调）")
    print(f"  时间: {t_h10_avg:.2f} ± {t_h10_std:.2f} ms")
    print(f"  抑制: {p_h10_avg:.2f} ± {p_h10_std:.2f} dB")
    print(f"  相比纯DCVB改进: {p_h10_avg - p_dcvb_avg:.2f} dB")
    print(f"  速度优势 vs MVDR: {t_mvdr_avg / t_h10_avg:.1f}× 更快")
    
    print("\n【方法4】直接 MVDR（闭式解，最优基准）")
    print(f"  时间: {t_mvdr_avg:.2f} ± {t_mvdr_std:.2f} ms")
    print(f"  抑制: {p_mvdr_avg:.2f} ± {p_mvdr_std:.2f} dB")
    
    # 可视化
    fig = plt.figure(figsize=(16, 5))
    
    # 子图1：性能对比（箱线图）
    ax1 = plt.subplot(1, 3, 1)
    positions = [1, 2, 3, 4]
    data = [perfs_dcvb, perfs_hybrid_3, perfs_hybrid_10, perfs_mvdr]
    bp = ax1.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['纯DCVB', 'DCVB+3次', 'DCVB+10次', 'MVDR'], fontsize=11)
    ax1.set_ylabel('干扰抑制 (dB)', fontsize=12)
    ax1.set_title('(a) 抑制深度对比（50样本）', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.axhline(y=p_mvdr_avg, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='MVDR平均')
    ax1.legend(fontsize=10)
    
    # 子图2：时间对比（箱线图）
    ax2 = plt.subplot(1, 3, 2)
    time_data = [[t*1000 for t in times_dcvb], 
                 [t*1000 for t in times_hybrid_3], 
                 [t*1000 for t in times_hybrid_10], 
                 [t*1000 for t in times_mvdr]]
    bp2 = ax2.boxplot(time_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      medianprops=dict(color='darkred', linewidth=2))
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['纯DCVB', 'DCVB+3次', 'DCVB+10次', 'MVDR'], fontsize=11)
    ax2.set_ylabel('计算时间 (ms)', fontsize=12)
    ax2.set_title('(b) 计算时间对比（50样本）', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, axis='y', alpha=0.3, which='both')
    
    # 子图3：性能-时间权衡（散点图）
    ax3 = plt.subplot(1, 3, 3)
    
    # 绘制所有样本
    ax3.scatter([t*1000 for t in times_dcvb], perfs_dcvb, alpha=0.3, s=30, c='orange', label='纯DCVB')
    ax3.scatter([t*1000 for t in times_hybrid_3], perfs_hybrid_3, alpha=0.3, s=30, c='blue', label='DCVB+3次')
    ax3.scatter([t*1000 for t in times_hybrid_10], perfs_hybrid_10, alpha=0.3, s=30, c='purple', label='DCVB+10次')
    ax3.scatter([t*1000 for t in times_mvdr], perfs_mvdr, alpha=0.3, s=30, c='green', label='MVDR')
    
    # 标注平均值
    methods = ['纯DCVB', 'DCVB+3次', 'DCVB+10次', 'MVDR']
    time_avgs = [t_dcvb_avg, t_h3_avg, t_h10_avg, t_mvdr_avg]
    perf_avgs = [p_dcvb_avg, p_h3_avg, p_h10_avg, p_mvdr_avg]
    colors = ['orange', 'blue', 'purple', 'green']
    markers = ['o', '^', 's', 'D']
    
    for method, t_avg, p_avg, color, marker in zip(methods, time_avgs, perf_avgs, colors, markers):
        ax3.scatter(t_avg, p_avg, s=300, c=color, marker=marker, 
                   edgecolors='black', linewidth=2.5, zorder=10)
        ax3.text(t_avg, p_avg + 1.5, method, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('计算时间 (ms)', fontsize=12)
    ax3.set_ylabel('干扰抑制 (dB)', fontsize=12)
    ax3.set_title('(c) 性能-时间权衡空间', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('exp_hybrid_warmstart_v2.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: exp_hybrid_warmstart_v2.png")
    
    # 打印混合方法的关键发现
    print("\n" + "="*70)
    print("混合方法的价值主张")
    print("="*70)
    
    improvement_3 = p_h3_avg - p_dcvb_avg
    speedup_3 = t_mvdr_avg / t_h3_avg
    gap_to_mvdr_3 = p_mvdr_avg - p_h3_avg
    
    print(f"\n🎯 最佳平衡点：DCVB + 3次微调")
    print(f"  ✅ 仅需 {t_h3_avg:.2f} ms（MVDR的 1/{speedup_3:.1f}）")
    print(f"  ✅ 抑制深度 {p_h3_avg:.2f} dB（比纯DCVB提升 {improvement_3:.2f} dB）")
    print(f"  ✅ 距离最优解仅 {abs(gap_to_mvdr_3):.2f} dB")
    print(f"\n💡 适用场景：")
    print(f"  - 实时跟踪：{1000/t_h3_avg:.0f} fps 吞吐量")
    print(f"  - 精度要求：中等（-{abs(p_h3_avg):.1f} dB 干扰抑制）")
    print(f"  - 资源受限：无需矩阵求逆（O(N²) vs O(N³)）")


if __name__ == '__main__':
    compare_hybrid_methods()
