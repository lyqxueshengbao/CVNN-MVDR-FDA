"""
混合预热启动实验：DCVB初始化 + MVDR微调
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

def mvdr_iterative_refine(R, a_target, w_init, max_iters=10):
    """
    从 w_init 开始，迭代微调至 MVDR 最优解
    使用梯度下降法，逐步降低输出功率
    
    Args:
        R: 协方差矩阵 (MN, MN)
        a_target: 目标导向矢量 (MN,)
        w_init: DCVB 的初始权值 (MN,)
        max_iters: 最大迭代次数
    
    Returns:
        w_history: 每次迭代的权值 (max_iters+1, MN)
        power_history: 每次迭代的输出功率
        suppression_history: 每次迭代的干扰抑制深度
    """
    MN = R.shape[0]
    device = R.device
    
    # 初始化
    w = w_init.clone()
    w_history = [w.clone()]
    power_history = []
    suppression_history = []
    
    # 计算初始功率
    P_init = torch.real(w.conj().T @ R @ w).item()
    power_history.append(P_init)
    
    # 学习率（自适应）
    lr = 0.1
    
    for it in range(max_iters):
        # 梯度：∇_w P = 2*R*w（对于 Hermitian R）
        grad = 2 * R @ w
        
        # 梯度下降
        w_new = w - lr * grad
        
        # 投影到约束流形：w^H * a = 1
        numerator = w_new.conj().T @ a_target - 1
        denominator = torch.norm(a_target)**2
        projection = a_target * (numerator / denominator)
        w_new = w_new - projection
        
        # 计算新功率
        P_new = torch.real(w_new.conj().T @ R @ w_new).item()
        
        # 如果功率增加，减小学习率并回退
        if P_new > power_history[-1]:
            lr *= 0.5
            continue
        
        # 接受更新
        w = w_new
        w_history.append(w.clone())
        power_history.append(P_new)
        
        # 计算抑制深度（相对于初始功率）
        suppression_db = 10 * np.log10(P_new / P_init)
        suppression_history.append(suppression_db)
        
        # 如果收敛（功率变化 < 0.1%）
        if len(power_history) > 1:
            relative_change = abs(P_new - power_history[-2]) / power_history[-2]
            if relative_change < 1e-3:
                break
    
    return torch.stack(w_history), np.array(power_history), np.array(suppression_history)


def compare_convergence():
    """
    对比三种方法的收敛曲线：
    1. DCVB初始化 + 微调（Warm-start）
    2. 零初始化 + 迭代（Cold-start）
    3. 直接MVDR（闭式解，作为最优基准）
    """
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 DCVB 模型
    model = ComplexBeamformerNet(cfg=cfg).to(device)
    
    model.load_state_dict(torch.load('fda_improved.pth', map_location=device))
    model.eval()
    
    # 生成测试数据
    simulator = FdaMimoSimulatorV2(cfg)
    X, a_tgt = simulator.generate_batch(range_diff_mode='fixed')
    
    # X 和 a_tgt 已经是 Tensor，无需转换
    X = X.to(device)
    a_tgt = a_tgt.to(device)
    
    # 计算协方差矩阵
    with torch.no_grad():
        R = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L  # (1, MN, MN)
        R = R[0]  # (MN, MN)
        a_tgt_vec = a_tgt[0]  # (MN,)
    
    # 方法1：DCVB初始化（Warm-start）
    print("\n=== 方法1：DCVB初始化（Warm-start）===")
    t0 = time.time()
    with torch.no_grad():
        w_dcvb = model(X, a_tgt)  # (1, MN)
        w_dcvb = w_dcvb[0]  # (MN,)
    t_dcvb = time.time() - t0
    
    # 计算 DCVB 的初始性能
    P_dcvb = torch.real(w_dcvb.conj().T @ R @ w_dcvb).item()
    print(f"DCVB 初始化时间: {t_dcvb*1000:.2f} ms")
    print(f"DCVB 输出功率: {10*np.log10(P_dcvb):.2f} dB")
    
    # 从 DCVB 权值开始微调
    t0 = time.time()
    w_warm_history, P_warm_history, supp_warm = mvdr_iterative_refine(
        R, a_tgt_vec, w_dcvb, max_iters=20
    )
    t_warm_refine = time.time() - t0
    
    print(f"微调迭代次数: {len(P_warm_history)-1}")
    print(f"微调时间: {t_warm_refine*1000:.2f} ms")
    print(f"最终输出功率: {10*np.log10(P_warm_history[-1]):.2f} dB")
    print(f"总时间: {(t_dcvb+t_warm_refine)*1000:.2f} ms")
    
    # 方法2：零初始化（Cold-start）
    print("\n=== 方法2：零初始化（Cold-start）===")
    MN = cfg.M * cfg.N
    w_zero = torch.zeros(MN, dtype=torch.complex64, device=device)
    # 投影到约束流形
    w_zero = a_tgt_vec / torch.norm(a_tgt_vec)**2
    
    t0 = time.time()
    w_cold_history, P_cold_history, supp_cold = mvdr_iterative_refine(
        R, a_tgt_vec, w_zero, max_iters=20
    )
    t_cold = time.time() - t0
    
    print(f"迭代次数: {len(P_cold_history)-1}")
    print(f"总时间: {t_cold*1000:.2f} ms")
    print(f"最终输出功率: {10*np.log10(P_cold_history[-1]):.2f} dB")
    
    # 方法3：直接MVDR（闭式解）
    print("\n=== 方法3：直接MVDR（闭式解）===")
    t0 = time.time()
    R_inv = torch.linalg.inv(R + 1e-6 * torch.eye(MN, device=device))
    numerator = R_inv @ a_tgt_vec
    denominator = a_tgt_vec.conj().T @ R_inv @ a_tgt_vec
    w_mvdr = numerator / denominator
    t_mvdr = time.time() - t0
    
    P_mvdr = torch.real(w_mvdr.conj().T @ R @ w_mvdr).item()
    
    print(f"MVDR 时间: {t_mvdr*1000:.2f} ms")
    print(f"MVDR 输出功率: {10*np.log10(P_mvdr):.2f} dB")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 子图1：输出功率收敛曲线
    ax = axes[0]
    iterations_warm = np.arange(len(P_warm_history))
    iterations_cold = np.arange(len(P_cold_history))
    
    ax.plot(iterations_warm, 10*np.log10(P_warm_history), 
            'b-o', label='Warm-start (DCVB初始化)', linewidth=2, markersize=6)
    ax.plot(iterations_cold, 10*np.log10(P_cold_history), 
            'r--s', label='Cold-start (零初始化)', linewidth=2, markersize=6)
    ax.axhline(10*np.log10(P_mvdr), color='g', linestyle=':', 
               linewidth=2, label='MVDR最优解')
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('输出功率 (dB)', fontsize=12)
    ax.set_title('(a) 收敛速度对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图2：累积时间对比
    ax = axes[1]
    
    # Warm-start：DCVB时间 + 每步微调时间
    time_per_iter_warm = t_warm_refine / (len(P_warm_history) - 1) if len(P_warm_history) > 1 else 0
    cumulative_time_warm = [t_dcvb + i * time_per_iter_warm for i in iterations_warm]
    
    # Cold-start：每步迭代时间
    time_per_iter_cold = t_cold / (len(P_cold_history) - 1) if len(P_cold_history) > 1 else 0
    cumulative_time_cold = [i * time_per_iter_cold for i in iterations_cold]
    
    ax.plot(cumulative_time_warm, 10*np.log10(P_warm_history), 
            'b-o', label='Warm-start', linewidth=2, markersize=6)
    ax.plot(cumulative_time_cold, 10*np.log10(P_cold_history), 
            'r--s', label='Cold-start', linewidth=2, markersize=6)
    ax.axhline(10*np.log10(P_mvdr), color='g', linestyle=':', 
               linewidth=2, label='MVDR')
    
    # 标注MVDR时间
    ax.axvline(t_mvdr, color='g', linestyle=':', alpha=0.5)
    ax.text(t_mvdr, ax.get_ylim()[0] + 0.1*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
            f'MVDR: {t_mvdr*1000:.1f}ms', 
            rotation=90, verticalalignment='bottom', fontsize=9, color='g')
    
    ax.set_xlabel('累积时间 (秒)', fontsize=12)
    ax.set_ylabel('输出功率 (dB)', fontsize=12)
    ax.set_title('(b) 时间效率对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图3：性能-时间权衡
    ax = axes[2]
    
    methods = ['DCVB\n仅初始化', 'Warm-start\n(3次迭代)', 'Cold-start\n(完整收敛)', 'MVDR\n闭式解']
    
    # 选择具有代表性的点
    if len(P_warm_history) >= 4:
        perf_warm_3iter = 10*np.log10(P_warm_history[3])
        time_warm_3iter = (t_dcvb + 3*time_per_iter_warm) * 1000
    else:
        perf_warm_3iter = 10*np.log10(P_warm_history[-1])
        time_warm_3iter = (t_dcvb + t_warm_refine) * 1000
    
    performances = [
        10*np.log10(P_dcvb),
        perf_warm_3iter,
        10*np.log10(P_cold_history[-1]),
        10*np.log10(P_mvdr)
    ]
    
    times = [
        t_dcvb * 1000,
        time_warm_3iter,
        t_cold * 1000,
        t_mvdr * 1000
    ]
    
    colors = ['orange', 'blue', 'red', 'green']
    markers = ['o', '^', 's', 'D']
    
    for i, (method, perf, t, c, m) in enumerate(zip(methods, performances, times, colors, markers)):
        ax.scatter(t, perf, s=200, c=c, marker=m, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(t, perf+1, method, ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('时间 (ms)', fontsize=12)
    ax.set_ylabel('输出功率 (dB)', fontsize=12)
    ax.set_title('(c) 性能-时间权衡', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.05, 20])
    
    plt.tight_layout()
    plt.savefig('exp_hybrid_warmstart.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: exp_hybrid_warmstart.png")
    
    # 打印总结
    print("\n" + "="*60)
    print("混合方法优势总结")
    print("="*60)
    
    # 计算3次迭代后的性能
    if len(P_warm_history) >= 4:
        improvement_3iter = 10*np.log10(P_warm_history[3] / P_dcvb)
        gap_to_mvdr_3iter = 10*np.log10(P_warm_history[3] / P_mvdr)
        print(f"\n仅用 3 次迭代 (Warm-start):")
        print(f"  相比DCVB改进: {improvement_3iter:.2f} dB")
        print(f"  距离MVDR: {gap_to_mvdr_3iter:.2f} dB")
        print(f"  总时间: {time_warm_3iter:.2f} ms (MVDR的 {time_warm_3iter/t_mvdr/1000:.1f}×)")
        
        speedup_vs_mvdr = (t_mvdr * 1000) / time_warm_3iter
        print(f"  速度优势: {speedup_vs_mvdr:.1f}× 快于MVDR")
    
    # Cold-start 需要的迭代次数
    print(f"\nCold-start 达到相同性能:")
    print(f"  需要迭代次数: {len(P_cold_history)-1}")
    print(f"  总时间: {t_cold*1000:.2f} ms")
    
    # 时间节省
    time_saved = (t_cold - (t_dcvb + t_warm_refine)) * 1000
    if time_saved > 0:
        print(f"\n使用 Warm-start 节省时间: {time_saved:.2f} ms ({time_saved/t_cold/10:.1f}%)")


if __name__ == '__main__':
    compare_convergence()
