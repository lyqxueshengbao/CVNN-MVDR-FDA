"""
混合架构综合对比实验：速度 vs 精度
直观展示 DCVB、MVDR 和 Hybrid 方法的性能权衡
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from config import Config
from data_gen_v2 import FdaMimoSimulatorV2
from model import ComplexBeamformerNet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_mvdr_direct(X, a_tgt, device):
    """MVDR 直接求逆解法 (包含 R 计算)"""
    B, MN, L = X.shape
    
    # 1. 计算协方差矩阵 R
    # R = (X * X^H) / L
    R = torch.matmul(X, X.conj().transpose(-1, -2)) / L
    
    # 对角加载
    loading = 1e-4 * torch.eye(MN, device=device).unsqueeze(0)
    R = R + loading
    
    # 2. 求逆并计算权值
    # w = R^-1 * a / (a^H * R^-1 * a)
    try:
        R_inv = torch.linalg.inv(R)
        num = torch.matmul(R_inv, a_tgt.unsqueeze(-1)) # (B, MN, 1)
        den = torch.matmul(a_tgt.unsqueeze(1).conj(), num) # (B, 1, 1)
        w = num / (den + 1e-12)
        w = w.squeeze(-1)
    except:
        # 伪逆作为备选
        R_inv = torch.linalg.pinv(R)
        num = torch.matmul(R_inv, a_tgt.unsqueeze(-1))
        den = torch.matmul(a_tgt.unsqueeze(1).conj(), num)
        w = num / (den + 1e-12)
        w = w.squeeze(-1)
        
    return w, R

def cg_step(R, a, w_init, steps=2):
    """执行指定步数的 CG 迭代"""
    w = w_init.clone()
    # 求解 Rx = a，初始猜测 x0 = w_init
    # 注意：这里简化处理，直接在权值空间微调
    # 实际上 MVDR 求解的是 Rx = a，然后归一化
    # 我们假设 w_init 已经是 Rx=a 的近似解（差一个比例因子）
    
    # 重新缩放 w_init 以匹配 Rx=a 的幅度
    # alpha = (R*w)^H * a / ||R*w||^2
    Rw = torch.matmul(R, w.unsqueeze(-1)).squeeze(-1)
    alpha = torch.sum(Rw.conj() * a, dim=1) / (torch.sum(Rw.conj() * Rw, dim=1) + 1e-12)
    x = w * alpha.unsqueeze(1)
    
    r = a - torch.matmul(R, x.unsqueeze(-1)).squeeze(-1)
    p = r.clone()
    rsold = torch.sum(r.conj() * r, dim=1)
    
    for _ in range(steps):
        Ap = torch.matmul(R, p.unsqueeze(-1)).squeeze(-1)
        alpha_cg = rsold / (torch.sum(p.conj() * Ap, dim=1) + 1e-12)
        
        x = x + alpha_cg.unsqueeze(1) * p
        r = r - alpha_cg.unsqueeze(1) * Ap
        rsnew = torch.sum(r.conj() * r, dim=1)
        
        p = r + (rsnew / (rsold + 1e-12)).unsqueeze(1) * p
        rsold = rsnew
        
    # 归一化得到最终权值
    # w = x / (a^H x)
    norm_factor = torch.sum(a.conj() * x, dim=1)
    w_final = x / (norm_factor.unsqueeze(1) + 1e-12)
    
    return w_final

def evaluate_batch(w, R):
    """计算批次的平均抑制深度"""
    # P = w^H R w
    # (B, 1, MN) @ (B, MN, MN) @ (B, MN, 1)
    w_uns = w.unsqueeze(-1)
    P = torch.matmul(w_uns.transpose(-1, -2).conj(), torch.matmul(R, w_uns))
    P = P.squeeze().real
    # 如果 P 是标量 (batch_size=1)，需要处理
    if P.ndim == 0:
        P = P.unsqueeze(0)
    return 10 * np.log10(P.cpu().numpy() + 1e-12)

def run_comprehensive_comparison():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model = ComplexBeamformerNet(cfg=cfg).to(device)
    try:
        model.load_state_dict(torch.load('fda_improved.pth', map_location=device, weights_only=False))
    except:
        print("Warning: Model not found, using random weights.")
    model.eval()
    
    simulator = FdaMimoSimulatorV2(cfg)
    
    # 存储结果
    results = {
        'DCVB': {'time': [], 'supp': []},
        'MVDR': {'time': [], 'supp': []},
        'Hybrid': {'time': [], 'supp': []}
    }
    
    num_trials = 50
    print(f"Running {num_trials} trials...")
    
    # 预热 (更充分的预热)
    print("Warming up CUDA...")
    X_warm, a_warm = simulator.generate_batch(range_diff_mode='fixed')
    X_warm = X_warm[0:1] # 使用相同的 batch size = 1
    a_warm = a_warm[0:1]
    
    for _ in range(100): # 预热 100 次
        with torch.no_grad():
            model(X_warm, a_warm)
            # 同时也预热 MVDR 的相关计算
            R_warm = torch.matmul(X_warm, X_warm.conj().transpose(-1, -2)) / cfg.L
            torch.linalg.inv(R_warm + 1e-4*torch.eye(cfg.M*cfg.N, device=device))
            
    torch.cuda.synchronize()
    print("Warmup complete.")
    
    for i in range(num_trials):
        # 生成数据 (Batch size = 1 to measure latency accurately)
        # 修改 simulator 以支持 batch_size=1 (如果 generate_batch 不支持参数，则手动切片)
        X_batch, a_batch = simulator.generate_batch(range_diff_mode='random')
        X = X_batch[0:1]
        a_tgt = a_batch[0:1]
        
        # 1. DCVB
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            w_dcvb = model(X, a_tgt)
        torch.cuda.synchronize()
        t_dcvb = (time.time() - t0) * 1000 # ms
        
        # 计算 R 用于评估和后续方法 (注意：DCVB本身不需要R，但评估需要)
        # 为了公平对比时间，DCVB的时间只包含推理。
        # 评估用的 R 计算不应计入 DCVB 的运行时间。
        R_batch = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L
        loading = 1e-4 * torch.eye(cfg.M * cfg.N, device=device).unsqueeze(0)
        R = R_batch + loading
        
        supp_dcvb = evaluate_batch(w_dcvb, R)[0]
        
        results['DCVB']['time'].append(t_dcvb)
        results['DCVB']['supp'].append(supp_dcvb)
        
        # 2. MVDR (Direct)
        # 时间包含：R计算 + 求逆
        torch.cuda.synchronize()
        t0 = time.time()
        w_mvdr, _ = get_mvdr_direct(X, a_tgt, device)
        torch.cuda.synchronize()
        t_mvdr = (time.time() - t0) * 1000
        
        supp_mvdr = evaluate_batch(w_mvdr, R)[0]
        
        results['MVDR']['time'].append(t_mvdr)
        results['MVDR']['supp'].append(supp_mvdr)
        
        # 3. Hybrid (DCVB + 2-step CG)
        # 时间包含：DCVB推理 + R计算 + 2步CG
        # 注意：Hybrid 需要 R，所以 R 的计算时间必须计入！
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Step A: DCVB
        with torch.no_grad():
            w_init = model(X, a_tgt)
            
        # Step B: Calc R (Hybrid 需要用到 R 来做 CG)
        R_hybrid = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L
        R_hybrid = R_hybrid + loading
        
        # Step C: CG Fine-tune (2 steps)
        w_hybrid = cg_step(R_hybrid, a_tgt, w_init, steps=2)
        
        torch.cuda.synchronize()
        t_hybrid = (time.time() - t0) * 1000
        
        supp_hybrid = evaluate_batch(w_hybrid, R)[0]
        
        results['Hybrid']['time'].append(t_hybrid)
        results['Hybrid']['supp'].append(supp_hybrid)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_trials}")

    # 统计
    print("\n=== 结果统计 ===")
    for method in results:
        avg_time = np.mean(results[method]['time'])
        avg_supp = np.mean(results[method]['supp'])
        print(f"{method}: Time = {avg_time:.2f} ms, Supp = {avg_supp:.2f} dB")

    # === 可视化 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1：速度-精度散点图
    colors = {'DCVB': '#1f77b4', 'MVDR': '#d62728', 'Hybrid': '#2ca02c'}
    markers = {'DCVB': 'o', 'MVDR': 's', 'Hybrid': '^'}
    
    for method in results:
        ax1.scatter(results[method]['time'], results[method]['supp'], 
                    c=colors[method], marker=markers[method], label=method, alpha=0.6, s=50)
        
        # 画平均点
        avg_t = np.mean(results[method]['time'])
        avg_s = np.mean(results[method]['supp'])
        ax1.scatter(avg_t, avg_s, c=colors[method], marker=markers[method], s=200, edgecolors='k', linewidth=2)
        ax1.text(avg_t, avg_s - 2, f"{method}\n({avg_t:.1f}ms, {avg_s:.1f}dB)", 
                 ha='center', va='top', fontsize=10, fontweight='bold', color=colors[method])

    ax1.set_xlabel('推理时间 (ms)', fontsize=12)
    ax1.set_ylabel('干扰抑制深度 (dB)', fontsize=12)
    ax1.set_title('速度 vs 精度权衡 (Trade-off)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # 反转Y轴，因为越低越好？不，抑制深度通常是负数，越小越好。
    # 或者用正数表示衰减量。这里保持负数，越低越好。
    
    # 图2：综合柱状图
    methods = list(results.keys())
    avg_times = [np.mean(results[m]['time']) for m in methods]
    avg_supps = [np.mean(results[m]['supp']) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax2_time = ax2.twinx()
    
    # 精度柱 (左轴)
    rects1 = ax2.bar(x - width/2, avg_supps, width, label='Suppression (dB)', color='skyblue', alpha=0.8)
    ax2.set_ylabel('干扰抑制 (dB)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim([min(avg_supps)*1.2, 0]) # 负数轴
    
    # 时间柱 (右轴)
    rects2 = ax2_time.bar(x + width/2, avg_times, width, label='Time (ms)', color='orange', alpha=0.8)
    ax2_time.set_ylabel('时间 (ms)', color='darkorange', fontsize=12)
    ax2_time.tick_params(axis='y', labelcolor='darkorange')
    ax2_time.set_ylim([0, max(avg_times)*1.5])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax2.set_title('平均性能对比', fontsize=14)
    
    # 添加数值标签
    def autolabel(rects, ax_target, format_str):
        for rect in rects:
            height = rect.get_height()
            ax_target.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, ax2, '{:.1f}')
    autolabel(rects2, ax2_time, '{:.1f}')
    
    plt.tight_layout()
    plt.savefig('exp_hybrid_comprehensive.png', dpi=300)
    print("\n图表已保存: exp_hybrid_comprehensive.png")

if __name__ == "__main__":
    run_comprehensive_comparison()
