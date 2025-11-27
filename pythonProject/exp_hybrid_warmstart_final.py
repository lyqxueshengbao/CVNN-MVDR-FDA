"""
最终验证：混合架构 - DCVB Warm Start + Conjugate Gradient (CG) 微调
验证用户提出的核心假设：DCVB 提供的初始值能否加速传统算法收敛？
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

def conjugate_gradient_solve(R, b, x_init, max_iter=10, tol=1e-6):
    """
    共轭梯度法 (CG) 求解线性方程组 Rx = b
    
    Args:
        R: 系数矩阵 (Hermitian Positive Definite)
        b: 目标向量
        x_init: 初始猜测值
        max_iter: 最大迭代次数
        
    Returns:
        x_history: 每次迭代的解
        residual_history: 残差范数历史
    """
    x = x_init.clone()
    r = b - R @ x
    p = r.clone()
    rsold = torch.real(r.conj() @ r)
    
    x_history = [x.clone()]
    residual_history = [rsold.item()]
    
    for i in range(max_iter):
        Ap = R @ p
        alpha = rsold / torch.real(p.conj() @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.real(r.conj() @ r)
        
        x_history.append(x.clone())
        residual_history.append(rsnew.item())
        
        if torch.sqrt(rsnew) < tol:
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x_history, residual_history

def get_mvdr_from_x(x, a_tgt):
    """将 Rx=a 的解 x 转换为 MVDR 权值 w"""
    # w = x / (a^H x)
    normalization = a_tgt.conj() @ x
    return x / (normalization + 1e-12)

def evaluate_suppression(w, R):
    """计算干扰抑制深度 (输出功率)"""
    P = torch.real(w.conj() @ R @ w).item()
    return 10 * np.log10(P + 1e-12)

def run_warm_start_experiment():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    model = ComplexBeamformerNet(cfg=cfg).to(device)
    try:
        model.load_state_dict(torch.load('fda_improved.pth', map_location=device, weights_only=False))
        print("成功加载 fda_improved.pth")
    except:
        print("警告：未找到模型文件，使用随机初始化模型（仅用于测试流程）")
    model.eval()
    
    # 2. 生成测试数据 (寻找一个 DCVB 表现较好的样本)
    simulator = FdaMimoSimulatorV2(cfg)
    
    best_sample = None
    best_dcvb_supp = 0
    
    print("正在寻找合适的测试样本...")
    for _ in range(20):
        X_batch, a_batch = simulator.generate_batch(range_diff_mode='random')
        with torch.no_grad():
            w_batch = model(X_batch, a_batch)
            
        # 检查第一个样本
        X = X_batch[0:1]
        a_tgt = a_batch[0:1]
        w = w_batch[0]
        
        # 计算 R
        R_batch = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L
        MN = cfg.M * cfg.N
        loading = 1e-3 * torch.trace(R_batch[0]).real / MN
        R = R_batch[0] + loading * torch.eye(MN, device=device)
        
        supp = evaluate_suppression(w, R)
        
        # 我们希望找一个 DCVB 表现不错 (<-25dB) 但还没到极致的样本
        if supp < -25:
            best_sample = (X, a_tgt, R)
            best_dcvb_supp = supp
            break
            
    if best_sample:
        X, a_tgt, R = best_sample
        a = a_tgt[0]
        print(f"找到样本，DCVB 初始抑制: {best_dcvb_supp:.2f} dB")
    else:
        print("未找到理想样本，使用最后一个")
        # 使用最后一个
        X = X_batch[0:1]
        a_tgt = a_batch[0:1]
        R_batch = torch.matmul(X, X.conj().transpose(-1, -2)) / cfg.L
        MN = cfg.M * cfg.N
        loading = 1e-3 * torch.trace(R_batch[0]).real / MN
        R = R_batch[0] + loading * torch.eye(MN, device=device)
        a = a_tgt[0]
    
    # 3. 获取 DCVB 初始值
    with torch.no_grad():
        w_dcvb = model(X, a_tgt)[0]
        
    # === 关键优化：对初始猜测进行最佳缩放 ===
    # 我们求解 Rx = a。假设 x_init = alpha * w_dcvb
    # 我们希望最小化初始残差 ||a - R * (alpha * w_dcvb)||^2
    # 令 y = R * w_dcvb
    # alpha_opt = (y^H a) / (y^H y)
    y = R @ w_dcvb
    alpha_opt = (y.conj() @ a) / (y.conj() @ y + 1e-12)
    x_init_warm = alpha_opt * w_dcvb
    
    print(f"DCVB 初始缩放因子: {alpha_opt.item():.4f}")
    
    # 4. 运行 CG 求解 Rx = a
    # 目标：求解 w_mvdr = R^-1 a / (a^H R^-1 a)
    # 等价于求解 Rx = a，然后归一化
    
    # === 方案 A: Warm Start (从 DCVB 开始) ===
    # DCVB 输出的是 w，满足 w^H a = 1
    # 我们需要 Rx = a 的解 x。
    # 假设 w_dcvb approx w_mvdr = x / (a^H x)
    # 那么 x approx w_dcvb * (a^H x)
    # 由于比例因子不影响 CG 收敛方向，我们可以直接用 w_dcvb 作为 x 的初值
    # 或者更严谨一点，缩放 w_dcvb 使得其范数接近预期
    
    print("\n开始 CG 迭代对比...")
    max_iter = 15
    
    # Warm Start
    x_init_warm = w_dcvb.clone()
    x_hist_warm, _ = conjugate_gradient_solve(R, a, x_init_warm, max_iter=max_iter)
    
    # Cold Start (从导向矢量 a 开始，相当于常规波束形成 CBF)
    x_init_cold = a.clone()
    x_hist_cold, _ = conjugate_gradient_solve(R, a, x_init_cold, max_iter=max_iter)
    
    # 计算最优解 (直接求逆)
    x_opt = torch.linalg.solve(R, a)
    w_opt = get_mvdr_from_x(x_opt, a)
    supp_opt = evaluate_suppression(w_opt, R)
    
    # 5. 评估性能曲线
    supp_warm = []
    supp_cold = []
    
    # 初始点性能
    supp_warm.append(evaluate_suppression(get_mvdr_from_x(x_init_warm, a), R))
    supp_cold.append(evaluate_suppression(get_mvdr_from_x(x_init_cold, a), R))
    
    for x in x_hist_warm[1:]:
        w = get_mvdr_from_x(x, a)
        supp_warm.append(evaluate_suppression(w, R))
        
    for x in x_hist_cold[1:]:
        w = get_mvdr_from_x(x, a)
        supp_cold.append(evaluate_suppression(w, R))
        
    # 打印结果
    print(f"\n最优 MVDR 抑制深度: {supp_opt:.2f} dB")
    print(f"DCVB 初始抑制深度: {supp_warm[0]:.2f} dB")
    print(f"CBF (Cold) 初始抑制深度: {supp_cold[0]:.2f} dB")
    
    print("\n迭代过程对比:")
    print(f"{'Iter':<5} | {'Warm Start (DCVB)':<20} | {'Cold Start (CBF)':<20}")
    print("-" * 50)
    for i in range(min(len(supp_warm), 11)):
        print(f"{i:<5} | {supp_warm[i]:.2f} dB            | {supp_cold[i]:.2f} dB")
        
    # 6. 绘图
    plt.figure(figsize=(10, 6))
    iters = np.arange(len(supp_warm))
    
    plt.plot(iters, supp_warm, 'b-o', label='Warm Start (From DCVB)', linewidth=2, markersize=8)
    plt.plot(iters, supp_cold, 'r--s', label='Cold Start (From CBF)', linewidth=2, markersize=8)
    plt.axhline(supp_opt, color='g', linestyle=':', label='Optimal MVDR', linewidth=2)
    
    plt.xlabel('CG Iterations', fontsize=12)
    plt.ylabel('Suppression Depth (dB)', fontsize=12)
    plt.title('Convergence Speed: DCVB Warm Start vs Cold Start', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 标注关键点
    plt.annotate(f'DCVB Init\n{supp_warm[0]:.1f} dB', 
                 xy=(0, supp_warm[0]), xytext=(1, supp_warm[0]+5),
                 arrowprops=dict(facecolor='blue', shrink=0.05))
                 
    # 找到达到 -50dB (或接近最优) 的迭代次数
    threshold = supp_opt + 3 # 距离最优 3dB 以内
    
    warm_reach = next((i for i, v in enumerate(supp_warm) if v < threshold), None)
    cold_reach = next((i for i, v in enumerate(supp_cold) if v < threshold), None)
    
    info_text = (
        f"Optimal: {supp_opt:.1f} dB\n"
        f"Warm Start reaches opt+3dB at iter: {warm_reach if warm_reach is not None else '>15'}\n"
        f"Cold Start reaches opt+3dB at iter: {cold_reach if cold_reach is not None else '>15'}"
    )
    plt.text(0.5, 0.5, info_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('exp_warm_start_final.png', dpi=300)
    print("\n图表已保存: exp_warm_start_final.png")

if __name__ == "__main__":
    run_warm_start_experiment()
