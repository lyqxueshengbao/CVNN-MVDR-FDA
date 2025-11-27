"""
真实可扩展性测试：测量不同阵列规模下的 DCVB vs MVDR 运行时间
寻找"交叉点"（Crossover Point）
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from model import ComplexBeamformerNet
from config import Config

# 临时修改 Config 以支持动态调整阵列大小
class DynamicConfig(Config):
    def __init__(self, M, N):
        super().__init__()
        self.M = M
        self.N = N
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def measure_times(M_list, N_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    results = {
        'mn': [],
        'dcvb': [],
        'mvdr': []
    }
    
    for M, N in zip(M_list, N_list):
        MN = M * N
        cfg = DynamicConfig(M, N)
        print(f"\nTesting Array Size: {M}x{N} = {MN} elements")
        
        # 1. 初始化模型 (随机权重即可，只测速度)
        try:
            model = ComplexBeamformerNet(cfg=cfg).to(device)
            model.eval()
        except Exception as e:
            print(f"Model creation failed for {MN}: {e}")
            break
            
        # 2. 生成伪数据
        B = 1 # 单帧处理
        L = 64
        X = torch.randn(B, MN, L, dtype=torch.complex64, device=device)
        a_tgt = torch.randn(B, MN, dtype=torch.complex64, device=device)
        
        # 3. 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(X, a_tgt)
                
                # MVDR 预热
                R = torch.matmul(X, X.conj().transpose(-1, -2)) / L
                R = R[0]
                a = a_tgt[0]
                try:
                    torch.linalg.inv(R + 1e-4*torch.eye(MN, device=device))
                except:
                    pass
        
        torch.cuda.synchronize()
        
        # 4. 测量 DCVB
        t_start = time.time()
        num_loops = 50
        with torch.no_grad():
            for _ in range(num_loops):
                _ = model(X, a_tgt)
        torch.cuda.synchronize()
        t_dcvb = (time.time() - t_start) / num_loops * 1000 # ms
        
        # 5. 测量 MVDR
        # 注意：MVDR 包括 R 矩阵计算 + 求逆 + 乘法
        t_start = time.time()
        for _ in range(num_loops):
            # R 矩阵计算 (O(MN^2 * L))
            R = torch.matmul(X, X.conj().transpose(-1, -2)) / L
            R = R[0]
            a = a_tgt[0]
            
            # 求逆 (O(MN^3))
            # 使用 solve 替代 inv 通常更快且更稳，但为了公平对比标准MVDR公式，用inv或solve
            # w = R^-1 * a / (...)
            # 这里模拟完整的数值过程
            R_loaded = R + 1e-4 * torch.eye(MN, device=device)
            try:
                R_inv = torch.linalg.inv(R_loaded)
                num = R_inv @ a
                den = a.conj() @ R_inv @ a
                w = num / den
            except:
                pass
                
        torch.cuda.synchronize()
        t_mvdr = (time.time() - t_start) / num_loops * 1000 # ms
        
        print(f"  DCVB: {t_dcvb:.2f} ms")
        print(f"  MVDR: {t_mvdr:.2f} ms")
        
        results['mn'].append(MN)
        results['dcvb'].append(t_dcvb)
        results['mvdr'].append(t_mvdr)
        
        # 清理显存
        del model, X, a_tgt, R
        torch.cuda.empty_cache()
        
    return results

def plot_results(results):
    mn = np.array(results['mn'])
    dcvb = np.array(results['dcvb'])
    mvdr = np.array(results['mvdr'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(mn, dcvb, 'b-o', label='DCVB (Deep Learning)', linewidth=2)
    plt.plot(mn, mvdr, 'r-s', label='MVDR (Matrix Inversion)', linewidth=2)
    
    plt.xlabel('Number of Antennas (MN)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Scalability Analysis: DCVB vs MVDR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    
    # 标注交叉点
    # 简单的线性插值找交叉点
    diff = mvdr - dcvb
    # 找到 diff 符号变化的地方
    for i in range(len(diff)-1):
        if diff[i] < 0 and diff[i+1] > 0:
            # 交叉发生在 i 和 i+1 之间
            # 简单的线性插值
            x1, y1 = mn[i], diff[i]
            x2, y2 = mn[i+1], diff[i+1]
            # y = kx + b, find x where y=0
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            cross_x = -b / k
            
            plt.axvline(cross_x, color='k', linestyle='--', alpha=0.5)
            plt.text(cross_x, plt.ylim()[0], f'Crossover ~ {int(cross_x)} elements', 
                     rotation=90, va='bottom', ha='right')
            break
            
    plt.savefig('exp_scalability_real.png', dpi=300)
    print("Plot saved to exp_scalability_real.png")

if __name__ == "__main__":
    # 测试序列：10x10=100, 14x14~200, 20x20=400, 28x28~800, 40x40=1600
    M_list = [10, 14, 20, 28, 40]
    N_list = [10, 14, 20, 28, 40]
    
    results = measure_times(M_list, N_list)
    plot_results(results)
