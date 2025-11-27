import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from model import ComplexBeamformerNet


def run_real_scaling_test():
    # 强制使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking REAL DCVB Model on {device}...")

    # === 测试不同的阵列规模 ===
    # 模拟从 100 到 1000 个阵元
    # 为了方便，我们固定 N=10，只增加 M (或者直接把 MN 当作总数)
    # 这里我们设定 total_elements = M * N
    total_elements_list = [100, 200, 400, 600, 800, 1000]

    mvdr_times = []
    dcvb_times = []

    cfg = Config()
    # 保持快拍数一致
    L = cfg.L

    for total_elements in total_elements_list:
        print(f"Testing Array Size MN = {total_elements} ...")

        # 1. 动态调整配置
        # 假设 N=10, M 随之变化 (或者 N=1, M=total)
        # 为了不破坏 model 里的 MN 计算，我们直接临时覆盖
        cfg.N = 10
        cfg.M = total_elements // 10
        MN = cfg.M * cfg.N

        # 2. 实例化真正的 DCVB 模型
        # 这会创建一个真正的 ComplexBeamformerNet，包含所有卷积和投影层
        model = ComplexBeamformerNet(cfg).to(device)
        model.eval()

        # 3. 准备数据 (Batch=1)
        # 输入 X: (1, MN, L)
        X = torch.randn(1, MN, L, dtype=torch.complex64).to(device)
        # 输入 a_tgt: (1, MN)
        a_tgt = torch.randn(1, MN, dtype=torch.complex64).to(device)

        # === 预热 (Warm Up) ===
        # 让 GPU 缓存一下 kernel
        for _ in range(10):
            # MVDR 预热
            R = torch.matmul(X.squeeze(0), X.squeeze(0).T.conj())
            _ = torch.linalg.inv(R)
            # DCVB 预热
            _ = model(X, a_tgt)
        torch.cuda.synchronize()

        # === 测速 MVDR (O(N^3)) ===
        start = time.time()
        for _ in range(100):
            # 1. 算协方差 (MN x MN)
            R = torch.matmul(X.squeeze(0), X.squeeze(0).T.conj())
            # 2. 加载对角 (模拟真实情况)
            R = R + 1e-6 * torch.eye(MN, device=device)
            # 3. 求逆 (瓶颈)
            R_inv = torch.linalg.inv(R)
            # 4. 算权值 (矩阵乘法)
            # w = R^-1 * a / (...)
            num = torch.matmul(R_inv, a_tgt.T)
            den = torch.matmul(a_tgt.conj(), num)
            _ = num / den
        torch.cuda.synchronize()
        mvdr_times.append((time.time() - start) / 100 * 1000)  # ms

        # === 测速 DCVB (O(N)) ===
        # 你的网络主要是 1D 卷积和全连接，相对于 MN 是线性的
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(X, a_tgt)
        torch.cuda.synchronize()
        dcvb_times.append((time.time() - start) / 100 * 1000)  # ms

    # === 绘图 ===
    plt.figure(figsize=(10, 6))

    # MVDR: 红色圆点
    plt.plot(total_elements_list, mvdr_times, 'r-o', linewidth=2, label='Traditional MVDR ($O((MN)^3)$)')

    # DCVB: 蓝色方块
    plt.plot(total_elements_list, dcvb_times, 'b-s', linewidth=2, label='Proposed DCVB ($O(MN)$)')

    plt.xlabel('Array Size (Total Elements $M \\times N$)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Scalability Analysis: Real Model vs. Matrix Inversion', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.savefig('scalability_real.png', dpi=300)
    plt.show()

    print("\nBenchmark Finished!")
    print("Check 'scalability_real.png' for the comparison plot.")
    # 打印最后一次的数据，方便写论文
    print(f"At MN={total_elements_list[-1]}:")
    print(f"MVDR Time: {mvdr_times[-1]:.4f} ms")
    print(f"DCVB Time: {dcvb_times[-1]:.4f} ms")
    print(f"Speedup: {mvdr_times[-1] / dcvb_times[-1]:.2f}x")


if __name__ == "__main__":
    run_real_scaling_test()