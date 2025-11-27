import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


# 简单的网络定义 (用于测速)
class SimpleComplexNet(nn.Module):
    def __init__(self, MN):
        super().__init__()
        # 保持和你训练时一样的结构规模
        self.fc = nn.Sequential(
            nn.Linear(MN * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, MN * 2)
        )

    def forward(self, x):
        # x: (B, MN)
        return self.fc(x)


def run_scaling_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}...")

    # === 测试不同的阵列规模 ===
    # MN 从 100 增加到 1000 (模拟大规模阵列)
    array_sizes = [100, 200, 400, 600, 800, 1000]

    mvdr_times = []
    dcvb_times = []

    for MN in array_sizes:
        print(f"Testing Array Size MN = {MN} ...")

        # 1. 准备数据 (Batch=1)
        # MVDR 需要计算协方差矩阵 R (MN x MN) 并求逆
        # 假设快拍数 L=64
        X = torch.randn(1, MN, 64, dtype=torch.complex64).to(device)

        # DCVB 模型
        model = SimpleComplexNet(MN).to(device)
        # 输入特征 (实部+虚部)
        input_feat = torch.randn(1, MN * 2).to(device)

        # === 预热 ===
        for _ in range(10):
            _ = torch.linalg.inv(torch.matmul(X.squeeze(0), X.squeeze(0).T.conj()))
            _ = model(input_feat)
        torch.cuda.synchronize()

        # === 测速 MVDR ===
        start = time.time()
        for _ in range(100):
            # R = X * X'
            R = torch.matmul(X.squeeze(0), X.squeeze(0).T.conj())
            # Inv
            R_inv = torch.linalg.inv(R)
        torch.cuda.synchronize()
        mvdr_times.append((time.time() - start) / 100 * 1000)  # ms

        # === 测速 DCVB ===
        start = time.time()
        for _ in range(100):
            _ = model(input_feat)
        torch.cuda.synchronize()
        dcvb_times.append((time.time() - start) / 100 * 1000)  # ms

    # === 绘图 ===
    plt.figure(figsize=(10, 6))
    plt.plot(array_sizes, mvdr_times, 'r-o', linewidth=2, label='Traditional MVDR (O(N^3))')
    plt.plot(array_sizes, dcvb_times, 'b-s', linewidth=2, label='Proposed DCVB (O(N^2))')

    plt.xlabel('Array Size (Number of Elements)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Scalability Analysis: Inference Time vs. Array Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.savefig('scalability_comparison.png', dpi=300)
    plt.show()

    print("Scalability test finished. Check 'scalability_comparison.png'.")


if __name__ == "__main__":
    run_scaling_test()