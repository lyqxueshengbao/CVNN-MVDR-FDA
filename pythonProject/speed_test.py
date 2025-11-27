import torch
import time
import numpy as np
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def get_mvdr_weights_fast(X_input, a_tgt):
    """
    传统 MVDR 的标准实现
    瓶颈在于: torch.linalg.inv (矩阵求逆)
    """
    # X_input: (1, MN, L)
    X = X_input.squeeze(0)
    L = X.shape[1]

    # 1. 计算协方差矩阵 R (MN x MN) -> O(MN^2 * L)
    R = torch.matmul(X, X.conj().T) / L
    R = R + 1e-6 * torch.eye(R.shape[0], device=R.device)

    # 2. 矩阵求逆 -> O(MN^3) !!! 这是最慢的一步
    R_inv = torch.linalg.inv(R)

    # 3. 权值计算
    a = a_tgt.T
    numerator = torch.matmul(R_inv, a)
    denominator = torch.matmul(a.conj().T, numerator)
    w = numerator / (denominator + 1e-10)
    return w


def run_speed_benchmark():
    # 初始化
    cfg = Config()
    sim = FdaMimoSimulator(cfg)

    # 加载模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.eval()

    # 准备一条数据 (预热用)
    X, a_tgt = sim.generate_batch()
    # 取单条数据模拟实时处理 (Batch=1)
    X_single = X[0:1]
    a_single = a_tgt[0:1]

    # === 预热 (Warm Up) ===
    print("Warming up...")
    for _ in range(100):
        _ = model(X_single, a_single)
        _ = get_mvdr_weights_fast(X_single, a_single)

    # 同步 CUDA (如果是 GPU)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # === 测试 DCVB (你的方法) ===
    print("Benchmarking DCVB (Deep Learning)...")
    start_time = time.time()
    num_trials = 2000
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(X_single, a_single)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()
    dcvb_avg_time = (end_time - start_time) / num_trials * 1000  # 毫秒

    # === 测试 MVDR (传统方法) ===
    print("Benchmarking MVDR (Matrix Inversion)...")
    start_time = time.time()
    for _ in range(num_trials):
        _ = get_mvdr_weights_fast(X_single, a_single)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()
    mvdr_avg_time = (end_time - start_time) / num_trials * 1000  # 毫秒

    # === 打印结果 ===
    print("\n" + "=" * 40)
    print(f"Results (Averaged over {num_trials} runs):")
    print(f"Device: {cfg.device}")
    print(f"Array Size: {cfg.M}x{cfg.N} = {cfg.M * cfg.N} elements")
    print("-" * 40)
    print(f"Traditional MVDR:  {mvdr_avg_time:.4f} ms / sample")
    print(f"Proposed DCVB:     {dcvb_avg_time:.4f} ms / sample")
    print("-" * 40)
    print(f"Speedup:           {mvdr_avg_time / dcvb_avg_time:.2f} x (倍)")
    print("=" * 40)


if __name__ == "__main__":
    run_speed_benchmark()