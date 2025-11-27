import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def get_mvdr_weights(X_input, a_tgt):
    """
    传统 MVDR 算法实现 (Sample Matrix Inversion)
    公式: w = (R^-1 * a) / (a^H * R^-1 * a)
    """
    # 1. 计算采样协方差矩阵 R (MN x MN)
    # X_input: (1, MN, L) -> squeeze -> (MN, L)
    X = X_input.squeeze(0)
    L = X.shape[1]

    # R = X * X^H / L
    R = torch.matmul(X, X.conj().T) / L

    # 2. 对角加载 (防止矩阵奇异，求逆崩掉)
    R = R + 1e-6 * torch.eye(R.shape[0], device=R.device)

    # 3. 矩阵求逆
    R_inv = torch.linalg.inv(R)

    # 4. 计算分子: R^-1 * a
    # a_tgt: (1, MN) -> transpose -> (MN, 1)
    a = a_tgt.T
    numerator = torch.matmul(R_inv, a)

    # 5. 计算分母: a^H * R^-1 * a (标量归一化系数)
    denominator = torch.matmul(a.conj().T, numerator)

    # 6. 得到权值
    w_mvdr = numerator / (denominator + 1e-10)
    return w_mvdr.squeeze()  # (MN,)


def calculate_beampattern(w, sim, target_theta, target_range, scan_ranges):
    gains = []
    for r in scan_ranges:
        a = sim.get_steering_vector(target_theta, r).to(w.device)
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


def run_comparison():
    # 1. 初始化
    cfg = Config()
    sim = FdaMimoSimulator(cfg)

    # 加载你的训练好的模型
    model = ComplexBeamformerNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load("fda_beamformer_final.pth"))  # 或者 latest
    model.eval()

    print("Generating scenario...")

    # 2. 设置测试场景
    theta_tgt = 10.0;
    r_tgt = 10000.0
    theta_jam = 10.0;
    r_jam = 12000.0

    # 3. 构造数据
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    L = cfg.L

    # 50dB 干扰
    sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414

    X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise
    X_input = X_input / torch.max(torch.abs(X_input))  # 归一化
    X_input = X_input.unsqueeze(0)

    a_tgt_test = v_s.unsqueeze(0)

    # 4. 方法一：你的 Deep Beamformer
    with torch.no_grad():
        w_deep = model(X_input, a_tgt_test).squeeze(0)

    # 5. 方法二：传统 MVDR
    w_mvdr = get_mvdr_weights(X_input, a_tgt_test)

    # 6. 绘图对比
    print("Plotting comparison...")
    scan_ranges = np.linspace(6000, 14000, 500)
    resp_deep = calculate_beampattern(w_deep, sim, theta_tgt, r_tgt, scan_ranges)
    resp_mvdr = calculate_beampattern(w_mvdr, sim, theta_tgt, r_tgt, scan_ranges)

    plt.figure(figsize=(10, 6))

    # MVDR (红虚线 - 理论最优)
    plt.plot(scan_ranges / 1000, resp_mvdr, 'r--', linewidth=2, label='Traditional MVDR (Matrix Inversion)')

    # Deep (蓝实线 - 你的方法)
    plt.plot(scan_ranges / 1000, resp_deep, 'b-', linewidth=2.5, alpha=0.8, label='Proposed DCVB (Deep Learning)')

    # 标记位置
    plt.axvline(x=r_tgt / 1000, color='green', linestyle=':', label='Target')
    plt.axvline(x=r_jam / 1000, color='orange', linestyle=':', label='Jammer')

    plt.title("Performance Comparison: Deep Learning vs. MVDR", fontsize=14)
    plt.xlabel("Range (km)", fontsize=12)
    plt.ylabel("Response (dB)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig("method_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_comparison()