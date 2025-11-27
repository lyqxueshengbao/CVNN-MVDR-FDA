import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def get_response(model_path, sim, theta_tgt, r_tgt, scan_ranges, X, a_tgt):
    model = ComplexBeamformerNet(Config()).to(Config().device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        w = model(X, a_tgt).squeeze(0)

    gains = []
    for r in scan_ranges:
        a = sim.get_steering_vector(theta_tgt, r).to(w.device)
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


def run_comparison():
    cfg = Config()
    sim = FdaMimoSimulator(cfg)

    # 场景: 目标10km, 干扰12km
    theta_tgt = 10.0;
    r_tgt = 10000.0
    theta_jam = 10.0;
    r_jam = 12000.0

    # 构造测试数据
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)
    L = cfg.L

    # 50dB 强干扰
    sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
    noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414

    # 归一化输入
    X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise  # 316 ~= sqrt(10^5) for 50dB
    X_input = X_input / torch.max(torch.abs(X_input))
    X_input = X_input.unsqueeze(0)

    a_tgt_test = v_s.unsqueeze(0)

    # 扫描范围
    scan_ranges = np.linspace(6000, 14000, 500)

    print("Computing responses...")
    resp_init = get_response("fda_init.pth", sim, theta_tgt, r_tgt, scan_ranges, X_input, a_tgt_test)
    resp_final = get_response("fda_final.pth", sim, theta_tgt, r_tgt, scan_ranges, X_input, a_tgt_test)

    # 绘图
    plt.figure(figsize=(10, 6))

    # Epoch 0
    plt.plot(scan_ranges / 1000, resp_init, '--', color='gray', linewidth=2, label='Epoch 0 (Projection Only)')

    # Epoch 200
    plt.plot(scan_ranges / 1000, resp_final, '-', color='blue', linewidth=3, label='Epoch 200 (Learned)')

    plt.axvline(x=r_tgt / 1000, color='green', linestyle=':', linewidth=2, label='Target (10km)')
    plt.axvline(x=r_jam / 1000, color='red', linestyle=':', linewidth=2, label='Jammer (12km)')

    plt.title("Ablation Study: The Impact of Deep Learning", fontsize=14)
    plt.xlabel("Range (km)", fontsize=12)
    plt.ylabel("Response (dB)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_comparison()