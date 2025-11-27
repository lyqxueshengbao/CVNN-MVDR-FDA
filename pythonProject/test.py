import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet


def calculate_beampattern(w, sim, target_theta, target_range, scan_ranges):
    gains = []
    # 扫描距离维
    for r in scan_ranges:
        a = sim.get_steering_vector(target_theta, r).to(w.device)
        # Gain = |w^H * a|^2
        gain = torch.abs(torch.vdot(w, a)) ** 2
        gains.append(gain.item())
    return 10 * np.log10(np.array(gains) + 1e-10)


def evaluate():
    cfg = Config()
    sim = FdaMimoSimulator(cfg)
    model = ComplexBeamformerNet(cfg).to(cfg.device)

    model.load_state_dict(torch.load("fda_beamformer.pth"))
    model.eval()

    print("Model loaded. Generating test scenario...")

    # --- 测试场景 ---
    theta_tgt = 10.0
    r_tgt = 10000.0  # 目标在 10km

    theta_jam = 10.0  # 主瓣干扰 (角度相同)
    r_jam = 12000.0  # 干扰在 12km

    JNR_test = 50  # 50dB 强干扰
    SNR_test = 0  # 0dB 弱目标

    # --- 1. 构造环境信号 X ---
    # 这一步是为了生成那个“包含干扰”的波形
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    v_j = sim.get_steering_vector(theta_jam, r_jam)

    L = cfg.L
    sig_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
    jam_wave = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / np.sqrt(2)
    noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / np.sqrt(2)

    signal = 10 ** (SNR_test / 20) * v_s.unsqueeze(1) * sig_wave
    jamming = 10 ** (JNR_test / 20) * v_j.unsqueeze(1) * jam_wave

    X_input = signal + jamming + noise

    # 归一化 (必须！)
    X_input = X_input / (torch.max(torch.abs(X_input)) + 1e-8)
    X_input = X_input.unsqueeze(0)  # (1, MN, L)

    # --- 2. 构造导向矢量 a_tgt (寻宝图) ---
    # 告诉网络：虽然 X 里干扰很响，但请把波束对准 (theta_tgt, r_tgt)
    a_tgt_test = sim.get_steering_vector(theta_tgt, r_tgt).unsqueeze(0)  # (1, MN)

    # --- 3. 网络推理 ---
    with torch.no_grad():
        # 传入 X 和 a_tgt !!!
        w_pred = model(X_input, a_tgt_test).squeeze(0)  # (MN,)

    # --- 4. 绘图 ---
    print("Computing beampattern...")
    scan_ranges = np.linspace(r_tgt - 4000, r_tgt + 4000, 500)
    response_db = calculate_beampattern(w_pred, sim, theta_tgt, r_tgt, scan_ranges)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_ranges / 1000, response_db, linewidth=2, label='Deep Beamformer')

    plt.axvline(x=r_tgt / 1000, color='g', linestyle='--', label='Target (10km)')
    plt.axvline(x=r_jam / 1000, color='r', linestyle='--', label='Jammer (12km)')

    plt.title(f'FDA-MIMO Range Profile (M={cfg.M}, N={cfg.N})\nConstraint: Gain=1 at Target', fontsize=14)
    plt.xlabel('Range (km)', fontsize=12)
    plt.ylabel('Beamformer Response (dB)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    idx_jam = (np.abs(scan_ranges - r_jam)).argmin()
    print(f"Jammer Suppression Depth: {response_db[idx_jam]:.2f} dB")

    plt.savefig('beampattern_fixed.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    evaluate()