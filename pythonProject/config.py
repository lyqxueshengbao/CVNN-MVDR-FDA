import torch

class Config:
    # --- 物理参数 ---
    M = 10              # 发射阵元
    N = 10              # 接收阵元
    f0 = 10e9           # 10 GHz
    delta_f = 30e3      # 30 kHz
    c = 3e8
    d_t = c / (2 * f0)  # 0.5 lambda
    d_r = c / (2 * f0)  # 0.5 lambda
    L = 64              # 快拍数

    # --- 训练参数 ---
    batch_size = 64
    lr = 1e-4           # 学习率 (求稳)
    epochs = 200        # 训练轮数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 场景参数 ---
    # 高压训练模式，强迫网络挖坑
    JNR_range = [40, 50]
    SNR_range = [0, 10]