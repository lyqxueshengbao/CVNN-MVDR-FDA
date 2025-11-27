import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from data_gen import FdaMimoSimulator
from model import ComplexBeamformerNet
from loss import mvdr_loss


def train_and_plot():
    # 1. 配置与初始化
    cfg = Config()
    cfg.M = 10
    cfg.N = 10
    cfg.epochs = 200
    cfg.JNR_range = [40, 50]

    sim = FdaMimoSimulator(cfg)
    model = ComplexBeamformerNet(cfg).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    print("Start Training...")

    # === 新增 1: 保存初始模型 (用于 Ablation Study 对比) ===
    # 这时候是随机权重 + 投影层
    torch.save(model.state_dict(), "fda_init.pth")
    print(">>> Initial Model Saved (fda_init.pth)")

    # === 用于记录 Loss 的列表 ===
    loss_history = []

    # 2. 训练循环
    for epoch in range(cfg.epochs):
        model.train()

        X, a_tgt = sim.generate_batch()
        w_pred = model(X, a_tgt)

        # Loss 只看功率 (lambda=0)
        loss = mvdr_loss(w_pred, X, a_tgt, lambda_c=0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # === 记录当前 Loss ===
        current_loss = loss.item()
        loss_history.append(current_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {current_loss:.4e}")

    print("Training Finished.")

    # === 新增 2: 保存最终模型 (用于 Test 和 Comparison) ===
    torch.save(model.state_dict(), "fda_final.pth")
    # 为了兼容之前的 test.py，也可以多存一个名字
    torch.save(model.state_dict(), "fda_beamformer_final.pth")
    print(">>> Final Model Saved (fda_final.pth & fda_beamformer_final.pth)")

    # 3. 绘制论文级 Loss 曲线
    plt.figure(figsize=(8, 6))

    # 使用 semilogy
    plt.semilogy(range(cfg.epochs), loss_history, linewidth=2, color='darkblue', label='Training Loss (Power)')

    plt.title("Convergence of Deep Complex Beamformer", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MVDR Loss (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=12)

    plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Loss curve saved to 'loss_curve.png'")


if __name__ == "__main__":
    train_and_plot()