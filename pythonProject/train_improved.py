import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from data_gen_v2 import FdaMimoSimulatorV2
from model import ComplexBeamformerNet
from loss import mvdr_loss


def train_improved_model():
    """改进版训练：使用范围随机的距离差"""
    print("=" * 60)
    print("改进版训练：距离差 1-3km 随机化")
    print("=" * 60)
    
    # 1. 配置与初始化
    cfg = Config()
    cfg.M = 10
    cfg.N = 10
    cfg.epochs = 200
    cfg.JNR_range = [40, 50]

    sim = FdaMimoSimulatorV2(cfg)
    model = ComplexBeamformerNet(cfg).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    print("开始训练...")
    print("关键改进：干扰距离差从固定2km改为1-3km随机")
    
    loss_history = []

    # 2. 训练循环
    for epoch in range(cfg.epochs):
        model.train()

        # === 关键：使用 'random' 模式 ===
        X, a_tgt = sim.generate_batch(range_diff_mode='random')
        w_pred = model(X, a_tgt)

        loss = mvdr_loss(w_pred, X, a_tgt, lambda_c=0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {current_loss:.4e}")

    print("训练完成！")

    # 3. 保存模型
    torch.save(model.state_dict(), "fda_improved.pth")
    print(">>> 改进模型已保存: fda_improved.pth")

    # 4. 绘制 Loss 曲线（与原版对比）
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(range(cfg.epochs), loss_history, linewidth=2, color='darkgreen', 
                 label='Improved (Range-Diff Random)')
    
    # 如果存在原版的 loss 记录，可以加载对比
    # 这里简化处理
    
    plt.title("Training Convergence: Improved Model", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MVDR Loss (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=12)

    plt.savefig("loss_curve_improved.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Loss 曲线已保存: loss_curve_improved.png")


def compare_training_strategies():
    """对比两种训练策略的效果"""
    print("\n" + "=" * 60)
    print("对比实验：固定距离 vs 范围随机")
    print("=" * 60)
    
    cfg = Config()
    sim = FdaMimoSimulatorV2(cfg)
    
    # 加载两个模型
    model_fixed = ComplexBeamformerNet(cfg).to(cfg.device)
    model_fixed.load_state_dict(torch.load("fda_beamformer_final.pth"))
    model_fixed.eval()
    
    model_improved = ComplexBeamformerNet(cfg).to(cfg.device)
    model_improved.load_state_dict(torch.load("fda_improved.pth"))
    model_improved.eval()
    
    # 测试场景
    theta_tgt = 10.0
    r_tgt = 10000.0
    theta_jam = 10.0
    
    # 测试不同的距离差
    range_diffs = np.arange(0.5, 5.1, 0.5)
    
    suppression_fixed = []
    suppression_improved = []
    
    L = cfg.L
    v_s = sim.get_steering_vector(theta_tgt, r_tgt)
    a_tgt_test = v_s.unsqueeze(0)
    
    for delta_r in range_diffs:
        r_jam = r_tgt + delta_r * 1000
        v_j = sim.get_steering_vector(theta_jam, r_jam)
        
        # 生成测试数据
        sig = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        jam = (torch.randn(1, L, device=cfg.device) + 1j * torch.randn(1, L, device=cfg.device)) / 1.414
        noise = (torch.randn(sim.MN, L, device=cfg.device) + 1j * torch.randn(sim.MN, L, device=cfg.device)) / 1.414
        
        X_input = 1.0 * v_s.unsqueeze(1) * sig + 316.0 * v_j.unsqueeze(1) * jam + noise
        X_input = X_input / torch.max(torch.abs(X_input))
        X_input = X_input.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            w_fixed = model_fixed(X_input, a_tgt_test).squeeze(0)
            w_improved = model_improved(X_input, a_tgt_test).squeeze(0)
        
        # 计算抑制深度
        gain_fixed = torch.abs(torch.vdot(w_fixed, v_j)) ** 2
        gain_improved = torch.abs(torch.vdot(w_improved, v_j)) ** 2
        
        suppression_fixed.append(10 * np.log10(gain_fixed.item() + 1e-10))
        suppression_improved.append(10 * np.log10(gain_improved.item() + 1e-10))
    
    # 绘图对比
    plt.figure(figsize=(10, 6))
    
    plt.plot(range_diffs, suppression_fixed, 'b--o', linewidth=2, markersize=7, 
             label='Fixed Training (Δr = 2km)')
    plt.plot(range_diffs, suppression_improved, 'g-s', linewidth=2.5, markersize=7, 
             label='Range-Random Training (Δr = 1-3km)')
    
    # 标记训练区域
    plt.axvspan(1.0, 3.0, alpha=0.1, color='green', label='Training Range')
    
    plt.xlabel('Target-Jammer Range Difference Δr (km)', fontsize=12)
    plt.ylabel('Jamming Suppression (dB)', fontsize=12)
    plt.title('Training Strategy Comparison: Fixed vs Range-Random', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('comparison_training_strategy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n对比结果已保存: comparison_training_strategy.png")
    print("\n性能分析:")
    print(f"固定训练 - 平均抑制: {np.mean(suppression_fixed):.2f} dB, 方差: {np.var(suppression_fixed):.2f}")
    print(f"范围训练 - 平均抑制: {np.mean(suppression_improved):.2f} dB, 方差: {np.var(suppression_improved):.2f}")
    print("\n预期：绿线（改进版）应该更平滑，方差更小")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # 只运行对比实验（假设已经训练好了）
        compare_training_strategies()
    else:
        # 完整流程：训练 + 对比
        train_improved_model()
        print("\n等待 3 秒后开始对比实验...")
        import time
        time.sleep(3)
        compare_training_strategies()
