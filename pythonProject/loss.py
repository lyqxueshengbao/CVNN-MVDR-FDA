import torch


def mvdr_loss(w_pred, X_in, a_tgt, lambda_c=0.0):
    """
    w_pred: 已经经过硬约束投影，Target Gain 恒等于 1
    Loss: 只需要最小化输出总功率 (Power Minimization)
    """
    w_H = w_pred.conj().unsqueeze(1)
    # y shape: (B, 1, L)
    y = torch.matmul(w_H, X_in).squeeze(1)

    # 最小化功率
    loss_power = torch.mean(torch.abs(y) ** 2)

    return loss_power