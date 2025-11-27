"""
调试脚本：理解为什么 MVDR 在少快拍时表现依然很好
"""
import torch
import numpy as np

print("=" * 50)
print("Debug: MVDR Behavior with Few Snapshots")
print("=" * 50)

MN = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for L in [1, 5, 10, 50, 100, 200]:
    # 模拟接收数据
    X = torch.randn(MN, L, dtype=torch.complex64, device=device)
    
    # 协方差矩阵
    R = torch.matmul(X, X.conj().T) / L
    
    # 检查秩
    rank = torch.linalg.matrix_rank(R).item()
    
    # 条件数 (越大越病态)
    try:
        cond = torch.linalg.cond(R).item()
    except:
        cond = float('inf')
    
    print(f"L={L:4d}: Rank={rank:4d}/{MN}, Condition Number={cond:.2e}")

print("\n" + "=" * 50)
print("Key Insight:")
print("=" * 50)
print("""
当 L < MN 时，协方差矩阵 R 是秩亏的 (Rank-Deficient)。
但 PyTorch 的 torch.linalg.inv() 在加了对角加载后依然能求逆。

真正的问题是：
1. 协方差矩阵估计的 *方差* 很大（不稳定）
2. 需要在多次随机试验中观察 *方差*，而不是看均值

MVDR 的干扰抑制深度可能在某些实现中被对角加载"救"了。
""")
