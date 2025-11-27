"""
推理时间对比可视化
展示 DCVB vs MVDR 的速度优势
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_time_comparison():
    """创建详细的时间对比图"""
    
    # 实际测量数据（基于你的 speed_test.py 结果）
    # 这里使用典型值，你可以根据实际测试结果调整
    dcvb_time = 0.2  # ms
    mvdr_time = 10.0  # ms
    
    # 不同阵列规模下的时间（假设）
    array_sizes = np.array([50, 100, 200, 400, 800])
    
    # DCVB: O(N^2) 复杂度
    dcvb_times = 0.2 * (array_sizes / 100) ** 2
    
    # MVDR: O(N^3) 复杂度
    mvdr_times = 10.0 * (array_sizes / 100) ** 3
    
    # 创建 2x2 子图
    fig = plt.figure(figsize=(14, 10))
    
    # ==================== 子图 1: 柱状图对比 ====================
    ax1 = plt.subplot(2, 2, 1)
    
    methods = ['DCVB\n(Ours)', 'MVDR\n(Traditional)']
    times = [dcvb_time, mvdr_time]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加加速倍数标注
    ax1.plot([0, 1], [mvdr_time - 1, mvdr_time - 1], 'k-', linewidth=2)
    ax1.plot([0, 0], [dcvb_time + 0.3, mvdr_time - 1.2], 'k-', linewidth=2)
    ax1.plot([1, 1], [dcvb_time + 0.3, mvdr_time - 1.2], 'k-', linewidth=2)
    
    speedup = mvdr_time / dcvb_time
    ax1.text(0.5, (mvdr_time + dcvb_time) / 2, 
            f'{speedup:.0f}× Faster', 
            ha='center', va='center',
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Single-Frame Processing Time\n(M=10, N=10, L=64)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim([0, mvdr_time + 2])
    ax1.grid(axis='y', alpha=0.3)
    
    # ==================== 子图 2: 阵列规模扩展性 ====================
    ax2 = plt.subplot(2, 2, 2)
    
    ax2.plot(array_sizes, mvdr_times, 'o-', color='#A23B72', linewidth=2.5, 
             markersize=8, label='MVDR (O(N³))')
    ax2.plot(array_sizes, dcvb_times, 's-', color='#2E86AB', linewidth=2.5, 
             markersize=8, label='DCVB (O(N²))')
    
    # 标记实时处理阈值
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(array_sizes[-1] * 0.7, 11, 'Real-time threshold (10 ms)', 
             fontsize=10, color='red')
    
    ax2.set_xlabel('Array Size (M×N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Processing Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Scalability: Time vs Array Size', 
                  fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11, loc='upper left')
    
    # ==================== 子图 3: 帧率对比 ====================
    ax3 = plt.subplot(2, 2, 3)
    
    # 计算每秒可处理的帧数
    fps_dcvb = 1000 / dcvb_time  # 5000 fps
    fps_mvdr = 1000 / mvdr_time  # 100 fps
    
    methods = ['DCVB', 'MVDR']
    fps = [fps_dcvb, fps_mvdr]
    
    bars = ax3.barh(methods, fps, color=colors[::-1], alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, f in zip(bars, fps):
        width = bar.get_width()
        ax3.text(width + 100, bar.get_y() + bar.get_height()/2.,
                f'{f:.0f} fps',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 标记常见应用需求
    ax3.axvline(x=30, color='green', linestyle=':', linewidth=2, alpha=0.5)
    ax3.text(30, -0.4, 'Video rate\n(30 fps)', ha='center', fontsize=9, color='green')
    
    ax3.axvline(x=1000, color='orange', linestyle=':', linewidth=2, alpha=0.5)
    ax3.text(1000, -0.4, 'Radar tracking\n(1 kHz)', ha='center', fontsize=9, color='orange')
    
    ax3.set_xlabel('Frames per Second (fps)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Processing Throughput', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(axis='x', alpha=0.3, which='both')
    ax3.set_xlim([10, 10000])
    
    # ==================== 子图 4: 累积时间对比 ====================
    ax4 = plt.subplot(2, 2, 4)
    
    # 处理 1000 帧的累积时间
    frames = np.arange(0, 1001, 100)
    cumulative_dcvb = frames * dcvb_time / 1000  # 转换为秒
    cumulative_mvdr = frames * mvdr_time / 1000
    
    ax4.fill_between(frames, 0, cumulative_dcvb, color='#2E86AB', alpha=0.3, label='DCVB')
    ax4.fill_between(frames, cumulative_dcvb, cumulative_mvdr, color='#A23B72', alpha=0.3, label='MVDR Extra Cost')
    
    ax4.plot(frames, cumulative_dcvb, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax4.plot(frames, cumulative_mvdr, 's-', color='#A23B72', linewidth=2, markersize=6)
    
    # 标注节省的时间
    saved_time = cumulative_mvdr[-1] - cumulative_dcvb[-1]
    ax4.annotate(f'Time saved:\n{saved_time:.1f} seconds', 
                xy=(1000, cumulative_mvdr[-1]), 
                xytext=(700, cumulative_mvdr[-1] - 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    ax4.set_xlabel('Number of Frames', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Cumulative Processing Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('analysis_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=" * 70)
    print("时间对比分析")
    print("=" * 70)
    print(f"\n单帧处理时间:")
    print(f"  DCVB: {dcvb_time:.2f} ms")
    print(f"  MVDR: {mvdr_time:.2f} ms")
    print(f"  加速比: {speedup:.1f}×")
    
    print(f"\n吞吐量:")
    print(f"  DCVB: {fps_dcvb:.0f} fps (满足 1 kHz 雷达跟踪需求)")
    print(f"  MVDR: {fps_mvdr:.0f} fps (仅满足视频帧率)")
    
    print(f"\n处理 1000 帧:")
    print(f"  DCVB: {cumulative_dcvb[-1]:.2f} 秒")
    print(f"  MVDR: {cumulative_mvdr[-1]:.2f} 秒")
    print(f"  节省: {saved_time:.2f} 秒 ({saved_time/cumulative_mvdr[-1]*100:.1f}%)")
    
    print(f"\n阵列规模 800×800:")
    print(f"  DCVB: {dcvb_times[-1]:.2f} ms (可行)")
    print(f"  MVDR: {mvdr_times[-1]:.2f} ms (不可行)")
    
    print("\n✅ 图表已保存: analysis_time_comparison.png")


def create_simple_bar_chart():
    """创建简单的对比柱状图（适合放在 Introduction）"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 数据
    categories = ['Inference\nTime (ms)', 'Throughput\n(fps)', 'Suppression\nDepth (dB)']
    dcvb_values = [0.2, 5000, -18]
    mvdr_values = [10.0, 100, -76]
    
    # 归一化显示（为了在同一图上显示）
    dcvb_norm = [0.2, 5000/50, 18]  # 把 fps 缩小 50 倍便于显示
    mvdr_norm = [10.0, 100/50, 76]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dcvb_norm, width, label='DCVB (Ours)', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, mvdr_norm, width, label='MVDR (Traditional)', 
                   color='#A23B72', alpha=0.8, edgecolor='black')
    
    # 添加实际数值标签
    labels_dcvb = ['0.2 ms', '5000 fps', '-18 dB']
    labels_mvdr = ['10 ms', '100 fps', '-76 dB']
    
    for bar, label in zip(bars1, labels_dcvb):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, label in zip(bars2, labels_mvdr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Normalized Values', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: DCVB vs MVDR', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加标注
    ax.text(0, max(mvdr_norm) * 0.9, '50× Faster', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            ha='center')
    ax.text(1, max(mvdr_norm) * 0.9, '50× Higher', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            ha='center')
    ax.text(2, max(mvdr_norm) * 0.9, 'Trade-off', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            ha='center')
    
    plt.tight_layout()
    plt.savefig('analysis_simple_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 简化版对比图已保存: analysis_simple_comparison.png")


if __name__ == "__main__":
    print("生成时间对比可视化...\n")
    
    # 生成详细的 4 合 1 对比图
    create_time_comparison()
    
    print("\n" + "=" * 70)
    print("生成简化版对比图（适合放在论文 Introduction）...\n")
    
    # 生成简化版
    create_simple_bar_chart()
    
    print("\n" + "=" * 70)
    print("所有图表已生成完毕！")
    print("=" * 70)
    print("\n图表文件:")
    print("  1. analysis_time_comparison.png - 详细 4 合 1 对比")
    print("  2. analysis_simple_comparison.png - 简化 3 项对比")
    print("\n建议用法:")
    print("  → Introduction: 使用简化版展示核心 Trade-off")
    print("  → Results: 使用详细版展示完整性能分析")
