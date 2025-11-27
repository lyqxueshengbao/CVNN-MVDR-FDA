"""
一键运行所有对比实验
"""
import os
import sys

def run_experiment(script_name, description):
    """运行单个实验脚本"""
    print("\n" + "=" * 70)
    print(f"开始运行: {description}")
    print("=" * 70 + "\n")
    
    result = os.system(f"python {script_name}")
    
    if result == 0:
        print(f"\n✅ {description} 完成！\n")
    else:
        print(f"\n❌ {description} 失败！\n")
        return False
    return True


def main():
    print("=" * 70)
    print("FDA-MIMO 深度波束成形器 - 完整实验套件")
    print("=" * 70)
    
    # 检查模型文件是否存在
    if not os.path.exists("fda_beamformer_final.pth"):
        print("❌ 错误: 未找到训练好的模型 'fda_beamformer_final.pth'")
        print("请先运行 train.py 训练模型！")
        return
    
    experiments = [
        ("exp_jnr_curve.py", "实验 1: JNR vs 干扰抑制深度"),
        ("exp_ablation_projection.py", "实验 2: 投影层消融实验"),
        ("exp_range_difference.py", "实验 3: 距离差影响分析"),
        ("exp_snr_curve.py", "实验 4: SNR vs 输出 SINR"),
        ("exp_generalization.py", "实验 5: 泛化性测试"),
    ]
    
    success_count = 0
    
    for script, desc in experiments:
        if run_experiment(script, desc):
            success_count += 1
        else:
            response = input(f"\n实验失败，是否继续运行剩余实验？(y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "=" * 70)
    print(f"实验完成统计: {success_count}/{len(experiments)} 成功")
    print("=" * 70)
    
    print("\n生成的图表文件:")
    result_files = [
        "exp_jnr_curve.png",
        "exp_ablation_projection.png",
        "exp_range_difference.png",
        "exp_snr_curve.png",
        "exp_generalization.png"
    ]
    
    for f in result_files:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} (未生成)")
    
    print("\n所有实验结果可直接用于论文！")


if __name__ == "__main__":
    main()
