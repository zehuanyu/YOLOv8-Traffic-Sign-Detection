"""
简化版白天/夜间评估
使用图片文件名过滤
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--type', type=str, choices=['day', 'night', 'all'], default='all')
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"评估性能 - {args.type.upper()}")
    print("=" * 70)
    
    print(f"\n模型: {args.model}")
    
    # 由于ultralytics限制，先显示整体结果
    # 然后建议手动筛选结果
    
    model = YOLO(args.model)
    
    print(f"\n运行验证...")
    metrics = model.val(data=args.data, split='val')
    
    print("\n" + "=" * 70)
    print("整体结果")
    print("=" * 70)
    print(f"mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    print("\n" + "=" * 70)
    print("注意")
    print("=" * 70)
    print("Ultralytics不直接支持按文件名过滤验证。")
    print("建议：")
    print("1. 手动统计results中day/night图片的检测结果")
    print("2. 或创建独立的val_day/val_night文件夹")
    print("3. 或使用visualize_results.ipynb分析单张图片")


if __name__ == "__main__":
    main()

